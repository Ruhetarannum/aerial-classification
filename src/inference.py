"""
inference.py

- Python 3.10 compatible
- CLASSIFICATION-only by default (no cropping)
- Optional YOLOv5 detection via torch.hub (reads results.xyxy only; DOES NOT use r.boxes)
- Classifier always runs on the full original image (no per-box cropping)
- Annotates image with boxes (if detection enabled) and draws classification label top-left
"""

from __future__ import annotations
from pathlib import Path
import io
import json
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision.transforms as T
import torchvision.models as models


def _log(msg: str) -> None:
    print(f"[inference] {msg}")


class DetectorClassifier:
    def __init__(self,
                 models_dir: Union[str, Path] = "models",
                 device: str = "cpu",
                 use_yolov5: bool = False,
                 yolo_conf_thresh: float = 0.25) -> None:
        """
        models_dir: path to folder containing your model file(s)
        device: 'cpu' or 'cuda' (cuda only used if available)
        use_yolov5: if True, attempt to load a YOLOv5 model via torch.hub from models/best.pt
                   (this uses torch.hub and expects YOLOv5-style results.xyxy).
        yolo_conf_thresh: default confidence threshold used for detection display.
        """
        self.models_dir = Path(models_dir)
        self.device = "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.use_yolov5 = bool(use_yolov5)
        self.yolo_conf_thresh = float(yolo_conf_thresh)

        # classifier files
        self.classification_path = self.models_dir / "classification_model_finetuned.pt"
        self.class_names_path = self.models_dir / "class_names.json"

        # yolov5 detection file expected (if using detection)
        self.yolov5_path = self.models_dir / "best.pt"

        # model holders
        self.yolo_model: Optional[Any] = None  # torch.hub model if loaded
        self.classifier: Optional[torch.nn.Module] = None
        self.class_names: Optional[List[str]] = None

        # transforms for classifier
        self.class_input_size = 224
        self.transform = T.Compose([
            T.Resize((self.class_input_size, self.class_input_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

        # load class names and classifier
        self._load_class_names()
        self._load_classifier()

        # load yolov5 via torch.hub only if user asked
        if self.use_yolov5:
            self._load_yolov5_hub()

    # -------------------------
    # Loading helpers
    # -------------------------
    def _load_class_names(self) -> None:
        if self.class_names_path.exists():
            try:
                with open(self.class_names_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list) and all(isinstance(x, str) for x in data):
                    self.class_names = data
                else:
                    _log("class_names.json is present but not a list[str]; ignoring.")
                    self.class_names = None
            except Exception as e:
                _log(f"Failed to load class_names.json: {e}")
                self.class_names = None
        else:
            self.class_names = None

    def build_classification_model(self, num_classes: int = 2) -> torch.nn.Module:
        """Default architecture (adjust if your checkpoint used custom model)."""
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model

    def _load_classifier(self) -> None:
        """
        Robust loader for the classification checkpoint.

        - Accepts checkpoints saved as a full nn.Module OR as state_dict-like mappings.
        - If state_dict present, inspects 'fc.weight' to infer num_classes and
        constructs the model with the correct head size before loading weights.
        - Strips 'module.' prefix if present in keys.
        - Writes diagnostic logs to help debug shape mismatches / bias issues.
        """
        if not self.classification_path.exists():
            _log(f"Classification checkpoint not found at {self.classification_path}. Classifier disabled.")
            self.classifier = None
            return

        try:
            ckpt = torch.load(self.classification_path, map_location="cpu")
        except Exception as e:
            _log(f"Failed to load classification checkpoint: {e}")
            traceback.print_exc()
            self.classifier = None
            return

        # If the checkpoint is a full model object, use it directly (best case)
        if isinstance(ckpt, torch.nn.Module):
            try:
                model = ckpt
                _log("Loaded full model object from checkpoint.")
                model.to(self.device)
                model.eval()
                self.classifier = model
                return
            except Exception as e:
                _log(f"Failed to use loaded model object directly: {e}")
                traceback.print_exc()
                self.classifier = None
                return

        # Normalize to a state_dict mapping if possible
        state_dict: Optional[Dict[str, Any]] = None
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt

        if not isinstance(state_dict, dict):
            _log("Checkpoint did not contain a usable state_dict and wasn't a model object.")
            self.classifier = None
            return

        # -----------------------
        # Inspect checkpoint keys (diagnostics)
        # -----------------------
        try:
            keys_preview = list(state_dict.keys())[:40]
            _log(f"[loader] state_dict keys preview (first 40): {keys_preview}")
        except Exception:
            pass

        # -----------------------
        # Decide num_classes robustly (DEFENSIVE)
        # -----------------------
        ckpt_num_classes = None
        try:
            # If fc.weight present, infer number of classes from its first dimension
            if 'fc.weight' in state_dict:
                ckpt_fc_w = state_dict['fc.weight']
                try:
                    # Case 1: torch tensor or anything with a proper .shape
                    if hasattr(ckpt_fc_w, "shape") and ckpt_fc_w.shape is not None:
                        shape0 = ckpt_fc_w.shape[0]
                    else:
                        # Case 2: fallback to numpy conversion (works for list/tuple)
                        arr = np.array(ckpt_fc_w)
                        if arr.ndim > 0:
                            shape0 = arr.shape[0]
                        else:
                            raise ValueError("fc.weight has no valid first dimension")
                    
                    ckpt_num_classes = int(shape0)
                    _log(f"[loader] checkpoint fc.weight shape indicates {ckpt_num_classes} classes.")

                except Exception as e:
                    _log(f"[loader] could not infer num_classes from fc.weight: {e}")
                    ckpt_num_classes = None
        except Exception:
            ckpt_num_classes = None

        names_num = len(self.class_names) if (self.class_names is not None and len(self.class_names) > 0) else None

        # Choose num_classes with sensible fallbacks
        if names_num is not None and ckpt_num_classes is not None:
            if names_num != ckpt_num_classes:
                _log(f"[loader] WARNING: class_names length ({names_num}) != checkpoint num_classes ({ckpt_num_classes}).")
                # prefer checkpoint so fc weights load; but keep user informed
                chosen_num = ckpt_num_classes
            else:
                chosen_num = names_num
        elif ckpt_num_classes is not None:
            chosen_num = ckpt_num_classes
        elif names_num is not None:
            chosen_num = names_num
        else:
            chosen_num = 2
            _log(f"[loader] WARNING: could not infer num_classes from checkpoint or class_names; defaulting to {chosen_num}")

        # Ensure chosen_num is a valid positive int
        try:
            chosen_num = int(chosen_num)
            if chosen_num <= 0:
                raise ValueError("num_classes must be > 0")
        except Exception:
            _log(f"[loader] Invalid num_classes inferred ({chosen_num}); defaulting to 2")
            chosen_num = 2

        _log(f"[loader] constructing model with num_classes={chosen_num}")

        # -----------------------
        # Build model with decided head size
        # -----------------------
        try:
            model = self.build_classification_model(num_classes=chosen_num)
        except Exception as e:
            _log(f"[loader] Failed to build classification model: {e}")
            traceback.print_exc()
            self.classifier = None
            return

        # -----------------------
        # Normalize state_dict keys and attempt to load
        # -----------------------
        new_state: Dict[str, Any] = {}
        for k, v in state_dict.items():
            nk = k[len("module."):] if k.startswith("module.") else k
            new_state[nk] = v

        # Report possible fc keys for diagnostics
        try:
            possible_fc_keys = [k for k in new_state.keys() if 'fc' in k or 'classifier' in k]
            _log(f"[loader] possible fc keys in checkpoint: {possible_fc_keys}")
            for k in possible_fc_keys:
                try:
                    s = tuple(new_state[k].shape)
                    _log(f"[loader] {k} -> {s}")
                except Exception:
                    _log(f"[loader] {k} -> (couldn't read shape)")
        except Exception:
            pass

        # Load weights (non-strict to tolerate small naming mismatches)
        try:
            model.load_state_dict(new_state, strict=False)
            _log("Loaded state_dict into model (strict=False).")
        except Exception as e:
            _log(f"Partial/failed load of state_dict: {e}")
            traceback.print_exc()

        # Move model to device and set eval mode
        try:
            model.to(self.device)
            model.eval()
        except Exception as e:
            _log(f"[loader] Failed to move model to device or set eval: {e}")
            traceback.print_exc()

        self.classifier = model

        # Final diagnostic: print fc weight/bias stats if possible
        try:
            fc = getattr(self.classifier, 'fc')
            w = fc.weight.data.cpu().numpy()
            b = fc.bias.data.cpu().numpy()
            _log(f"[loader] final fc shape = {w.shape}; bias = {b.tolist()}")
            _log(f"[loader] final fc weight mean={float(w.mean()):.6f}, std={float(w.std()):.6f}")
        except Exception:
            _log("[loader] could not inspect final fc layer stats.")



    def _load_yolov5_hub(self) -> None:
        """
        Load YOLOv5 via torch.hub using 'ultralytics/yolov5' repo and a local 'best.pt' weights file.
        This loader uses torch.hub and will only be attempted if use_yolov5=True.
        The code will not use ultralytics or access r.boxes; it reads results.xyxy only.
        """
        if not self.yolov5_path.exists():
            _log(f"YOLOv5 weights not found at {self.yolov5_path}; detection will not be available.")
            self.yolo_model = None
            return

        try:
            _log("Loading YOLOv5 via torch.hub (ultralytics/yolov5)...")
            # trust_repo=True allows loading local custom weights file
            self.yolo_model = torch.hub.load("ultralytics/yolov5", "custom", str(self.yolov5_path), trust_repo=True)  # type: ignore
            _log("YOLOv5 (torch.hub) loaded successfully.")
            # set device for hub model if supported
            try:
                if self.device == "cuda":
                    self.yolo_model.cuda()
                else:
                    self.yolo_model.cpu()
            except Exception:
                # some hub wrappers ignore .cpu()/.cuda(); it's fine
                pass
        except Exception as e:
            _log(f"Failed to load YOLOv5 via torch.hub: {e}")
            traceback.print_exc()
            self.yolo_model = None

    # -------------------------
    # Helper: parse YOLOv5 results.xyxy output
    # -------------------------
    @staticmethod
    def _parse_yolov5_xyxy(yres: Any, conf_thresh: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Accepts results returned by a YOLOv5 hub model (model(image) or model.predict(...)).
        We only read results.xyxy and do NOT touch r.boxes.

        Returns:
            xyxy: np.ndarray shape (N,4)
            confs: np.ndarray shape (N,)
            clsids: np.ndarray shape (N,)
        If nothing parseable -> three empty arrays.
        """
        try:
            # yres may be a Results wrapper or a list-like; normalize access
            obj = yres[0] if isinstance(yres, (list, tuple)) and len(yres) > 0 else yres
            if hasattr(obj, "xyxy"):
                container = getattr(obj, "xyxy")
                # container might be list-like where first element is tensor/array
                arr0 = container[0] if isinstance(container, (list, tuple)) and len(container) > 0 else container
                # convert to numpy if tensor-like
                if isinstance(arr0, torch.Tensor):
                    arr = arr0.detach().cpu().numpy()
                else:
                    arr = np.array(arr0)
                if arr.ndim == 2 and arr.shape[1] >= 6:
                    xyxy = arr[:, :4]
                    confs = arr[:, 4]
                    clsids = arr[:, 5].astype(int)
                    # apply conf threshold (optional)
                    if conf_thresh > 0:
                        mask = confs >= conf_thresh
                        xyxy = xyxy[mask]
                        confs = confs[mask]
                        clsids = clsids[mask]
                    return xyxy, confs, clsids
        except Exception:
            traceback.print_exc()
        return np.array([]), np.array([]), np.array([])

    # -------------------------
    # Classification (full image)
    # -------------------------
    def _classify_full_image(self, pil_img: Image.Image) -> Tuple[str, float]:
        """
        Robust full-image classifier:
        - ensures classifier is present and callable
        - converts input to torch.Tensor if needed
        - uses tensor[None,...] to add batch dim (no unsqueeze())
        - moves input to same device as model
        - logs helpful diagnostics on failure and re-raises
        """
        if self.classifier is None:
            raise RuntimeError("Classifier not loaded (self.classifier is None). Check model file and loader logs.")

        # ensure model is in eval mode
        try:
            self.classifier.eval()
        except Exception:
            pass

        # apply transforms (should give a torch.Tensor)
        transformed = self.transform(pil_img)

        # Make sure transformed is a torch.Tensor
        if not isinstance(transformed, torch.Tensor):
            try:
                transformed = torch.as_tensor(np.array(transformed)).permute(2, 0, 1).float()
            except Exception as e:
                _log(f"[classify] failed converting transform output to tensor: {e}")
                raise

        # add batch dim WITHOUT unsqueeze
        inp = transformed[None, ...]  # shape (1,C,H,W)

        # Move input to device and correct dtype
        try:
            inp = inp.to(dtype=torch.float32, device=self.device)
        except Exception as e:
            _log(f"[classify] failed to move input to device {self.device}: {e}")
            # fallback: send input to cpu (best-effort)
            try:
                inp = inp.to(dtype=torch.float32, device="cpu")
                _log("[classify] moved input to cpu as fallback")
            except Exception:
                raise

        # Sanity checks before call
        if not callable(self.classifier):
            # Dump diagnostics
            _log(f"[classify] classifier is not callable: type={type(self.classifier)}")
            raise RuntimeError("Classifier object is not callable. Inspect the classifier variable.")

        # Try model call and capture exceptions with helpful diagnostics
        try:
            with torch.no_grad():
                out = self.classifier(inp)  # <-- the line that failed for you previously
        except Exception as e:
            # Collect diagnostics
            try:
                model_type = type(self.classifier)
                model_device = next(self.classifier.parameters()).device if any(True for _ in self.classifier.parameters()) else torch.device("cpu")
                _log(f"[classify] model_type={model_type}, model_device={model_device}")
            except Exception:
                _log("[classify] could not read model parameters for device info")

            _log(f"[classify] Input info: type={type(inp)}, dtype={inp.dtype}, device={inp.device}, shape={tuple(inp.shape)}")
            # try to print a few tensor values safely
            try:
                sample = inp.cpu().numpy().flatten()[:10].tolist()
                _log(f"[classify] input sample values (first 10): {sample}")
            except Exception:
                pass

            _log(f"[classify] model call raised exception: {e}")
            raise

        # Convert outputs to numpy/probs safely
        try:
            logits = out.detach().cpu().numpy()
            if logits.ndim == 2 and logits.shape[0] == 1:
                logits = logits[0]
            exps = np.exp(logits - np.max(logits))
            probs = exps / (exps.sum() + 1e-12)
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])
            label = self.class_names[top_idx] if (self.class_names and top_idx < len(self.class_names)) else str(top_idx)
            return label, top_prob
        except Exception as e:
            _log(f"[classify] failed converting model output to probs: {e}")
            raise


    # -------------------------
    # Main public API
    # -------------------------
    def detect_and_classify(self, image: Union[str, Path, bytes], conf_thresh: Optional[float] = None
                            ) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """
        Classify full image (always). Optionally also run YOLOv5 detection (only if use_yolov5=True and model loaded).
        - image: path or bytes
        - conf_thresh: override default detection confidence threshold for parsing/display
        Returns:
            annotated PIL Image, results_list
        results_list:
            - If detection disabled: [{"class_label":..., "class_conf":...}]
            - If detection enabled and boxes found: list of dicts:
                {"box":[x1,y1,x2,y2], "det_conf":..., "det_class":..., "class_label":..., "class_conf":...}
        """
        # load PIL image
        if isinstance(image, (bytes, bytearray)):
            pil = Image.open(io.BytesIO(bytes(image))).convert("RGB")
        else:
            pil = Image.open(str(image)).convert("RGB")

        # classification on full image
        class_label, class_conf = self._classify_full_image(pil)

        results_list: List[Dict[str, Any]] = []
        annotated = pil.copy()
        draw = ImageDraw.Draw(annotated)

        # detection block (YOLOv5 via torch.hub) - parse only results.xyxy
        if self.use_yolov5 and self.yolo_model is not None:
            try:
                # Many hub models accept numpy array as input
                raw = self.yolo_model(np.array(pil))
                ct = self._parse_yolov5_xyxy(raw, conf_thresh=(conf_thresh if conf_thresh is not None else self.yolo_conf_thresh))
                xyxy, det_confs, det_clsids = ct
                if xyxy.size == 0:
                    # no detections -> return classification-only result
                    results_list.append({"class_label": class_label, "class_conf": class_conf})
                else:
                    # annotate boxes and add combined results (classification on full image)
                    for i in range(xyxy.shape[0]):
                        box = xyxy[i].tolist()
                        det_conf = float(det_confs[i]) if i < len(det_confs) else 0.0
                        det_cls = int(det_clsids[i]) if i < len(det_clsids) else -1
                        results_list.append({
                            "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                            "det_conf": det_conf,
                            "det_class": det_cls,
                            "class_label": class_label,
                            "class_conf": class_conf
                        })
                        # draw rectangle
                        x1, y1, x2, y2 = map(int, box)
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            except Exception as e:
                _log(f"YOLOv5 detection run failed: {e}")
                traceback.print_exc()
                results_list.append({"class_label": class_label, "class_conf": class_conf})
        else:
            # detection not enabled or not available => classification-only
            results_list.append({"class_label": class_label, "class_conf": class_conf})

        # draw full-image classification label at top-left
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=16)
        except Exception:
            font = ImageFont.load_default()

        label_text = f"{class_label} {class_conf:.2f}"
        # use getbbox to compute bounding box, avoids textsize()
        try:
            bbox = font.getbbox(label_text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = (len(label_text) * 7, 16)

        draw.rectangle([0, 0, text_w + 6, text_h + 6], fill="red")
        draw.text((3, 3), label_text, fill="white", font=font)

        return annotated, results_list
