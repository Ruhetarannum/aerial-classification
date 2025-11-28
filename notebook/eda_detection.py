# notebooks/eda_detection_onegrid.py
"""
YOLO-only detection EDA — single-frame grid display.

What it does:
 - Detects image folders under DATA_DIR (handles train/valid/test or train/images etc.)
 - Finds best-effort matching label dir (labels/ sibling or root/labels)
 - Samples images from train/val and overlays YOLO boxes (if .txt labels exist)
 - Samples images from test (unlabeled) and shows them in a single grid
 - All images for a sample are shown in one matplotlib figure (no multiple popups)

Edit DATA_DIR below to your dataset folder and run:
    python notebooks/eda_detection_onegrid.py
"""

import os
import random
import math
from collections import defaultdict
from PIL import Image, ImageDraw, UnidentifiedImageError
import matplotlib.pyplot as plt

# ====================== EDIT THIS ======================
DATA_DIR = r"C:\Users\ruhet\Labmentix_Projects\aerial-classsification\data_unzipped\object_detection_Dataset"
# Max images to show per grid
MAX_IMAGES_PER_GRID = 24
# Image extensions to consider
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
# =======================================================

def safe_listdir(p):
    try:
        return sorted(os.listdir(p))
    except Exception:
        return []

def contains_images(p):
    try:
        for f in safe_listdir(p):
            if f.lower().endswith(IMG_EXTS):
                return True
    except Exception:
        pass
    return False

def find_image_dirs_under(root):
    """
    Find directories containing images:
      - root itself
      - direct children (e.g., train)
      - one level deeper in each child (e.g., train/images or train/classA)
    """
    found = []
    if contains_images(root):
        found.append(root)
    for entry in safe_listdir(root):
        p = os.path.join(root, entry)
        if not os.path.isdir(p):
            continue
        if contains_images(p):
            found.append(p)
            continue
        # check one level deeper
        for sub in safe_listdir(p):
            sp = os.path.join(p, sub)
            if os.path.isdir(sp) and contains_images(sp):
                found.append(sp)
    # dedupe
    return list(dict.fromkeys(found))

def find_label_dir_for_images(images_dir, root):
    """
    Try finding label dir for a given images_dir:
     - sibling 'labels' under same parent (or labels/<sub>)
     - root/labels or root/labels/<sub>
     - labels in same images_dir (txt alongside images)
     - any top-level labels* folder under root
    """
    img_parent = os.path.dirname(images_dir)
    img_basename = os.path.basename(images_dir)

    # sibling parent/labels or parent/labels/<basename>
    cand = os.path.join(img_parent, "labels")
    if os.path.isdir(cand):
        cand_sub = os.path.join(cand, img_basename)
        if os.path.isdir(cand_sub):
            return cand_sub
        return cand

    # root/labels or root/labels/<basename>
    root_labels = os.path.join(root, "labels")
    if os.path.isdir(root_labels):
        cand_sub = os.path.join(root_labels, img_basename)
        if os.path.isdir(cand_sub):
            return cand_sub
        return root_labels

    # labels alongside images (image.jpg and image.txt in same folder)
    for f in safe_listdir(images_dir)[:200]:
        if f.lower().endswith(IMG_EXTS):
            base = os.path.splitext(f)[0]
            txt = os.path.join(images_dir, base + ".txt")
            if os.path.exists(txt):
                return images_dir

    # fallback: any top-level dir starting with 'labels'
    for entry in safe_listdir(root):
        if entry.lower().startswith("labels"):
            p = os.path.join(root, entry)
            if os.path.isdir(p):
                return p

    return None

def read_yolo_labels(label_path):
    """
    Reads YOLO .txt labels and returns list of (class_idx, cx, cy, w, h)
    """
    boxes = []
    try:
        with open(label_path, 'r') as f:
            for ln in f:
                parts = ln.strip().split()
                if len(parts) >= 5:
                    try:
                        cls = int(float(parts[0]))
                        cx, cy, bw, bh = map(float, parts[1:5])
                        boxes.append((cls, cx, cy, bw, bh))
                    except Exception:
                        continue
    except Exception:
        pass
    return boxes

def draw_yolo_overlay_image(img_path, label_path, max_dim=800):
    """
    Return a PIL.Image object with YOLO boxes drawn (if label_path exists).
    Resizes large images for reasonable display but keeps aspect ratio.
    """
    try:
        im = Image.open(img_path).convert("RGB")
    except UnidentifiedImageError:
        # return a small "bad image" placeholder
        im = Image.new("RGB", (224,224), color=(50,50,50))
        draw = ImageDraw.Draw(im)
        draw.text((10,10), "CORRUPT", fill="white")
        return im

    # optional resize for display if very large
    w,h = im.size
    max_side = max_dim
    if max(w,h) > max_side:
        scale = max_side / max(w,h)
        im = im.resize((int(w*scale), int(h*scale)))
        w,h = im.size

    if label_path and os.path.exists(label_path):
        boxes = read_yolo_labels(label_path)
        if boxes:
            draw = ImageDraw.Draw(im)
            for (cls, cx, cy, bw, bh) in boxes:
                x1 = (cx - bw/2) * w
                x2 = (cx + bw/2) * w
                y1 = (cy - bh/2) * h
                y2 = (cy + bh/2) * h
                draw.rectangle([x1, y1, x2, y2], outline="red", width=max(1, int(max(w,h)/200)))
                # optionally draw class index
                draw.text((x1+2, max(0,y1-12)), str(cls), fill="white")
    return im

def show_images_grid_from_pil(images, title="Image Grid", cols=6):
    """
    images: list of PIL.Image objects
    Shows them in one single matplotlib figure grid.
    """
    if not images:
        print("No images to display.")
        return
    n = len(images)
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    fig_w = cols * 3
    fig_h = rows * 3
    plt.figure(figsize=(fig_w, fig_h))
    plt.suptitle(title, fontsize=14)
    for i, im in enumerate(images, 1):
        plt.subplot(rows, cols, i)
        plt.imshow(im)
        plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

def sample_and_show(images_dir, label_dir, label_dir_used, tag):
    """
    Sample images from images_dir (with labels if available) and show single-grid.
    label_dir_used: path to label dir (or None)
    tag: descriptive name like 'train' or 'val' or 'test'
    """
    imgs = [f for f in safe_listdir(images_dir) if f.lower().endswith(IMG_EXTS)]
    if not imgs:
        print(f"No images found in {images_dir}")
        return
    # sample up to MAX_IMAGES_PER_GRID
    sample_n = min(MAX_IMAGES_PER_GRID, len(imgs))
    sample_files = random.sample(imgs, sample_n)
    pil_list = []
    for fn in sample_files:
        img_path = os.path.join(images_dir, fn)
        label_path = None
        if label_dir_used:
            # if label_dir_used is same folder as images_dir, or label_dir_used contains subfolders matching images_dir base
            if os.path.isdir(label_dir_used):
                # if label dir matches by subname
                candidate = os.path.join(label_dir_used, os.path.splitext(fn)[0] + ".txt")
                if os.path.exists(candidate):
                    label_path = candidate
                else:
                    # maybe label_dir_used is same-level with image names
                    candidate2 = os.path.join(label_dir_used, os.path.splitext(fn)[0] + ".txt")
                    if os.path.exists(candidate2):
                        label_path = candidate2
            # fallback: check images_dir for txt next to image
            txt_in_same = os.path.join(images_dir, os.path.splitext(fn)[0] + ".txt")
            if os.path.exists(txt_in_same):
                label_path = txt_in_same
        else:
            # check for txt in images_dir
            txt_in_same = os.path.join(images_dir, os.path.splitext(fn)[0] + ".txt")
            if os.path.exists(txt_in_same):
                label_path = txt_in_same

        pil = draw_yolo_overlay_image(img_path, label_path)
        pil_list.append(pil)

    show_images_grid_from_pil(pil_list, title=f"{tag} samples from {os.path.basename(images_dir)}", cols=6)

def main(root):
    if not os.path.isdir(root):
        print("ERROR: DATA_DIR does not exist:", root)
        return

    print(f"Top-level listing for DATA_DIR: {root}\n")
    for e in safe_listdir(root)[:300]:
        p = os.path.join(root, e)
        tag = "DIR" if os.path.isdir(p) else "FILE"
        print(f"  {tag:4} {e}")
    print("")

    img_dirs = find_image_dirs_under(root)
    if not img_dirs:
        print("No image directories found. Make sure dataset is unzipped and DATA_DIR is correct.")
        return

    # Identify train/val/test among found directories when possible
    # Prefer directories whose name contains train/val/test
    named = {"train": [], "val": [], "test": [], "other": []}
    for d in img_dirs:
        bn = os.path.basename(os.path.normpath(d)).lower()
        if "train" in bn:
            named["train"].append(d)
        elif bn in ("val", "valid", "validation"):
            named["val"].append(d)
        elif "test" in bn:
            named["test"].append(d)
        else:
            named["other"].append(d)

    # Process each group (train, val, other) and display a single grid per group
    groups = [("train", named["train"]), ("val", named["val"]), ("test", named["test"]), ("other", named["other"])]
    for tag, dirs in groups:
        for d in dirs:
            print(f"\nProcessing {tag} dir: {d}")
            label_dir = find_label_dir_for_images(d, root)
            if label_dir:
                print("  using label dir:", label_dir)
            else:
                print("  no label dir detected for this images folder (labels may be missing)")

            # show grid (overlaid if labels exist)
            sample_and_show(d, label_dir, label_dir, tag)

    print("\nDone. If you want different sample size, edit MAX_IMAGES_PER_GRID at top of script.")

if __name__ == "__main__":
    main(DATA_DIR)
