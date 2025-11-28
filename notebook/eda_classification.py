# notebooks/eda_classification_onegrid.py
"""
Classification EDA — single-frame grid display.

Supported layouts:
  - ROOT/classA/*.jpg  (folder-per-class)
  - ROOT/train/classA/*.jpg  and ROOT/valid/classA/*.jpg
  - ROOT/images/*.jpg  with optional labels.csv containing columns: filename,label

What it does:
  - Detects dataset layout (class folders, train/valid, or flat+CSV)
  - Prints counts, basic size stats
  - Samples up to MAX_IMAGES_PER_GRID images per split (train/val/test/flat)
  - Displays samples in ONE single matplotlib figure per split (grid)
  - Safe against corrupt images; uses simple resize without Pillow-specific filters

Edit DATA_DIR and run:
    python notebooks/eda_classification_onegrid.py
"""

import os
import random
import math
import csv
from collections import defaultdict, Counter
from PIL import Image, ImageDraw, UnidentifiedImageError
import matplotlib.pyplot as plt
import numpy as np

# =========== CONFIG =============
# Set this to the folder that contains your classification dataset.
# Examples:
#  - C:/datasets/classification                      (folder-per-class)
#  - C:/datasets/classification/ (with train/ and valid/ subfolders)
DATA_DIR = r"C:\Users\ruhet\Labmentix_Projects\aerial-classsification\data_unzipped\classification_dataset"

# maximum images to show in a single grid (per split)
MAX_IMAGES_PER_GRID = 24

# image file extensions to consider
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
# =================================

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

def find_class_folders(root):
    """Detect folder-per-class at root (class folders directly under root)."""
    classes = {}
    for entry in safe_listdir(root):
        p = os.path.join(root, entry)
        if os.path.isdir(p):
            imgs = [os.path.join(p, f) for f in safe_listdir(p) if f.lower().endswith(IMG_EXTS)]
            if imgs:
                classes[entry] = imgs
    return classes

def find_train_valid(root):
    """Detect train/valid subfolders and class folders within them."""
    splits = {}
    for split_name in ("train", "train_images", "training", "val", "valid", "validation"):
        p = os.path.join(root, split_name)
        if os.path.isdir(p):
            # are there class folders inside?
            class_folders = find_class_folders(p)
            if class_folders:
                splits[split_name] = class_folders
            else:
                # maybe images directly inside train/
                imgs = [os.path.join(p, f) for f in safe_listdir(p) if f.lower().endswith(IMG_EXTS)]
                if imgs:
                    splits[split_name] = {"__flat__": imgs}
    return splits

def find_flat_with_csv(root):
    """
    If images are flat in root or in a folder 'images', look for labels.csv or labels.tsv
    CSV expected columns: filename,label (header optional)
    """
    # find image dir candidate
    candidates = []
    if contains_images(root):
        candidates.append(root)
    images_folder = os.path.join(root, "images")
    if os.path.isdir(images_folder) and contains_images(images_folder):
        candidates.append(images_folder)

    # look for csv
    csv_candidates = [os.path.join(root, "labels.csv"), os.path.join(root, "labels.tsv"),
                      os.path.join(root, "labels.txt"), os.path.join(root, "labels.csv")]
    csv_candidates += [os.path.join(root, f) for f in safe_listdir(root) if f.lower().endswith(('.csv','.tsv','.txt'))]
    found_csv = None
    for c in csv_candidates:
        if os.path.isfile(c):
            # do a quick sniff to see if it likely contains filename,label columns
            try:
                with open(c, newline='', encoding='utf-8') as fh:
                    reader = csv.reader(fh)
                    rows = [r for r in (next(reader, []), next(reader, [])) if r]
                    # basic heuristic: second column exists
                    if rows and len(rows[0]) >= 2:
                        found_csv = c
                        break
            except Exception:
                continue

    if candidates and found_csv:
        # build mapping
        mapping = {}
        try:
            with open(found_csv, newline='', encoding='utf-8') as fh:
                reader = csv.reader(fh)
                for row in reader:
                    if not row:
                        continue
                    if len(row) >= 2:
                        fname = row[0].strip()
                        label = row[1].strip()
                        if fname:
                            mapping[fname] = label
        except Exception:
            mapping = {}
        return candidates[0], found_csv, mapping
    return None, None, None

def quick_image_size_stats(paths, max_samples=500):
    sizes = []
    cnt = 0
    for p in paths:
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
        except Exception:
            pass
        cnt += 1
        if cnt >= max_samples:
            break
    if not sizes:
        return None
    arr = np.array(sizes)
    stats = {
        "mean": tuple(arr.mean(axis=0).astype(int)),
        "median": tuple(np.median(arr, axis=0).astype(int)),
        "min": tuple(arr.min(axis=0).astype(int)),
        "max": tuple(arr.max(axis=0).astype(int))
    }
    return stats

def draw_label_on_image(pil_img, label_text):
    """Draw a small label bar at top-left of the PIL image (in-place)."""
    try:
        draw = ImageDraw.Draw(pil_img)
        w, h = pil_img.size
        # small rectangle height
        rect_h = max(16, int(h * 0.06))
        # semi-opaque background
        draw.rectangle([0, 0, w, rect_h], fill=(0,0,0,160))
        # white text
        draw.text((4, 2), str(label_text), fill="white")
    except Exception:
        pass
    return pil_img

def safe_open_and_resize(path, max_dim=640):
    """Open image, resize to max_dim maintain aspect ratio, using simple resize() (no filters)."""
    try:
        im = Image.open(path).convert("RGB")
    except UnidentifiedImageError:
        # placeholder for corrupt image
        im = Image.new("RGB", (min(640, max_dim), min(480, max_dim)), color=(60,60,60))
        draw = ImageDraw.Draw(im)
        draw.text((10,10), "CORRUPT", fill="white")
        return im
    except Exception:
        im = Image.new("RGB", (min(640, max_dim), min(480, max_dim)), color=(80,40,40))
        draw = ImageDraw.Draw(im)
        draw.text((10,10), "ERROR", fill="white")
        return im

    try:
        w,h = im.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            # SAFE resize without specifying resampling filter (works across Pillow versions)
            im = im.resize((new_w, new_h))
    except Exception:
        # if anything goes wrong, return original image
        pass
    return im

def show_images_grid(image_paths, labels=None, title="Image Grid", max_images=MAX_IMAGES_PER_GRID):
    """
    Show a list of image paths (and optional labels) in a single matplotlib grid.
    labels can be a list parallel to image_paths (shorter/longer ignored).
    """
    if not image_paths:
        print("No images to display.")
        return
    image_paths = image_paths[:max_images]
    n = len(image_paths)
    cols = min(6, n)
    rows = math.ceil(n / cols)
    plt.figure(figsize=(cols * 3, rows * 3))
    plt.suptitle(title, fontsize=14)
    for i, p in enumerate(image_paths, 1):
        plt.subplot(rows, cols, i)
        im = safe_open_and_resize(p)
        # draw label if provided
        if labels and i-1 < len(labels) and labels[i-1]:
            draw_label_on_image(im, labels[i-1])
        plt.imshow(im)
        plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

def main(root):
    if not os.path.isdir(root):
        print("ERROR: DATA_DIR does not exist:", root)
        return

    print("Top-level listing of DATA_DIR:")
    for e in safe_listdir(root)[:300]:
        p = os.path.join(root, e)
        tag = "DIR" if os.path.isdir(p) else "FILE"
        print(f"  {tag:4} {e}")
    print("")

    # 1) Try folder-per-class at root
    class_folders = find_class_folders(root)
    if class_folders:
        print("Detected folder-per-class at root.")
        counts = {k: len(v) for k,v in class_folders.items()}
        print("Class counts (top 20):")
        for k, v in Counter(counts).most_common(20):
            pass
        for k in sorted(counts.keys()):
            print(f"  {k}: {counts[k]}")
        # show a grid with samples across classes (one image per class up to MAX_IMAGES_PER_GRID)
        sample_paths = []
        sample_labels = []
        for cls, imgs in class_folders.items():
            if imgs:
                pick = random.choice(imgs)
                sample_paths.append(pick)
                sample_labels.append(cls)
                if len(sample_paths) >= MAX_IMAGES_PER_GRID:
                    break
        print(f"\nShowing up to {MAX_IMAGES_PER_GRID} class-representative images (one per class).")
        show_images_grid(sample_paths, sample_labels, title="Class-representative samples")
        return

    # 2) Try train/valid subfolders
    splits = find_train_valid(root)
    if splits:
        print("Detected train/valid style structure.")
        for split_name, class_map in splits.items():
            print(f"\nSplit: {split_name}")
            # flatten to list of (path,label)
            flat = []
            for cls, imgs in class_map.items():
                if cls == "__flat__":
                    for p in imgs:
                        flat.append((p, None))
                else:
                    for p in imgs:
                        flat.append((p, cls))
            print(f"  total images in {split_name}: {len(flat)}")
            if not flat:
                continue
            # sample up to MAX_IMAGES_PER_GRID and show
            sample = random.sample(flat, min(MAX_IMAGES_PER_GRID, len(flat)))
            paths = [p for p,_ in sample]
            labs = [lbl for _,lbl in sample]
            show_images_grid(paths, labs, title=f"{split_name} samples")
        return

    # 3) Try flat images + CSV mapping
    img_dir, csv_path, mapping = find_flat_with_csv(root)
    if img_dir and mapping:
        print("Detected flat images with CSV labels.")
        print("Using CSV:", csv_path)
        # build list of files present with labels
        found = []
        for fname, lbl in mapping.items():
            p = os.path.join(img_dir, fname)
            if os.path.exists(p) and p.lower().endswith(IMG_EXTS):
                found.append((p, lbl))
        print(f"  mapped labelled files found: {len(found)}")
        if found:
            sample = random.sample(found, min(MAX_IMAGES_PER_GRID, len(found)))
            paths = [p for p,_ in sample]
            labs = [l for _,l in sample]
            show_images_grid(paths, labs, title="CSV-labelled samples")
            return

    # 4) Fallback: find any images under root (nested)
    all_imgs = []
    for subdir, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                all_imgs.append(os.path.join(subdir, f))
    if all_imgs:
        print("No class folders found — falling back to a random sample of images under root.")
        print("Total images found:", len(all_imgs))
        sample = random.sample(all_imgs, min(MAX_IMAGES_PER_GRID, len(all_imgs)))
        show_images_grid(sample, title="Random image samples (no labels detected)")
        return

    print("No images found by the script. Possible causes:")
    print(" - DATA_DIR is incorrect or not unzipped")
    print(" - Images use unusual extensions (not jpg/png)")
    print(" - Labels are in a custom CSV with different schema")

if __name__ == "__main__":
    main(DATA_DIR)
