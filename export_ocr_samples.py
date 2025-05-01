# export_ocr_samples.py
# version: 0.1.3
# path: export_ocr_samples.py

import os
import sys
import cv2
import yaml
import glob
import argparse

def load_regions(yaml_path):
    """Load ROI definitions from YAML."""
    if not os.path.exists(yaml_path):
        print(f"Error: regions.yaml not found at {yaml_path}")
        sys.exit(1)
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f) or {}
    return data.get('regions', {})

def main():
    parser = argparse.ArgumentParser(
        description="Export OCR samples from ROI screenshots folder."
    )
    parser.add_argument(
        "--screenshots-dir", "-s",
        type=str,
        default=os.path.join("src", "roi_screenshots"),
        help="Directory containing ROI screenshot files"
    )
    parser.add_argument(
        "--regions-yaml", "-r",
        type=str,
        default=os.path.join("src", "regions.yaml"),
        help="Path to src/regions.yaml"
    )
    parser.add_argument(
        "--out-dir", "-o",
        type=str,
        default=os.path.join("training_texts_dir", "images"),
        help="Output directory for OCR sample images"
    )
    args = parser.parse_args()

    screenshots_dir = args.screenshots_dir
    regions_yaml = args.regions_yaml
    out_dir = args.out_dir

    # Ensure output dir exists
    os.makedirs(out_dir, exist_ok=True)

    # Load region definitions
    regions = load_regions(regions_yaml)

    # Gather all screenshot image files
    patterns = [os.path.join(screenshots_dir, "*.png"), os.path.join(screenshots_dir, "*.jpg")]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    if not files:
        print(f"No screenshot files found in {screenshots_dir}")
        return

    for filepath in files:
        filename = os.path.basename(filepath)
        name_part = filename.rsplit('_', 2)[0]  # strip timestamp and extension
        region_info = regions.get(name_part)
        # Only export text-type ROIs
        if not region_info or region_info.get("type") != "text":
            continue
        # Read image and save as TIFF
        img = cv2.imread(filepath)
        if img is None:
            print(f"Warning: failed to read {filepath}")
            continue
        out_filename = os.path.splitext(filename)[0] + ".tif"
        out_path = os.path.join(out_dir, out_filename)
        cv2.imwrite(out_path, img)
        print(f"Saved OCR sample: {out_path}")

if __name__ == "__main__":
    main()

