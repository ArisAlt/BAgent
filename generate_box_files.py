# generate_box_files.py
# version: 0.1.1
# path: generate_box_files.py

import os
import sys
import glob
import subprocess
import argparse
import shutil
import pytesseract


def main():
    """
    Generate .box files for all .tif images in the images directory
    using Tesseract's makebox mode.
    """
    parser = argparse.ArgumentParser(
        description="Generate Tesseract .box files for all .tif images."
    )
    parser.add_argument(
        "--images-dir", "-i",
        type=str,
        default="training_texts_dir/images",
        help="Directory containing .tif images"
    )
    parser.add_argument(
        "--box-dir", "-b",
        type=str,
        default="training_texts_dir/box",
        help="Directory to save generated .box files"
    )
    parser.add_argument(
        "--tesseract-cmd", "-t",
        type=str,
        default=os.environ.get("TESSERACT_CMD"),
        help="Path to the tesseract executable (or set TESSERACT_CMD env var; defaults to your PATH)"
    )
    args = parser.parse_args()

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    # Determine tesseract command
    tess_cmd = args.tesseract_cmd or shutil.which("tesseract")
    if not tess_cmd:
        print("❌ Error: Cannot find 'tesseract' on your PATH.")
        print("   • On Windows, install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki")
        print("   • Then add its folder (e.g. run Bat file as admin to add tesseract at your PATH.")
        sys.exit(1)

    images_dir = args.images_dir
    box_dir = args.box_dir

    if not os.path.isdir(images_dir):
        print(f"❌ Error: Images directory not found: {images_dir}")
        sys.exit(1)

    os.makedirs(box_dir, exist_ok=True)

    tif_files = glob.glob(os.path.join(images_dir, "*.tif"))
    if not tif_files:
        print(f"⚠️ No .tif files found in {images_dir}")
        return

    for tif_path in tif_files:
        base = os.path.splitext(os.path.basename(tif_path))[0]
        out_base = os.path.join(box_dir, base)
        print(f"🔍 Generating box for: {tif_path}")
        try:
            subprocess.run(
                [tess_cmd, tif_path, out_base, "batch.nochop", "makebox"],
                check=True
            )
            print(f"  ✅ Generated: {out_base}.box")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Tesseract failed for {tif_path}: {e}")
        except FileNotFoundError:
            print(f"  ❌ Executable not found: {tess_cmd}")
            print("     Please install Tesseract and/or provide --tesseract-cmd path.")
            break

if __name__ == "__main__":
    main()
