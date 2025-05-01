# src/ocr_finetune.py
# version: 0.1.0

import subprocess

def train_ocr_model(training_dir, output_prefix):
    """
    Fine-tune Tesseract OCR on EVE-specific text.
    Expects subfolders:
      - images/: .tif samples
      - box/: .box files
    """
    cmds = [
        f"tesseract {training_dir}/images/train.tif {output_prefix} --psm 6 lstm.train",
        f"combine_tessdata -e {output_prefix}.traineddata {output_prefix}.lstm",
        f"lstmtraining --model_output {output_prefix}_fine --continue_from {output_prefix}.lstm "
        f"--traineddata {training_dir}/traineddata --train_listfile {training_dir}/train_list.txt "
        f"--max_iterations 2000",
        f"combine_tessdata {output_prefix}_fine"
    ]
    for c in cmds:
        subprocess.run(c, shell=True, check=True)
    print("OCR fine-tuning complete:", output_prefix + "_fine.traineddata")
