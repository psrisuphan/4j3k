# ForJustice3K
A browser extension designed to detect and filter hate speech in Thai-language posts and comments across social media platforms. The tool automatically identifies harmful or abusive language and blocks it by default, while giving users the option to unblock the content if they choose.

**Group project for Artificial intelligence.

## Usage

### 1. Environment setup
- Ensure Python 3.9+ is available.
- (Optional) Create and activate a virtual environment: `python -m venv venv && source venv/bin/activate`.
- Install dependencies: `pip install -r requirements.txt`.

### 2. Prepare training data
- Place the labelled CSV in `data/HateThaiSent.csv` (default expected by the training script).
- The CSV must contain a `Message` column with text and a `Hatespeech` column with `hatespeech`/`nonhatespeech` labels.

### 3. Train the classifier
Fine-tune WangchanBERTa with the bundled training helper:

```bash
python train_model.py \
    --data data/HateThaiSent.csv \
    --output-dir models/wangchanberta-hatespeech \
    --epochs 2 \
    --batch-size 8
```

Key options:
- `--device auto|cpu|cuda|mps`: choose the execution device.
- `--freeze-encoder` or `--trainable-layer-count N`: control how much of the backbone to fine-tune.
- `--max-length`, `--learning-rate`, `--gradient-accumulation`: adjust training performance.
- `--lr-scheduler`, `--warmup-steps`, `--warmup-ratio`: configure the learning-rate schedule and warmup strategy.
- `--fp16` / `--bf16`: enable mixed-precision training on supported hardware.
- `--gradient-checkpointing`, `--group-by-length`: trade compute for lower memory and reduce padding overhead.
- `--torch-compile`: turn on PyTorch 2.x graph capture for potential throughput gains.

### 4. Run inference
After training finishes, score new text with age-aware post-processing:

```bash
python predict.py "ใส่ข้อความภาษาไทยที่นี่" --age 15 --model models/wangchanberta-hatespeech
```

The script prints a JSON object containing the raw probability, whether the content should be blocked for the provided age, and the policy thresholds that were applied.

### 5. Export metrics
Training writes evaluation metrics to `models/wangchanberta-hatespeech/eval_metrics.json`. Share this file or use it to monitor regression between fine-tuning runs.

### 6. Next steps
- Integrate the exported model directory into the browser extension packaging workflow.
- Adjust `age_policy.py` if you need different moderation thresholds per age bracket.
