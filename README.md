# ForJustice3K
A AI model designed to detect hatespeech from message given. Based on [WangchanBERTa](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased).
Trained with [HateThaiSent](https://github.com/dsmlr/HateThaiSent), [ThaiToxicityTweetCorpus](https://github.com/tmu-nlp/ThaiToxicityTweetCorpus).

**Group project for Artificial intelligence.

## Usage

### 1. Environment setup
- Ensure Python 3.9+ is available.
- (Optional) Create and activate a virtual environment: `python -m venv venv && source venv/bin/activate`.
- Install dependencies: `pip install -r requirements.txt`.

If you're on Windows and want GPU acceleration, keep the optional `torch-directml` dependency; on other platforms it is safe to ignore if installation fails.

### 2. Prepare training data
- Place the labelled CSV in `data/HateThaiSent.csv` (default expected by the training script).
- The CSV must contain a `Message` column with text and a `Hatespeech` column with `hatespeech`/`nonhatespeech` labels.
- Optional: download and convert the Thai Toxicity Tweet corpus with `python convert_thai_toxicity_tweet.py --output data/ThaiToxicityTweet_converted.csv`; the training script consumes this file automatically (run training with `--extra-data` and no values to opt out).

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
- `--device auto|cpu|cuda|mps|dml|directml`: choose the execution device (DirectML unlocks Windows GPU acceleration).
- `--freeze-encoder` or `--trainable-layer-count N`: control how much of the backbone to fine-tune.
- `--extra-data <path ...>`: append additional CSVs that share the HateThaiSent schema; pass `--extra-data` with no values to disable the defaults.
- `--max-length`, `--learning-rate`, `--gradient-accumulation`: adjust training performance.
- `--max-gpu-memory-fraction`: leave VRAM headroom when working on limited GPUs such as Google Colab T4 instances.
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

### 6. Batch scoring helper
Use `sample_predictions.py` for quick qualitative checks or to score your own sentences in bulk:

```bash
python sample_predictions.py --model models/wangchanberta-hatespeech --input sentences.txt --output predictions.csv
```

The script prints per-sentence decisions, reports accuracy when `expected` labels are provided, and can export a CSV for downstream analysis.
Default demo prompts now live in `sample_sentences.jsonl`; edit that file to tweak the built-in examples or pass a custom `.jsonl`/plain-text file through `--input`.

### 7. Next steps
- Integrate the exported model directory into the browser extension packaging workflow.
- Adjust `age_policy.py` if you need different moderation thresholds per age bracket.
- Run `csv_output.py` during data exploration to inspect label distribution and spot missing values.

## Model credit

L. Lowphansirikul, C. Polpanumas, N. Jantrakulchai, and S. Nutanong, "WangchanBERTa: Pretraining transformer-based Thai language models," arXiv preprint arXiv:2101.09635, 2021. [Online]. Available: https://arxiv.org/abs/2101.09635.

## Dataset credits

This project stands on the shoulders of public Thai-language moderation corpora. Please cite the sources below and respect the licences listed by the maintainers when redistributing or publishing results:

- HateThaiSent (`data/HateThaiSent.csv`). Data Science and Machine Learning Research Group (DSMLR), King Mongkut's University of Technology Thonburi. *HateThaiSent: Hate speech and sentiment dataset for Thai social media* (v1.0, 2021). Available at https://github.com/dsmlr/HateThaiSent. See the repository for detailed licensing terms.
- Thai Toxicity Tweet Corpus (`data/ThaiToxicityTweet_converted.csv`). National Electronics and Computer Technology Center (NECTEC) and Artificial Intelligence Association of Thailand (AIAT). *Thai Toxicity Tweet Corpus* (2020). Available via https://archive.org/download/ThaiToxicityTweetCorpus/data.zip with supporting documentation at https://github.com/tmu-nlp/ThaiToxicityTweetCorpus. Review the corpus licence before redistribution.
