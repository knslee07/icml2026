# Domain-Specific Alignment for Information Extraction

Code and data for the ICML 2026 submission:

**"Domain-Specific Alignment for Information Extraction: A Framework for Social Science Data Collection"**

## Abstract

Social science research increasingly relies on extracting structured data from archival documents. Using consultant name extraction from SEC filings as a testbed, we compare base Gemma 3 27B (69.1% F1), general IE alignment via IEInstruct (83.2% F1), and domain-specific alignment (94.0% F1), demonstrating that specialized extraction tasks benefit from targeted training despite advances in general IE.

## Repository Structure

```
├── src/
│   ├── train.py          # Domain-specific LoRA fine-tuning
│   ├── inference.py      # Multi-stage extraction pipeline
│   └── evaluate.py       # Evaluation metrics computation
├── data/
│   └── gold_labels.json  # Test set annotations (83 filings, 317 consultants)
└── scripts/
    └── run_evals.sh      # Evaluation runner
```

## External Data (Hugging Face Hub)

Due to file size constraints, adapters and evaluation data are hosted on Hugging Face:

### LoRA Adapters

| Model | Hugging Face Link | F1 |
|-------|-------------------|-----|
| Domain-Specific | [cs-file-uploads/domain-specific-adapter](https://huggingface.co/cs-file-uploads/domain-specific-adapter) | 94.0% |
| General IE (NER) | [cs-file-uploads/ie-ner-adapter](https://huggingface.co/cs-file-uploads/ie-ner-adapter) | 74.2% |
| General IE (Balanced) | [cs-file-uploads/ie-balanced-adapter](https://huggingface.co/cs-file-uploads/ie-balanced-adapter) | 83.2% |

### Evaluation Data

| Dataset | Hugging Face Link | Description |
|---------|-------------------|-------------|
| Proxy Statements | [cs-file-uploads/eval_data](https://huggingface.co/datasets/cs-file-uploads/eval_data) | 84 SEC filings for evaluation |

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python 3.10+
- PyTorch 2.1+ with CUDA 12.1
- transformers >= 4.40.0
- peft >= 0.10.0
- bitsandbytes >= 0.43.0

## Hardware Requirements

| Model | Training | Inference |
|-------|----------|-----------|
| Gemma 3 27B | A100 80GB | 2x RTX 3090 (48GB total) |
| Gemma 3 12B | A40 48GB | 1x RTX 3090 (24GB) |

## Quick Start

### 1. Download Adapters and Data

Download from Hugging Face Hub:

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

# Download adapters
huggingface-cli download cs-file-uploads/domain-specific-adapter --local-dir ./adapters/domain_specific
huggingface-cli download cs-file-uploads/ie-ner-adapter --local-dir ./adapters/ie_ner
huggingface-cli download cs-file-uploads/ie-balanced-adapter --local-dir ./adapters/ie_balanced

# Download evaluation data
huggingface-cli download cs-file-uploads/eval_data --repo-type dataset --local-dir ./data/eval
```

### 2. Run Inference

```bash
# Domain-specific model
python src/inference.py \
    --adapter ./adapters/domain_specific \
    --data ./data/eval \
    --output results/pred_domain.json \
    --instruction long

# Base model (vanilla)
python src/inference.py \
    --vanilla \
    --data ./data/eval \
    --output results/pred_vanilla.json
```

### 3. Evaluate

```bash
python src/evaluate.py \
    --predictions ./results \
    --gold ./data/gold_labels.json \
    --output results/eval_domain.csv \
    --normalize
```

## Training (Optional)

To reproduce training (requires your own training data):

```bash
python src/train.py \
    --model 27b \
    --data your_training_data.jsonl \
    --output ./checkpoints/
```

### Training Data Format

JSONL with instruction/input/output fields:

```json
{
    "instruction": "You are a meticulous reader of proxy-statement excerpts...",
    "input": "TEMPORAL CONTEXT: This document was filed in 2023...\nTEXT TO ANALYZE:\n...",
    "output": "{RET: 'Pearl Meyer & Partners, LLC'}, {SURV: 'Mercer', 'Radford'}"
}
```

## Model Configurations

### Domain-Specific Training

| Parameter | Value |
|-----------|-------|
| Base model | Gemma 3 27B-IT |
| Quantization | 4-bit NF4 (QLoRA) |
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup ratio | 3% |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Max sequence length | 5,120 tokens |
| Training epochs | 3 |
| Validation split | 20% |

### Inference Pipeline

| Stage | Description |
|-------|-------------|
| 1. Retrieval | Keyword-based filtering (250 words context) |
| 2. Chunking | ~4,000 tokens per chunk with 25-token overlap |
| 3. Extraction | LLM query with structured output |
| 4. Validation | Grounding check against source text |
| 5. Aggregation | Cross-chunk deduplication and normalization |

## Results

### Main Results (Table 1 in paper)

| Model | Type | P | R | F1 |
|-------|------|---|---|-----|
| Gemma 3 27B-IT | zero-shot | .645 | .744 | .691 |
| + General IE (NER) | fine-tuned | .683 | .814 | .742 |
| + General IE (Balanced) | fine-tuned | .822 | .842 | .832 |
| + Domain-Specific (Ours) | fine-tuned | **.937** | **.943** | **.940** |
| GPT-5-nano | zero-shot | .781 | .789 | .785 |
| GPT-5-mini | zero-shot | .865 | .852 | .859 |

### Category-Level Results

| Model | RET F1 | SURV F1 |
|-------|--------|---------|
| Base (zero-shot) | 0.676 | 0.708 |
| General IE (Balanced) | 0.793 | 0.875 |
| Domain-Specific | **0.906** | **0.971** |

## Citation

```bibtex
@inproceedings{anonymous2026domainspecific,
    title={Domain-Specific Alignment for Information Extraction: A Framework for Social Science Data Collection},
    author={Anonymous},
    booktitle={International Conference on Machine Learning},
    year={2026}
}
```

## License

This code is released under the MIT License. See LICENSE for details.

The evaluation data consists of publicly available SEC filings.
