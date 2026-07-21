# SRL-Chain: Semantic Role–Guided Reasoning Chain for Aspect-Based Sentiment Analysis

This repository contains the implementation of **SRL-Chain**, a semantic role-guided reasoning framework for **Aspect-Based Sentiment Analysis (ABSA)**.

SRL-Chain introduces **Semantic Role Labeling (SRL)** into LLM-based sentiment reasoning and decomposes ABSA into a three-step reasoning chain:

1. **Semantic Role Analysis**: Identify the semantic structure related to the target aspect.
2. **Opinion Extraction**: Extract explicit or implicit opinion expressions toward the aspect.
3. **Sentiment Classification**: Predict the sentiment polarity based on the extracted opinion.

---

## Framework Overview

Given a sentence and an aspect term, SRL-Chain first generates an SRL-based semantic representation, then performs step-by-step reasoning for sentiment prediction.

```
Sentence + Aspect
        |
        v
Semantic Role Analysis (SRL)
        |
        v
Opinion Extraction
        |
        v
Sentiment Classification
```

---

## Repository Structure

```
SRL-Chain-ABSA/
│
├── zero_shot_inference.py          # Zero-shot inference pipeline
├── distill_train_data.py           # Generate distillation data for fine-tuning
├── srl_processor_AllenNLP_BERT.py  # SRL processing module
│
├── core/
│   ├── api_utils.py                # API and configuration utilities
│   └── batch_processor.py          # Batch processing and checkpoint recovery
│
├── requirements.txt
├── .env.example
├── README.md
└── LICENSE
```

---

## Installation

### 1. Create environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 4. Configure API key

Copy `.env.example`:

```bash
cp .env.example .env
```

Then add your OpenAI API key:

```
OPENAI_API_KEY=your_api_key
```

---

## Usage

### Zero-shot inference

Run:

```bash
python zero_shot_inference.py
```

Before running, modify the configuration parameters:

- `DATA_PATH`
- `MODEL`
- `SAVE_DIR`
- `OUTPUT_FILENAME`

---

### Generate fine-tuning data

Run:

```bash
python distill_train_data.py
```

This script uses LLM-based knowledge distillation to generate SRL-Chain reasoning data for downstream fine-tuning.

---

## Dataset

Experiments are conducted on standard ABSA datasets:

- Rest14
- Rest15
- Rest16
- Lap14

The datasets should be obtained from their original releases and placed under the `data/` directory.

Expected format:

```json
{
  "sentence": "The food was great.",
  "aspect": "food",
  "label": "positive"
}
```

---

## Citation

If you find this repository useful, please cite:

```bibtex
@article{your_paper,
  title={SRL-Chain: Semantic Role–Guided Reasoning Chain for Aspect-Based Sentiment Analysis},
  author={Your Name},
  year={2026}
}
```

---

## License

This project is released under the MIT License.