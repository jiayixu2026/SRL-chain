# SRL-Chain: Semantic Role Labeling-Guided Reasoning Chain for Aspect-Based Sentiment Analysis

This repository contains the implementation of **SRL-Chain**, a
**Semantic Role Labeling (SRL)-guided reasoning framework** for
**Aspect-Based Sentiment Analysis (ABSA)**.

SRL-Chain introduces **Semantic Role Labeling (SRL)** into LLM-based
sentiment reasoning and formulates ABSA as a three-step reasoning chain.
By leveraging predicate-argument structures, SRL-Chain provides explicit
semantic information for aspect-oriented sentiment analysis.

The framework consists of three steps:

1.  **Semantic Role Analysis**\
    Identify the semantic structure related to the target aspect through
    SRL representations.

2.  **Opinion Extraction**\
    Extract explicit or implicit opinion expressions associated with the
    target aspect.

3.  **Sentiment Classification**\
    Predict the sentiment polarity based on the extracted opinion
    information.

------------------------------------------------------------------------

## Framework Overview

Given a sentence and an aspect term, SRL-Chain first generates a
Semantic Role Labeling representation and then performs step-by-step
reasoning for sentiment prediction.

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

------------------------------------------------------------------------

## Repository Structure

    SRL-Chain-ABSA/

    ├── zero_shot_inference.py
    │   └── Zero-shot sentiment classification pipeline

    ├── distill_train_data.py
    │   └── Generate reasoning-chain data for fine-tuning

    ├── srl_processor_AllenNLP_BERT.py
    │   └── SRL processing module based on AllenNLP BERT SRL model

    ├── core/
    │   ├── api_utils.py
    │   │   └── API calls and configuration utilities
    │   └── batch_processor.py
    │       └── Batch processing and checkpoint recovery

    ├── requirements.txt
    ├── .env.example
    ├── README.md
    └── LICENSE

------------------------------------------------------------------------

## Installation

### 1. Create environment

``` bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2. Install dependencies

``` bash
pip install -r requirements.txt
```

### 3. Download spaCy model

``` bash
python -m spacy download en_core_web_sm
```

### 4. Configure API key

Copy `.env.example`:

``` bash
cp .env.example .env
```

For Windows:

``` bash
copy .env.example .env
```

Then configure:

``` text
OPENAI_API_KEY=your_api_key
```

------------------------------------------------------------------------

## Usage

### Zero-shot Inference

``` bash
python zero_shot_inference.py
```

Before running, modify:

-   `DATA_PATH`
-   `MODEL`
-   `SAVE_DIR`
-   `OUTPUT_FILENAME`

### Generate Fine-tuning Data

``` bash
python distill_train_data.py
```

This script generates SRL-Chain reasoning data through LLM-based
knowledge distillation for downstream model fine-tuning.

------------------------------------------------------------------------

## Dataset

SRL-Chain is evaluated on widely used ABSA benchmark datasets:

-   Rest14
-   Rest15
-   Rest16
-   Lap14

Datasets should be obtained from their original releases and placed
under the `data/` directory.

Expected format:

``` json
{
  "sentence": "The food was great.",
  "aspect": "food",
  "label": "positive"
}
```

------------------------------------------------------------------------

## Experimental Settings

The framework supports:

-   LLM-based zero-shot inference
-   LoRA-based fine-tuning
-   SRL-guided reasoning chain generation

Models used in experiments include:

-   GPT-3.5-Turbo
-   Llama-2-7B-Chat
-   Llama-3-8B-Instruct

------------------------------------------------------------------------

## License

This project is released under the MIT License.
