import os
import time
import random
import numpy as np

try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
except ImportError:
    pass

import openai

from core.api_utils import (
    load_config, get_config, write_json, read_json,
    safe_str_concat, get_completion_from_messages
)
from core.batch_processor import process_batch_data
from srl_processor_AllenNLP_BERT import get_semantic_role_labeling_v2


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

set_seed(42)

config = load_config()

OPENAI_API_KEY = get_config("OPENAI_API_KEY", "")
MODEL = get_config("MODEL", "gpt-3.5-turbo-1106")

DATA_PATH = get_config("DISTILL_DATA", "...")
SAVE_DIR = get_config("DISTILL_SAVE", "...")
OUTPUT_FILENAME = get_config("DISTILL_OUTPUT", "...")
TEMP_PREFIX = get_config("DISTILL_TEMP", "...")

BATCH_SIZE = int(get_config("BATCH_SIZE", "100"))
TEST_SAMPLES = int(get_config("TEST_SAMPLES", "-1"))

openai.api_key = OPENAI_API_KEY


def prompt_train_step1_srl_analysis(sentence, aspect, srl_structure, ground_truth):
    message = [
        {"role": "system", "content": (
            "You are an expert AI assistant specialized in Semantic Role Labeling (SRL) and linguistic analysis. "
            "Your response must be precise and limited to 150 words or less."
        )},
        {"role": "user", "content": f"""Given the sentence: '{sentence}'
The target aspect is: '{aspect}'
The known ground-truth sentiment polarity towards '{aspect}' is: '{ground_truth}'

Below is the Semantic Role Labeling (SRL) analysis for this sentence:
{srl_structure}

LINGUISTIC KEY:
- V: Predicate (Main action/state)
- ARG0: Agent / Evaluator
- ARG1: Patient / Evaluation Target
- ARG2: Attribute / Evaluation Criterion
- ARGM-*: Modifiers (Negation, Manner, Location, etc.)

TASK:
Based on the provided SRL table and knowing that the final sentiment is '{ground_truth}', analyze what semantic roles '{aspect}' plays in this structure. Explain which arguments, predicates, or modifiers are directly or indirectly linked to '{aspect}' that rationally support this '{ground_truth}' sentiment."""},
    ]
    return message


def prompt_train_step2_opinion_extraction(sentence, aspect, aspect_info, ground_truth):
    message = [
        {"role": "system", "content": (
            "You are an expert AI assistant that extracts user opinions from textual context. "
            "Your response must be concise and limited to 100 words or less."
        )},
        {"role": "user", "content": f"""Given the sentence: '{sentence}'
Target aspect: '{aspect}'
Ground-truth sentiment: '{ground_truth}'

Here is the semantic linkage analysis from Step 1:
{aspect_info}

TASK:
Considering the sentence context and the semantic role connections analyzed above, explicitly identify the speaker's specific opinion expressions, descriptors, or implicit attitude towards '{aspect}' that justifies the '{ground_truth}' sentiment."""},
    ]
    return message


def prompt_train_step3_standardized_label(sentence, aspect, ground_truth, polarity_expr):
    message = [
        {"role": "system", "content": (
            "You are a sentiment analysis data generator. You must follow the exact Output Format provided below.\n"
            "Output Format:\n"
            "The sentiment towards {Aspect} in the given sentence is {positive, negative or neutral}. Because [Provide a 1-sentence concise reason based on the opinion extracted].\n"
        )},
        {"role": "user", "content": f"""Sentence: '{sentence}'
Aspect: '{aspect}'
Extracted Opinion Path: {polarity_expr}

TASK:
Generate the final step of the reasoning chain. You MUST strictly use the required ground-truth sentiment polarity '{ground_truth}' for '{aspect}'."""},
    ]
    return message


def get_distillation_completion(text):
    sentence = text['sentence']
    aspect = text['aspect']
    label = text['label']

    srl_structure = get_semantic_role_labeling_v2(sentence, aspect=aspect)

    step_1_msg = prompt_train_step1_srl_analysis(sentence, aspect, srl_structure, label)
    aspect_info = get_completion_from_messages(step_1_msg, model=MODEL, max_retries=5)
    time.sleep(0.4)

    step_2_msg = prompt_train_step2_opinion_extraction(sentence, aspect, aspect_info, label)
    polarity_expr = get_completion_from_messages(step_2_msg, model=MODEL, max_retries=5)
    time.sleep(0.4)

    step_3_msg = prompt_train_step3_standardized_label(sentence, aspect, label, polarity_expr)
    output_lb = get_completion_from_messages(step_3_msg, model=MODEL, max_retries=5)

    full_target_chain = safe_str_concat(
        "1--- ", aspect_info.strip(), "\n",
        "2--- ", polarity_expr.strip(), "\n",
        "3--- ", output_lb.strip()
    )

    output_entry = {
        'sentence': sentence,
        'aspect': aspect,
        'label': label,
        'srl_structure': srl_structure,
        'distilled_aspect_info': aspect_info,
        'distilled_opinion': polarity_expr,
        'distilled_output_lb': output_lb,
        'fine_tune_input': f"Sentence: {sentence}\nAspect: {aspect}\nSRL Table:\n{srl_structure}",
        'fine_tune_target': full_target_chain,
        'method': "SRL-Chain-Distill"
    }
    return output_entry


if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        exit(1)

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 60)
    print("Syn-SRL-Chain - Fine-tuning Data Distillation Generator")
    print("=" * 60)
    print(f"Base Model: {MODEL}")
    print(f"Input Data: {DATA_PATH}")
    print(f"Output Path: {SAVE_DIR}/{OUTPUT_FILENAME}")
    print("=" * 60)

    raw_data = read_json(DATA_PATH)
    print(f"Loaded {len(raw_data)} raw samples")

    if TEST_SAMPLES > 0:
        raw_data = raw_data[:TEST_SAMPLES]
        print(f"[Test Mode] Running with first {len(raw_data)} samples only")

    final_filepath = process_batch_data(
        data=raw_data,
        save_dir=SAVE_DIR,
        output_filename=OUTPUT_FILENAME,
        temp_prefix=TEMP_PREFIX,
        process_single_fn=get_distillation_completion,
        batch_size=BATCH_SIZE,
        progress_desc="Generating distillation batch"
    )
