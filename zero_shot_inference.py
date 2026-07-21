import os
import time

try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
except ImportError:
    pass

import openai

from core.api_utils import (
    load_config, get_config, read_json,
    safe_str_concat, get_completion_from_messages
)
from core.batch_processor import process_batch_data
from srl_processor_AllenNLP_BERT import get_semantic_role_labeling_v2


load_config()

OPENAI_API_KEY = get_config("OPENAI_API_KEY", "")
MODEL = get_config("MODEL", "gpt-3.5-turbo-1106")

DATA_PATH = get_config("ZERO_SHOT_DATA", "...")
SAVE_DIR = get_config("ZERO_SHOT_SAVE", "...")
OUTPUT_FILENAME = get_config("ZERO_SHOT_OUTPUT", "...")
TEMP_PREFIX = get_config("ZERO_SHOT_TEMP", "...")

BATCH_SIZE = int(get_config("BATCH_SIZE", "100"))
TEST_SAMPLES = int(get_config("TEST_SAMPLES", "-1"))

openai.api_key = OPENAI_API_KEY


def prompt_for_aspect_information_srl(sentence, aspect, srl_structure):
    message = [
        {"role": "system", "content": ("You are an AI assistant that helps people find information."
                      "Please refine your reply and ensure its accuracy. "
                      "The reply length is limited to 200 words or less.")},
        {"role": "user", "content": f"""Given the sentence '{sentence}',

The following is the Semantic Role Labeling (SRL) result of this sentence:

{srl_structure}

Semantic roles explain:
- ARG0: Agent (who performs the action) or Evaluator
- ARG1: Patient (who receives the action) or Evaluation Target
- ARG2: Evaluation Criterion or Attribute
- ARG3: Indirect Object
- ARGM-LOC: Location (where)
- ARGM-TMP: Time (when)
- ARGM-MNR: Manner (how)
- ARGM-CAU: Cause (why)
- V: Predicate (the main verb)

Based on the semantic role information, analyze what roles '{aspect}' plays in the sentence, 
and what information is related to '{aspect}' through these semantic roles."""},
    ]
    return message


def prompt_for_polarity_inferring(sentence, aspect, aspect_info):
    message = [
        {"role": "system", "content": ("You are an AI assistant that helps people find information."
                      "Please refine your reply and ensure its accuracy. "
                      "The reply length is limited to 120 words or less.")},
        {"role": "user", "content": (f"Given the sentence '{sentence}', " +
              aspect_info +
              f"Considering the context and information related to '{aspect}', what is the speaker's opinion towards {aspect}?")},
    ]
    return message


def prompt_for_polarity_label(sentence, aspect, polarity_expr):
    message = [
        {"role": "system", "content": (
            "You are an sentiment analysis expert. I will provide you with a sentence and a certain aspect mentioned in the sentence. "
            "Please analyze the sentiment polarity of that aspect in a given sentence,"
            "Output:\n'''\nThe sentiment towards {{Aspect}} in the given sentence is {{positive, negative or neutral}}.Because\n")},
        {"role": "user", "content": (f"Given the sentence '{sentence}', " +
              polarity_expr +
              f"Based on the common sense and such speaker's opinion, what is the sentiment polarity towards '{aspect}'?")},
    ]
    return message


def get_completion(text):
    sentence = text['sentence']
    aspect = text['aspect']
    label = text['label']

    srl_structure = get_semantic_role_labeling_v2(sentence, aspect=aspect)

    step_1_message = prompt_for_aspect_information_srl(sentence, aspect, srl_structure)
    aspect_info = get_completion_from_messages(step_1_message, model=MODEL)
    time.sleep(0.5)

    step_2_message = prompt_for_polarity_inferring(sentence, aspect, aspect_info)
    polarity_expr = get_completion_from_messages(step_2_message, model=MODEL)
    time.sleep(0.5)

    step_3_message = prompt_for_polarity_label(sentence, aspect, polarity_expr)
    output_lb = get_completion_from_messages(step_3_message, model=MODEL)

    output_entry = {
        'sentence': sentence,
        'aspect': aspect,
        'label': label,
        'srl_structure': srl_structure,
        'aspect_info': aspect_info,
        'opinion': polarity_expr,
        'output_lb': output_lb,
        'output': safe_str_concat("1---", aspect_info, "2---", polarity_expr, "3---", output_lb),
        'match': "null",
        'method': "SRL-Chain"
    }
    return output_entry

if __name__ == '__main__':
    if not OPENAI_API_KEY:
        raise SystemExit("Error: OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"Error: Data file not found at {DATA_PATH}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 60)
    print("SRL-Chain ABSA Processing - Zero-Shot Inference")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Data: {DATA_PATH}")
    print(f"Save: {SAVE_DIR}")
    print(f"Output File: {OUTPUT_FILENAME}")
    print(f"Temp Prefix: {TEMP_PREFIX}")
    print("=" * 60)

    data = read_json(DATA_PATH)
    print(f"\nLoaded {len(data)} samples")

    if TEST_SAMPLES > 0:
        data = data[:TEST_SAMPLES]
        print(f"Running first {len(data)} samples for testing")

    process_batch_data(
        data=data,
        save_dir=SAVE_DIR,
        output_filename=OUTPUT_FILENAME,
        temp_prefix=TEMP_PREFIX,
        process_single_fn=get_completion,
        batch_size=BATCH_SIZE,
        progress_desc="Processing batch"
    )

    print(f"\nDone! Results saved to {SAVE_DIR}/{OUTPUT_FILENAME}")


