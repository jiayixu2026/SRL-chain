import os
import json
import time

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


def load_config():
    if HAS_DOTENV:
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)
    return os.environ


def get_config(key, default=None):
    return os.environ.get(key, default)


def write_json(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def safe_str_concat(*args):
    return "".join([str(arg) if arg is not None else "" for arg in args])


def get_completion_from_messages(messages, model="gpt-3.5-turbo-1106",
                                  max_retries=3, temperature=0,
                                  return_token_usage=False):
    import openai

    for retry in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=800,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            response_content = response.choices[0].message.content
            if not response_content or response_content.strip() == "":
                raise ValueError("API returned empty content")

            if return_token_usage:
                input_tokens = response.usage.prompt_tokens
                generated_tokens = response.usage.completion_tokens
                return response_content, input_tokens, generated_tokens
            else:
                return response_content

        except Exception as e:
            error_msg = str(e)
            if "SSL" in error_msg or "Connection" in error_msg or "EOF" in error_msg \
                    or "empty content" in error_msg or "rate limit" in error_msg.lower():
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 2
                    print(f"\n[Warning] Connection error/Rate limit (attempt {retry + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            print(f"\n[Fatal Error] API failed after {max_retries} retries: {e}")
            raise RuntimeError(f"API call failed after {max_retries} retries: {e}")
