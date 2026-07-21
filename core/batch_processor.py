import os
import json
from tqdm import tqdm

from .api_utils import write_json, read_json


def process_batch_data(data, save_dir, output_filename, temp_prefix,
                        process_single_fn, batch_size=100,
                        progress_desc="Processing batch"):
    os.makedirs(save_dir, exist_ok=True)
    total_batches = (len(data) + batch_size - 1) // batch_size

    for i in range(0, len(data), batch_size):
        batch_idx = i // batch_size
        temp_filename = f'{temp_prefix}_{batch_idx}.jsonl'
        temp_filepath = os.path.join(save_dir, temp_filename)

        processed_count = 0
        if os.path.exists(temp_filepath):
            with open(temp_filepath, 'r', encoding='utf-8') as f:
                processed_count = sum(1 for _ in f)
            print(f"[Resume] Batch {batch_idx + 1}/{total_batches}: "
                  f"{processed_count} samples already processed, continuing...")
            is_last_batch = (batch_idx == total_batches - 1)
            samples_in_batch = len(data) - i if is_last_batch else batch_size
            if processed_count >= samples_in_batch:
                print(f"[Skip] Batch {batch_idx + 1}/{total_batches} already fully processed, skipping...")
                continue

        batch_data = data[i:i + batch_size]
        start_offset = processed_count

        mode = 'a' if processed_count > 0 else 'w'
        with open(temp_filepath, mode, encoding='utf-8') as f_out:
            remaining_data = batch_data[start_offset:]
            if len(remaining_data) <= 0:
                continue

            for idx_in_batch, text in enumerate(tqdm(
                    remaining_data, total=len(remaining_data),
                    desc=f"{progress_desc} {batch_idx + 1}/{total_batches}")):
                try:
                    output_entry = process_single_fn(text)
                    f_out.write(json.dumps(output_entry, ensure_ascii=False) + '\n')
                    f_out.flush()
                except RuntimeError as e:
                    print(f"\n[!!! STOPPED !!!] Program halted at batch {batch_idx + 1}, "
                          f"sample {start_offset + idx_in_batch + 1}.")
                    print(f"Error details: {e}")
                    print(f"Already saved {processed_count + idx_in_batch} samples in this batch.")
                    print("You can re-run the script to continue from where it left off.")
                    exit(1)
                except KeyboardInterrupt:
                    print(f"\n[!!! STOPPED !!!] Manual interrupt at batch {batch_idx + 1}, "
                          f"sample {start_offset + idx_in_batch + 1}.")
                    print(f"Already saved {processed_count + idx_in_batch} samples in this batch.")
                    exit(1)

        batch_json_array = []
        with open(temp_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    batch_json_array.append(json.loads(line))
        batch_json_path = temp_filepath.replace('.jsonl', '.json')
        write_json(batch_json_path, batch_json_array)
        print(f"[Success] Batch {batch_idx + 1}/{total_batches} completed and saved as {batch_json_path}")

    print("\nAll batches completed successfully! Merging files...")
    final_data = []
    for i in range(0, len(data), batch_size):
        batch_json_path = os.path.join(save_dir, f'{temp_prefix}_{i // batch_size}.json')
        if os.path.exists(batch_json_path):
            batch_data = read_json(batch_json_path)
            final_data.extend(batch_data)
        else:
            jsonl_path = os.path.join(save_dir, f'{temp_prefix}_{i // batch_size}.jsonl')
            if os.path.exists(jsonl_path):
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    batch_data = [json.loads(line) for line in f if line.strip()]
                final_data.extend(batch_data)

    final_filepath = os.path.join(save_dir, output_filename)
    write_json(final_filepath, final_data)
    print(f"Final results saved to: {final_filepath} (total {len(final_data)} samples)")
    print("Temporary files retained – you may delete them manually when no longer needed.")

    return final_filepath
