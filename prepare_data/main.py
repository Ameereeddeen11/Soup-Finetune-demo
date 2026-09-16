"""
This script downloads dataset from HuggingFace and prepares it as JSONL format for training with Soup
"""
import json
from pathlib import Path
from datasets import load_dataset

SAMPLE_SIZE = 1000
OUTPUT_PATH = Path("data/train.jsonl")

def main():
    print("Downloading dataset from HuggingFace...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    print(f"The entire dataset has {len(dataset)} samples.")

    dataset = dataset.shuffle(seed=42).select(range(SAMPLE_SIZE))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in dataset:
            record = {
                "instruction": row["instruction"],
                "input": row["input"],
                "output": row["output"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Done: {SAMPLE_SIZE} samples written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()