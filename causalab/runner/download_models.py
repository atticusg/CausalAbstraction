"""Download HuggingFace models to cache."""

import argparse
import os

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Download models to HuggingFace cache")
    parser.add_argument("models", nargs="+", help="Model names to download")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    for model_name in tqdm(args.models, desc="Downloading models"):
        AutoTokenizer.from_pretrained(model_name, token=hf_token)
        AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)


if __name__ == "__main__":
    main()
