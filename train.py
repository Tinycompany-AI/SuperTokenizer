
# coding: utf-8
# Copyright IsNoobGrammer and aloobun, 2025


"""
SuperTokenizer Training Script

This script now loads all configuration from 'train.yaml'.
Edit 'train.yaml' to change training parameters, tokenizer settings, and pretokenization behavior.
"""
import os
import re
import yaml
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    trainers,
    normalizers,
    processors,
    decoders,
    Regex,
)
from datasets import load_dataset
from helper import *

def load_config(yaml_path: str = "train.yaml") -> dict:
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config YAML '{yaml_path}' not found.")
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("YAML config must be a dictionary at the top level.")
    return config

config = load_config()
print("Loaded config:\n", yaml.dump(config, sort_keys=False))

SEPARATOR = config.get("separator", "#$BiBo@&")
VOCAB_SIZE = config.get("vocab_size", 105900)
SPECIAL_TOKENS = config.get("special_tokens", ["<|endoftext|>"])
DATASET_NAME = config.get("dataset_name", "fhai50032/pds-tk")
TEXT_COLUMN = config.get("text_column", "text")
AUG_MAX_CHUNK_LEN = config.get("aug_max_chunk_len", 25)
test_distribution = config.get("test_distribution", {"1-5": 0.02, "6-18": 0.70, "18-28": 0.28})
OUTPUT_PATH = config.get("output_path", "my_trained_tokenizer_2")
PUSH_TO_HUB = config.get("push_to_hub", {})
MIN_FREQUENCY = config.get("min_frequency", 2)
SHOW_PROGRESS = config.get("show_progress", True)
INITIAL_ALPHABET = config.get("initial_alphabet", None)
MAX_TOKEN_LENGTH = config.get("max_token_length", min(32, int((AUG_MAX_CHUNK_LEN*1.2)//1)))
CONTINUING_SUBWORD_PREFIX = config.get("continuing_subword_prefix", "")
MODEL_MAX_LENGTH = config.get("model_max_length", 8192)

print(f"Tokenizer configured to split on separator: '{SEPARATOR}' and remove it.")

# --- Visualization: Plot the test_distribution and overlay a Gaussian baseline ---
def plot_distribution_with_gaussian(test_distribution: dict):
    import matplotlib.pyplot as plt
    import numpy as np
    # Parse bins and probabilities
    bins = []
    probs = []
    for rng, p in test_distribution.items():
        if '-' in rng:
            a, b = map(int, rng.split('-'))
            bins.append((a+b)/2)
        else:
            bins.append(int(rng))
        probs.append(float(p))
    bins = np.array(bins)
    probs = np.array(probs)

    # Estimate mean and stddev from the distribution
    mean = np.sum(bins * probs) / np.sum(probs)
    std = np.sqrt(np.sum(((bins - mean) ** 2) * probs) / np.sum(probs))
    min_bin = min(bins)
    max_bin = max(bins)
    x = np.linspace(min_bin, max_bin, 200)
    gauss = (1/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean)/std)**2)
    gauss = gauss / gauss.sum() * probs.sum()  # Normalize area to match

    plt.figure(figsize=(8,5))
    plt.bar(bins, probs, width=1.5, alpha=0.6, label='YAML Distribution')
    plt.plot(x, gauss, 'r--', label=f'Gaussian (μ={mean:.1f}, σ={std:.1f})')
    plt.xlabel('Chunk Length (chars)')
    plt.ylabel('Probability')
    plt.title('Character Chunk Split Probability Distribution')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot the distribution before training
plot_distribution_with_gaussian(test_distribution)



tokenizer = Tokenizer(
    models.BPE(byte_fallback=True, unk_token=None, fuse_unk=False)
)

tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFC(),
])

tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(
        pattern=Regex(re.escape(SEPARATOR)),
        behavior="removed",
        invert=False,
    ),
    pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        trim_offsets=True,
        use_regex=False
    ),
])

tokenizer.post_processor = processors.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)
tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
    min_frequency=MIN_FREQUENCY,
    show_progress=SHOW_PROGRESS,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet() if INITIAL_ALPHABET is None else INITIAL_ALPHABET,
    max_token_length=MAX_TOKEN_LENGTH,
    continuing_subword_prefix=CONTINUING_SUBWORD_PREFIX,
)

# print("Tokenizer components configured with behavior='removed'.")

def get_corpus():
    from datasets import concatenate_datasets
    ds1 = load_dataset(DATASET_NAME, split="train")
    shuffled = ds1.map(lambda x: {TEXT_COLUMN: augment_text_with_distribution(x[TEXT_COLUMN], test_distribution, SEPARATOR)}, num_proc=8)
    print("ds")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    return [text for text in shuffled[TEXT_COLUMN] if isinstance(text, str) and text.strip()]

# ds=get_corpus()[0]
# print(ds)

print("Starting tokenizer training (using Probabilistic spliting )...")

os.environ["TOKENIZERS_PARALLELISM"] = "true"
tokenizer.train_from_iterator(
    get_corpus(),
    trainer=trainer,
    length=None 
)

tokenizer.save(OUTPUT_PATH)
print(f"Tokenizer saved to: {OUTPUT_PATH}")

from transformers import PreTrainedTokenizerFast
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    padding_side="right",
    model_max_length=MODEL_MAX_LENGTH
)
tk1 = hf_tokenizer

if PUSH_TO_HUB.get("repo_id") and PUSH_TO_HUB.get("token"):
    tk1.push_to_hub(
        PUSH_TO_HUB["repo_id"],
        token=PUSH_TO_HUB["token"],
        split=PUSH_TO_HUB.get("split", "train")
    )
    print(f"Tokenizer pushed to hub: {PUSH_TO_HUB['repo_id']}")
else:
    print("Skipping push_to_hub (missing repo_id or token in config)")

samples = [
    "किलोमीटर=1000मीटर",  # Hindi with numbers
    "def add(a,b):\n    return a+b",  # Code
    "I'm running 1000s of tests!",  # English contractions
    "<|im_start|>user Hello<|im_end|>",  # ChatML
    "Heklo worlD3s 432563 7843 8846#773$$%",
    "😜🫤☹️😖🤢🤮😇🐻‍❄️🦄🐾🐽🐍🦞🦐🦿🤴🧑‍🦲👨‍🚒👨‍🚀",
    "ॐ त्र्यम्बकं यजामहे सुगन्धिं पुष्टिवर्धनम् उर्वारुकमिव बन्धनान्मृत्योर्मुक्षीय मामृतात् ॐ. ",
    """<think>
What da fffffff
</think>
you still testing?"""
]

for text in samples:
    encoded = tk1.encode(text)
    print(f"\nInput  : {text}")
    print(f"Ids: {encoded}")
    print(f"Tokens: {tk1.tokenize(text)}")
    print(f"Decoded: {tk1.decode(encoded)}")
