import json
import re
from collections import Counter

from datasets import load_dataset


DATASET_NAME = "aitetic/bookcorpus"
SPLIT = "train"



def main():

    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    total_sentences = len(dataset)
    print(total_sentences)


if __name__ == "__main__":
    main()
