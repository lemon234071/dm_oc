# -*- coding: utf-8 -*-
import os
import json
import logging
import collections
from itertools import chain

import torch
from transformers import cached_path

logger = logging.getLogger(__file__)

DATASETS_URL = {
    "personachat": "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"}
MAX_HISTORY = 15


def get_data(tokenizer, dataset_path, dataset_cache, dataset_name):
    """ Get tokenized dataset from URL or cache."""
    dataset_path = dataset_path or DATASETS_URL[dataset_name]
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        cache_file = cached_path(dataset_path)
        with open(cache_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)

        dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)
    return dataset


def build_personachat(data, tokenizer):
    eos = tokenizer.eos_token_id
    dataset = collections.defaultdict(list)

    for dialog in data:
        for utterance in dialog["utterances"]:
            session = utterance["history"][-(2 * MAX_HISTORY + 1):] + [
                utterance["candidates"][-1]]  # [-(2 * args.max_history + 1):]
            sequence = [s + [eos] for s in session]
            dataset["input_ids_pad"].append(list(chain(*sequence)))
            dataset["token_type_ids_pad"].append([j for i, s in enumerate(sequence) for j in [i] * len(s)])
            dataset["lm_labels_pad"].append(([-1] * sum(len(s) for s in sequence[:-1])) + sequence[-1])
            assert len(dataset["input_ids_pad"][-1]) == len(dataset["token_type_ids_pad"][-1]) == len(
                dataset["lm_labels_pad"][-1])
    dataset["all_ids"] = [i for i in range(len(dataset["lm_labels_pad"]))]
    return dataset
