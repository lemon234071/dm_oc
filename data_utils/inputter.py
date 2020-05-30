# -*- coding: utf-8 -*-
import os
import json
import logging
from itertools import chain
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

from data_utils.dataset_wb import WBDataset, WBdistDataset

from data_utils.data_process import build_personachat

SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]"]

logger = logging.getLogger(__file__)


def collate(dataset, pad_id, batach_first=False):
    tensor_dataset = []
    for input_name in dataset.keys():
        if "pad" in input_name:
            if "label" in input_name:
                input_tensor = pad_sequence(
                    [torch.tensor(feature, dtype=torch.long) for feature in dataset[input_name]],
                    batch_first=batach_first, padding_value=-1)
            else:
                input_tensor = pad_sequence(
                    [torch.tensor(feature, dtype=torch.long) for feature in dataset[input_name]],
                    batch_first=batach_first, padding_value=pad_id)
        else:
            input_tensor = torch.tensor(dataset[input_name], dtype=torch.long)
        tensor_dataset.append(input_tensor)
    return tensor_dataset


def build_dataloader(dataset, batch_size, shuffle, pad_id, batch_first):
    """ Prepare the dataset for training and evaluation """

    logger.info("Pad inputs and convert to Tensor")
    tensor_dataset = collate(dataset, pad_id, batch_first)
    dataset = TensorDataset(*tensor_dataset)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_dataloaders(args, tokenizer):
    logger.info("Build train and validation dataloaders")

    datasets, raw_samples = get_data(tokenizer, args.data_path, args.dataset_cache)
    train_dataset, valid_dataset = WBDataset(datasets["train"], tokenizer), WBDataset(datasets["valid"], tokenizer)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset,
                              sampler=train_sampler,
                              collate_fn=train_dataset.collate,
                              num_workers=args.num_workers,
                              batch_size=args.train_batch_size,
                              shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler,
                              collate_fn=valid_dataset.collate,
                              num_workers=args.num_workers,
                              batch_size=args.valid_batch_size,
                              shuffle=False)

    return train_loader, valid_loader, train_sampler, valid_sampler


def build_dist_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    logger.info("Build train and validation dataloaders")

    train_dataset = WBdistDataset(tokenizer, data_path=args.train_path)
    valid_dataset = WBdistDataset(tokenizer, data_path=args.valid_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset,
                              collate_fn=train_dataset.collate,
                              pin_memory=(args.device == "cuda"),
                              num_workers=args.num_workers,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset,
                              collate_fn=valid_dataset.collate,
                              pin_memory=(args.device == "cuda"),
                              num_workers=args.num_workers,
                              sampler=valid_sampler,
                              batch_size=args.valid_batch_size,
                              shuffle=False)
    return train_loader, valid_loader, train_sampler, valid_sampler
