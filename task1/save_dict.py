import os
import sys
import time
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from json_music_corpus import JSONTokenCorpus
import matplotlib.pyplot as plt
# from transformers import TransfoXLConfig, TransfoXLModel
import torch
# Add Transformer-XL path
_BASE_DIR = os.path.dirname(__file__)
_TXL_DIR = os.path.join(_BASE_DIR, '..', 'transformer-xl', 'pytorch')
_TXL_UTILS_DIR = os.path.join(_TXL_DIR, 'utils')
for p in (_TXL_DIR, _TXL_UTILS_DIR):
    if p not in sys.path:
        sys.path.append(p)
from compat_txl import MemTransformerLMCompat
import matplotlib
def get_args():
    p = argparse.ArgumentParser(description='Train Transformer-XL on JSON event tokens')
    p.add_argument('--data_dir', type=str, default=os.path.join('datas'), help='JSON dataset directory')
    p.add_argument('--work_dir', type=str, default=os.path.join('task1', 'runs', 'txl_music'), help='output directory')
    p.add_argument('--seed', type=int, default=1111)
    p.add_argument('--cuda', action='store_true')
    # Model
    p.add_argument('--n_layer', type=int, default=8)
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--d_head', type=int, default=64)
    p.add_argument('--d_inner', type=int, default=2048)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--dropatt', type=float, default=0.1)
    p.add_argument('--tgt_len', type=int, default=256)
    p.add_argument('--mem_len', type=int, default=256)
    p.add_argument('--ext_len', type=int, default=0)
    p.add_argument('--same_length', action='store_true')
    p.add_argument('--clamp_len', type=int, default=-1)
    # Train
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--eval_batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--clip', type=float, default=0.25)
    p.add_argument('--warmup_steps', type=int, default=0)
    p.add_argument('--cosine_anneal', action='store_true')
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--eval_interval', type=int, default=100)
    return p.parse_args()

def main():
    args = get_args()
    os.makedirs(args.work_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

    print(f'Loading JSON corpus from {args.data_dir} ...')
    corpus = JSONTokenCorpus(args.data_dir, seed=args.seed)
    ntokens = len(corpus.vocab)
    print(f'Vocab size: {ntokens}')

    # Persist vocab for generation
    vocab_path = os.path.join(args.work_dir, 'vocab.tsv')
    corpus.save_vocab(vocab_path)