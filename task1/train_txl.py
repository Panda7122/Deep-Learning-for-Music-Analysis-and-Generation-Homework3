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


def format_ppl(loss: float) -> str:
    try:
        return f"ppl {math.exp(loss):.2f}"
    except OverflowError:
        return "ppl inf"


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

    # Build model
    model = MemTransformerLMCompat(
        n_token=ntokens,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_head=args.d_head,
        d_inner=args.d_inner,
        dropout=args.dropout,
        dropatt=args.dropatt,
        tie_weight=True,
        d_embed=args.d_model,
        div_val=1,
        tie_projs=[False],
        pre_lnorm=True,
        tgt_len=args.tgt_len,
        ext_len=args.ext_len,
        mem_len=args.mem_len,
        cutoffs=[],
        adapt_inp=False,
        same_length=args.same_length,
        attn_type=0,
        clamp_len=args.clamp_len,
        sample_softmax=-1,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def set_lr(lr: float):
        for pg in optimizer.param_groups:
            pg['lr'] = lr

    base_lr = args.lr
    global_step = 0

    # Iterators
    tr_iter = corpus.get_iterator('train', bsz=args.batch_size, bptt=args.tgt_len, device=device, ext_len=args.ext_len, shuffle=True)
    va_iter = corpus.get_iterator('valid', bsz=args.eval_batch_size, bptt=min(args.tgt_len, 128), device=device, ext_len=args.ext_len, shuffle=False)

    # Train loop
    best_val = float('inf')
    step = 0
    training_losses = []
    valid_losses = []
    for ep in range(1, args.epochs + 1):
        model.train()
        mems = tuple()
        running_loss = 0.0
        tokens = 0
        start_time = time.time()

        for (data, target, seq_len) in tr_iter:
            step += 1
            global_step += 1

            # Simple warmup
            if args.warmup_steps > 0 and global_step <= args.warmup_steps:
                set_lr(base_lr * float(global_step) / float(args.warmup_steps))
            elif args.cosine_anneal:
                # Cosine anneal per-epoch (no total step info) between base_lr and base_lr*0.1
                # This is a light heuristic; you can replace with a proper scheduler if desired.
                t = (global_step % max(1, args.log_interval)) / float(max(1, args.log_interval))
                lr_now = base_lr * (0.55 + 0.45 * (1 + math.cos(math.pi * t)) / 2)
                set_lr(lr_now)
            optimizer.zero_grad()
            # print(data, target, mems)
            ret = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            # Guard against non-finite loss
            if not torch.isfinite(loss):
                print('Non-finite loss encountered. Reducing LR by 2x and skipping this batch.')
                set_lr(max(1e-7, optimizer.param_groups[0]['lr'] * 0.5))
                mems = tuple()
                continue

            (loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            running_loss += loss.item() * seq_len
            tokens += seq_len

            if step % args.log_interval == 0:
                cur_loss = running_loss / max(tokens, 1)
                elapsed = time.time() - start_time
                print(f"| epoch {ep:3d} step {step:7d} | ms/batch {1000.0*elapsed/args.log_interval:.2f} | loss {cur_loss:.3f} | {format_ppl(cur_loss)}")
                running_loss = 0.0
                tokens = 0
                start_time = time.time()

            if step % args.eval_interval == 0:
                val_loss = evaluate(model, va_iter)
                print('=' * 100)
                print(f'| Eval at step {step}: val loss {val_loss:.3f} | {format_ppl(val_loss)}')
                print('=' * 100)
                # Save best
                if val_loss < best_val:
                    best_val = val_loss
                    ckpt_path = os.path.join(args.work_dir, 'model.pt')
                    torch.save({'state_dict': model.state_dict(),
                                'config': {
                                    'n_token': ntokens,
                                    'n_layer': args.n_layer,
                                    'n_head': args.n_head,
                                    'd_model': args.d_model,
                                    'd_head': args.d_head,
                                    'd_inner': args.d_inner,
                                    'dropout': args.dropout,
                                    'dropatt': args.dropatt,
                                    'tgt_len': args.tgt_len,
                                    'ext_len': args.ext_len,
                                    'mem_len': args.mem_len,
                                    'same_length': args.same_length,
                                    'clamp_len': args.clamp_len,
                                }}, ckpt_path)
                    print(f'Saved best checkpoint (state_dict) to {ckpt_path}')

        # End epoch eval
        training_losses.append(best_val)
        val_loss = evaluate(model, va_iter)
        valid_losses.append(val_loss)
        print(f'| End epoch {ep}: val loss {val_loss:.3f} | {format_ppl(val_loss)}')
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.work_dir, 'model.pt')
            torch.save({'state_dict': model.state_dict(),
                        'config': {
                            'n_token': ntokens,
                            'n_layer': args.n_layer,
                            'n_head': args.n_head,
                            'd_model': args.d_model,
                            'd_head': args.d_head,
                            'd_inner': args.d_inner,
                            'dropout': args.dropout,
                            'dropatt': args.dropatt,
                            'tgt_len': args.tgt_len,
                            'ext_len': args.ext_len,
                            'mem_len': args.mem_len,
                            'same_length': args.same_length,
                            'clamp_len': args.clamp_len,
                        }}, ckpt_path)
            print(f'Saved best checkpoint (state_dict) to {ckpt_path}')
        
        # Plot loss curves and save figure
    matplotlib.use('Agg')

    ep_idxs = list(range(1, len(valid_losses) + 1))
    plt.figure(figsize=(7, 4))
    if training_losses:
        plt.plot(ep_idxs, training_losses, marker='o', label='train (best so far)')
    plt.plot(ep_idxs, valid_losses, marker='x', label='valid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(args.work_dir, 'loss_curve.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved loss curve to {out_path}")
def evaluate(model, eval_iter) -> float:
    model.eval()
    total_len, total_loss = 0, 0.0
    with torch.no_grad():
        mems = tuple()
        for (data, target, seq_len) in eval_iter:
            ret = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            total_loss += seq_len * loss.item()
            total_len += seq_len
    return total_loss / max(total_len, 1)


if __name__ == '__main__':
    main()
