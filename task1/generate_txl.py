import os
import sys
import argparse
import time
from typing import List, Tuple
from miditok import REMI, TokSequence,Event
from miditok import TokenizerConfig
from symusic import Score
import torch
import torch.nn.functional as F
import json

# Add TXL path
_BASE_DIR = os.path.dirname(__file__)
_TXL_DIR = os.path.join(_BASE_DIR, '..', 'transformer-xl', 'pytorch')
_TXL_UTILS_DIR = os.path.join(_TXL_DIR, 'utils')
for p in (_TXL_DIR, _TXL_UTILS_DIR):
    if p not in sys.path:
        sys.path.append(p)
from compat_txl import MemTransformerLMCompat
from proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax


def load_vocab_tsv(path: str):
    idx2sym: List[str] = []
    sym2idx = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, sym = line.split('\t', 1)
            idx = int(idx_str)
            while len(idx2sym) <= idx:
                idx2sym.append('')
            idx2sym[idx] = sym
            sym2idx[sym] = idx
    return idx2sym, sym2idx


def top_k_sample(logits: torch.Tensor, k: int = 50, temperature: float = 1.0) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    print(f"num of element{logits.numel()}")
    if 0 < k < logits.numel():
        topk = torch.topk(logits, k)
        probs = F.normalize(topk.values, dim=-1)
        print("probs:", end='')
        print(probs)
        idx = torch.multinomial(probs, 1).item()
        return int(topk.indices[idx].item())
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def generate_32_bars(model: MemTransformerLMCompat, sym2idx, idx2sym, start_tokens: List[str], tgt_len: int, device: torch.device, temperature: float, top_k: int) -> Tuple[List[str], List[int]]:
    model.eval()
    print(f"sym2idx: {sym2idx}")
    with torch.no_grad():
        start_ids = [sym2idx.get('0', 0)]

        mems = model.init_mems()
        inp = torch.tensor(start_ids, dtype=torch.long, device=device).view(-1, 1)
        core_out, mems = model._forward(inp, mems=mems)

        generated: List[int] = list(start_ids)
        bar_sym = '0'
        bar_idx = sym2idx.get(bar_sym, None)
        bar_count = sum(1 for i in generated if (bar_idx is not None and i == bar_idx))

        max_steps = 20000
        steps = 0
        while steps < max_steps:
            steps += 1
            last_id = torch.tensor([generated[-1]], dtype=torch.long, device=device).view(1, 1)
            core_out, mems = model._forward(last_id, mems=mems)
            hid = core_out[-1].squeeze(0)
            crit: ProjectedAdaptiveLogSoftmax = model.crit
            logits = crit._compute_logit(
                hid, crit.out_layers[0].weight, crit.out_layers[0].bias, crit.out_projs[0]
            )
            print(logits)
            next_id = top_k_sample(logits.view(-1), k=top_k, temperature=temperature)
            print(next_id)
            generated.append(next_id)

            if bar_idx is not None and next_id == bar_idx:
                bar_count += 1
                if bar_count >= 33:
                    break

        if bar_idx is not None:
            count = 0
            cutoff = len(generated)
            for i, t in enumerate(generated):
                if t == bar_idx:
                    count += 1
                    if count == 33:
                        cutoff = i
                        break
            generated = generated[:cutoff]

        return [idx2sym[i] for i in generated], generated


def _parse_rest_ticks(rest_val: str, ticks_per_beat: int = 8) -> int:
    try:
        # Expect formats like "1.0.4", "2.0.2", etc. Use the first integer segment as beats
        beat_part = rest_val.split(".", 1)[0]
        beats = int(float(beat_part))
        return beats * ticks_per_beat
    except Exception:
        return ticks_per_beat

def _build_events_and_ticks(tokens: List[str]):
    # Build events timeline similar to datas/*.json with simple tick semantics
    # Assumptions:
    # - 4 beats per bar
    # - 8 ticks per beat (32 ticks per bar)
    TICKS_PER_BEAT = 8
    TICKS_PER_BAR = 32

    events: List[Event] = []
    ticks_bars: List[int] = []
    time_ticks = 0
    bar_start = 0
    bars_seen = 0

    def add_event(type_: str, value, desc):
        events.append(Event(type_, value, time_ticks, 0, desc))

    for tok in tokens:
        if '_' in tok:
            ttype, tval = tok.split('_', 1)
        else:
            ttype, tval = tok, ''

        if ttype == 'Bar':
            if bars_seen == 0:
                bar_start = time_ticks
            else:
                bar_start += TICKS_PER_BAR
                time_ticks = bar_start
            ticks_bars.append(bar_start)
            bars_seen += 1
            add_event('Bar', 'None', 0)
        elif ttype == 'Position':
            try:
                pos = int(float(tval))
            except Exception:
                pos = 0
            time_ticks = bar_start + pos
            add_event('Position', pos, pos)
        elif ttype == 'Tempo':
            try:
                tempo_val = float(tval)
            except Exception:
                tempo_val = tval
            add_event('Tempo', tempo_val, tempo_val)
        elif ttype == 'Rest':
            ticks = _parse_rest_ticks(tval, TICKS_PER_BEAT)
            add_event('Rest', tval, f'{ticks} ticks')
            time_ticks += ticks
        else:
            val: object = tval
            if tval.replace('.', '', 1).isdigit():
                try:
                    if '.' in tval:
                        val = float(tval)
                    else:
                        val = int(tval)
                except Exception:
                    val = tval
            add_event(ttype, val, val)

    last_time = time_ticks
    if len(ticks_bars) == 0:
        ticks_bars.append(0)
    while ticks_bars[-1] + TICKS_PER_BAR <= ((last_time + TICKS_PER_BAR - 1) // TICKS_PER_BAR) * TICKS_PER_BAR:
        ticks_bars.append(ticks_bars[-1] + TICKS_PER_BAR)

    ticks_beats = list(range(0, last_time + TICKS_PER_BEAT, TICKS_PER_BEAT))
    return events, ticks_bars, ticks_beats


# TokSequence-style JSON builders (reference to user's function)
def event2json(event) -> dict:
    if isinstance(event, list):
        return list(map(event2json, event))
    if isinstance(event, dict):
        return event
    return {
        'type_': getattr(event, 'type', getattr(event, 'type_', None)),
        'value': getattr(event, 'value', None),
        'time': getattr(event, 'time', 0),
        'program': getattr(event, 'program', 0),
        'desc': getattr(event, 'desc', 0),
    }

def tokSeq2json(tok):
    if isinstance(tok, list):
        return list(map(tokSeq2json, tok))
    return {
        'tokens': tok.tokens,
        'ids': tok.ids,
        'bytes': tok.bytes,
        'events': list(map(event2json, tok.events)),
        'are_ids_encoded': tok.are_ids_encoded,
        '_ticks_bars': tok._ticks_bars,
        '_ticks_beats': tok._ticks_beats,
        '_ids_decoded': tok._ids_decoded,
    }


def get_args():
    p = argparse.ArgumentParser(description='Generate 32 bars from trained Transformer-XL model')
    p.add_argument('--work_dir', type=str, required=True, help='directory containing model.pt and vocab.tsv')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_k', type=int, default=50)
    p.add_argument('--seed_tokens', type=str, nargs='*', default=['Bar_None', 'Position_0'])
    p.add_argument('--tgt_len', type=int, default=256)
    p.add_argument('--out', type=str, default=None, help='path for plain text tokens (one per line)')
    p.add_argument('--json_out', action='store_true', help='also write JSON output; always includes tokens and ids')
    p.add_argument('--wrap_list', action='store_true', help='wrap JSON as a list with one object ([{"tokens": [...] }])')
    p.add_argument('--datas_json', action='store_true', help='write datas-style JSON with tokens, ids, events, ticks (always wrapped list)')
    p.add_argument('--out_json', type=str, default=None, help='path for JSON output (defaults to samples/sample_<ts>.json)')
    p.add_argument('--out_midi', type=str, default=None, help='path for MIDI output (defaults to samples/sample_<ts>.json)')
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')
    pitch_range = range(21, 109)
    beat_res = {(0,4): 8, (4,12): 4}
    num_velocities = 32
    config = TokenizerConfig(
        pitch_range=pitch_range, 
        beat_res=beat_res, 
        num_velocities=num_velocities, 
        use_chords=True, 
        use_rests=True,
        use_tempos=True,
        num_tempos=32,
        tempo_range=(40, 250),
        use_programs=False
    )


    tokenizer = REMI(config)
    ckpt_path = os.path.join(args.work_dir, 'model.pt')
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict) and 'state_dict' in obj and 'config' in obj:
        cfg = obj['config']
        model = MemTransformerLMCompat(
            n_token=cfg['n_token'],
            n_layer=cfg['n_layer'],
            n_head=cfg['n_head'],
            d_model=cfg['d_model'],
            d_head=cfg['d_head'],
            d_inner=cfg['d_inner'],
            dropout=cfg['dropout'],
            dropatt=cfg['dropatt'],
            tie_weight=True,
            d_embed=cfg['d_model'],
            div_val=1,
            tie_projs=[False],
            pre_lnorm=True,
            tgt_len=cfg['tgt_len'],
            ext_len=cfg['ext_len'],
            mem_len=cfg['mem_len'],
            cutoffs=[],
            adapt_inp=False,
            same_length=cfg['same_length'],
            attn_type=0,
            clamp_len=cfg['clamp_len'],
            sample_softmax=-1,
        ).to(device)
        model.load_state_dict(obj['state_dict'])
    else:
        # Fallback: full object was saved
        model = obj.to(device)
    model.eval()

    idx2sym, sym2idx = load_vocab_tsv(os.path.join(args.work_dir, 'vocab.tsv'))
    # Quick sanity checks to help diagnose <UNK>-only outputs
    if '0' not in sym2idx:
        print("[WARN] 'Bar_None' not found in vocab. Seed/context may become <UNK>. Consider rebuilding vocab from datas/.")
    model.reset_length(args.tgt_len, model.ext_len, model.mem_len)

    toks, ids = generate_32_bars(model, sym2idx, idx2sym, args.seed_tokens, args.tgt_len, device, args.temperature, args.top_k)

    os.makedirs(os.path.join(args.work_dir, 'samples'), exist_ok=True)
    ts = int(time.time())
    out_txt = args.out or os.path.join(args.work_dir, 'samples', f'sample_{ts}.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        for t in toks:
            f.write(t + '\n')
    print(f'Wrote generated 32 bars tokens to {out_txt}')

    # If everything is <UNK>, hint likely vocab/data mismatch
    if all(t == '<UNK>' for t in toks):
        print('[WARN] All generated tokens decoded to <UNK>. This usually means the vocab.tsv is invalid or was built incorrectly.')
        print('       Fix: retrain after ensuring datas/*.json are parsed correctly, which rebuilds vocab.tsv, then regenerate.')

    if args.json_out or args.datas_json:
        out_json = args.out_json or os.path.join(args.work_dir, 'samples', f'sample_{ts}.json')
        out_midi = args.out_midi or os.path.join(args.work_dir, 'samples', f'sample_{ts}.mid')
        if args.datas_json:
            events, ticks_bars, ticks_beats = _build_events_and_ticks(toks)
            tok = TokSequence(
                tokens=toks,
                ids=ids,
                bytes='',
                events=events,
                are_ids_encoded=False,
                _ticks_bars=ticks_bars,
                _ticks_beats=ticks_beats,
                _ids_decoded=[],
            )
            
            payload = [tok]
        else:
            obj = { 'tokens': toks, 'ids': ids }
            payload = [obj] if args.wrap_list else obj
        score = tokenizer([tok])
        score.dump_midi(out_midi)
        if args.datas_json:
            print(f'Wrote midi to {out_midi}')
        else:
            print(f'Wrote midi output to {out_midi} (wrap_list={args.wrap_list})')
        with open(out_json, 'w', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False, indent=2))
        if args.datas_json:
            print(f'Wrote datas-style JSON to {out_json}')
        else:
            print(f'Wrote JSON output to {out_json} (wrap_list={args.wrap_list})')


if __name__ == '__main__':
    main()
