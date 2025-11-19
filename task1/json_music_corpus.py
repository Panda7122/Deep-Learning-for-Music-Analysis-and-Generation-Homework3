import os
import sys
import json
import glob
from typing import List, Tuple, Dict

import numpy as np
import torch
try:
    from miditok import TokSequence, Event  # optional, not required
except Exception:
    TokSequence = None
    Event = None

# Ensure Transformer-XL utils are importable
_TXL_DIR = os.path.join(os.path.dirname(__file__), '..', 'transformer-xl', 'pytorch')
if _TXL_DIR not in sys.path:
    sys.path.append(_TXL_DIR)
from utils.vocabulary import Vocab
from data_utils import LMShuffledIterator

# --- Helpers to convert TokSequence/Event <-> JSON like datas/*.json ---
def event2json(event) -> dict:
    # Accept either a miditok Event or a plain dict and standardize to dict
    if isinstance(event, list):
        return list(map(event2json, event))
    if isinstance(event, dict):
        return event
    # miditok Event-like object
    return {
        'type_': getattr(event, 'type', getattr(event, 'type_', None)),
        'value': getattr(event, 'value', None),
        'time': getattr(event, 'time', 0),
        'program': getattr(event, 'program', 0),
        'desc': getattr(event, 'desc', 0),
    }

def tokSeq2json(tok) -> dict:
    # Reference implementation provided by user; accept list for convenience
    if isinstance(tok, list):
        return list(map(tokSeq2json, tok))
    # Support either miditok.TokSequence or a simple object with attributes
    return {
        'tokens': getattr(tok, 'tokens', []),
        'ids': getattr(tok, 'ids', []),
        'bytes': getattr(tok, 'bytes', ''),
        'events': list(map(event2json, getattr(tok, 'events', []))),
        'are_ids_encoded': getattr(tok, 'are_ids_encoded', False),
        '_ticks_bars': getattr(tok, '_ticks_bars', []),
        '_ticks_beats': getattr(tok, '_ticks_beats', []),
        '_ids_decoded': getattr(tok, '_ids_decoded', []),
    }
def _load_json_tokens(path: str) -> List[List[str]]:
    sequences: List[List[str]] = []

    # If the file is a plain text file with comma-separated tokens, parse directly
    if path.lower().endswith('.txt'):
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Support multiple sequences separated by newlines optionally
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            toks = [t.strip() for t in line.split(',') if t.strip()]
            if toks:
                sequences.append(toks)
        return sequences

    # Otherwise assume JSON file
    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception:
            # Fallback: try reading raw text and treat as a single comma-separated sequence
            raw = f.read()
            toks = [t.strip() for t in raw.split(',') if t.strip()]
            if toks:
                sequences.append(toks)
            return sequences

    # Accept either list of objects, list of token-strings, or a single object
    if isinstance(data, list):
        # Could be a list of token-strings, or a list of objects with 'tokens'
        # Detect simple list-of-strings
        if data and all(isinstance(x, str) for x in data):
            sequences.append([str(x) for x in data])
        else:
            for item in data:
                if isinstance(item, dict) and 'tokens' in item:
                    sequences.append(list(map(str, item['tokens'])))
                elif isinstance(item, list) and all(isinstance(x, str) for x in item):
                    # Already a plain token list
                    sequences.append(list(map(str, item)))
                else:
                    # Try to derive tokens from an object with 'events'
                    if isinstance(item, dict) and 'events' in item:
                        toks: List[str] = []
                        for e in item.get('events', []):
                            t = str(e.get('type_', ''))
                            v = e.get('value')
                            if t:
                                tok = f"{t}_{v}" if v is not None and v != '' else t
                                toks.append(tok)
                        if toks:
                            sequences.append(toks)
    elif isinstance(data, dict):
        if 'tokens' in data and isinstance(data['tokens'], list):
            sequences.append(list(map(str, data['tokens'])))
        elif 'events' in data:
            toks: List[str] = []
            for e in data.get('events', []):
                t = str(e.get('type_', ''))
                v = e.get('value')
                if t:
                    tok = f"{t}_{v}" if v is not None and v != '' else t
                    toks.append(tok)
            if toks:
                sequences.append(toks)

    return sequences


def _split_indices(n: int, seed: int = 42, ratios: Tuple[float, float, float] = (0.95, 0.03, 0.02)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    train_idx = idx[:n_train]
    valid_idx = idx[n_train:n_train + n_valid]
    test_idx = idx[n_train + n_valid:]
    return train_idx, valid_idx, test_idx


class JSONTokenCorpus:
    """Loads token sequences from datas/*.json and builds a Vocab.

    Each sequence is a list of event strings (e.g., 'Bar_None', 'Position_0', 'Pitch_60', ...).
    """

    def __init__(self, data_dir: str, seed: int = 42):
        self.data_dir = data_dir
        self.seed = seed

        # Gather sequences
        paths = sorted(glob.glob(os.path.join(data_dir, '*.txt')))
        sequences: List[List[str]] = []
        for p in paths:
            sequences.extend(_load_json_tokens(p))

        if len(sequences) == 0:
            raise RuntimeError(f'No token sequences found under {data_dir}')

        # Split by sequence
        train_idx, valid_idx, test_idx = _split_indices(len(sequences), seed=seed)
        self._train_seqs = [sequences[i] for i in train_idx]
        self._valid_seqs = [sequences[i] for i in valid_idx]
        self._test_seqs = [sequences[i] for i in test_idx]

        # Build vocab from train+valid
        self.vocab = Vocab(special=[], lower_case=False)
        self.vocab.count_sents(self._train_seqs)
        if self._valid_seqs:
            self.vocab.count_sents(self._valid_seqs)
        self.vocab.build_vocab()

        # Encode
        self.train: List[torch.LongTensor] = self.vocab.encode_sents(self._train_seqs, ordered=False)
        self.valid: List[torch.LongTensor] = self.vocab.encode_sents(self._valid_seqs, ordered=False)
        self.test: List[torch.LongTensor] = self.vocab.encode_sents(self._test_seqs, ordered=False)

    def get_iterator(self, split: str, bsz: int, bptt: int, device='cpu', ext_len: int = 0, shuffle: bool = False):
        data = {'train': self.train, 'valid': self.valid, 'test': self.test}[split]
        return LMShuffledIterator(data, bsz, bptt, device=device, ext_len=ext_len, shuffle=(shuffle and split == 'train'))

    def save_vocab(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            for idx, sym in enumerate(self.vocab.idx2sym):
                f.write(f'{idx}\t{sym}\n')

    @staticmethod
    def load_vocab(path: str) -> Dict[str, int]:
        sym2idx: Dict[str, int] = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx_str, sym = line.split('\t', 1)
                sym2idx[sym] = int(idx_str)
        return sym2idx
