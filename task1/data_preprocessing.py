from miditok import REMI, TokenizerConfig,CPWord
from miditok.utils import split_files_for_training
from torch.utils.data import DataLoader
from pathlib import Path
from miditok import TokSequence,Event
from miditoolkit import MidiFile
from symusic import Score
from tqdm import tqdm
import json
import os
import numpy as np
import miditoolkit
import copy
import pickle

from midi_utils import *
def save_token(tok, save_path):
    with open(save_path, 'w+') as f:
        f.write(",".join(map(str, tok)))
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


tokenizer = CPWord(config)

files_paths = list(Path("./Pop1K7/midi_analyzed").glob("**/*.mid"))
save_dir = "./datas"
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

event2word, word2event = pickle.load(open('./basic_event_dictionary.pkl', 'rb'))

all_tokens = {}
for path in tqdm(files_paths):
    try:
        filename = Path(path).stem + ".txt"
        tqdm.write(f"{path}")
        note_items, tempo_items = read_items(path)
        note_items = quantize_items(note_items)
        chord_items = chord_extract(path, note_items[-1].end)
        items = tempo_items + note_items

        max_time = note_items[-1].end
        groups = group_items(items, max_time)
        events = item2event(groups)
        
        save_path = os.path.join(save_dir, filename)
        toks = []
        for event in events:
            e = '{}_{}'.format(event.name, event.value)
            if e in event2word:
                toks.append(event2word[e])
            else:
                # OOV
                if event.name == 'Note Velocity':
                    # replace with max velocity based on our training data
                    toks.append(event2word['Note Velocity_21'])
        save_token(toks, save_path)
        tqdm.write(','.join(map(str, toks[:10])))
        all_tokens[str(path)] = toks
    except Exception as e:
        tqdm.write(f"Skipping {path}: {e}")
