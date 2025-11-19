import torch
import glob
from tqdm import tqdm
from torch import nn
from torch.utils.data.dataloader import DataLoader, Dataset
import numpy as np
import pickle
import task1.midi_utils as midi_utils
import os
import argparse
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
_BASE_DIR = os.path.dirname(__file__)
_TXL_DIR = os.path.join(_BASE_DIR, '..', 'transformer-xl', 'pytorch')
_TXL_UTILS_DIR = os.path.join(_TXL_DIR, 'utils')

## set the input length. must be same with the model config
X_LEN = 1024

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, help='gpu device.', default='cuda')
    parser.add_argument('--ckp_path', type=str, help='checkpoint save path.', default='')
    parser.add_argument('--output_folder', type=str, help='midi save folder.', default='')
    ####################################################
    # you can define your arguments here. there is a example below.
    # parser.add_argument('--device', type=str, help='gpu device.', default='cuda')
    ####################################################
    parser.add_argument('--dict_path', type=str, help='the dictionary path.', default='./dictionary.pkl')
    args = parser.parse_args()
    return args

opt = parse_opt()
event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Transformer configuration
        vocab_size = len(event2word)
        d_model = 512
        nhead = 8
        num_layers = 12
        dim_feedforward = 2048
        dropout = 0.1

        # embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(X_LEN, d_model)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False  # we'll feed (seq, batch, embed)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # output projection
        self.fc_out = nn.Linear(d_model, vocab_size)

        # simple weight init
        nn.init.normal_(self.tok_emb.weight, mean=0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0, std=0.02)
        nn.init.xavier_uniform_(self.fc_out.weight)

    # monkey-patch a working forward into the class so we don't need to modify the later stub
    def forward(self, x):
        """
        x: LongTensor of shape (batch, seq_len)
        returns logits: FloatTensor of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = x.size()
        device = x.device

        # positions
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

        # embeddings
        emb = self.tok_emb(x) + self.pos_emb(pos)  # (batch, seq_len, d_model)

        # transformer expects (seq_len, batch, embed)
        emb = emb.permute(1, 0, 2)

        # pass through encoder
        out = self.transformer(emb)  # (seq_len, batch, d_model)

        # back to (batch, seq_len, d_model)
        out = out.permute(1, 0, 2)

        # project to vocab
        logits = self.fc_out(out)  # (batch, seq_len, vocab_size)
        return logits

def temperature_sampling(logits, temperature, topk):
    #################################################
    # 1. adjust softmax with the temperature parameter
    # 2. choose top-k highest probs
    # 3. normalize the topk highest probs
    # 4. random choose one from the top-k highest probs as result by the probs after normalize
    #################################################
    # logits: 1D numpy array
    if temperature is None or temperature <= 0:
        temperature = 1e-8
    # temperature scaling
    scaled = logits / float(temperature)

    # ensure topk is valid
    topk = max(1, min(topk, scaled.shape[0]))

    # get topk indices (unsorted), then sort them by logit descending
    topk_idx = np.argpartition(scaled, -topk)[-topk:]
    topk_idx = topk_idx[np.argsort(scaled[topk_idx])[::-1]]

    topk_logits = scaled[topk_idx]

    # stable softmax on topk logits
    exp_logits = np.exp(topk_logits - np.max(topk_logits))
    probs = exp_logits / np.sum(exp_logits)

    # sample one index from topk according to probs
    chosen = np.random.choice(len(topk_idx), p=probs)
    return int(topk_idx[chosen])
    
def test(n_target_bar = 32, temperature = 1.2, topk = 5, output_filename = '', prompt = False):
    # check path folder
    try:
        os.makedirs(opt.output_folder, exist_ok=True)
        tqdm.write(f"dir '{opt.output_folder}' is created")
    except:
        pass
    output_filename = os.path.join(opt.output_folder, output_filename)
    event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))
    with torch.no_grad():
        # load model
        checkpoint = torch.load(opt.ckp_path, weights_only=False)
        model = Model().to(opt.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        batch_size = 1

        if prompt:  
            # If prompt, load prompt file, extract events, create tokens. (similar to dataset preparation)
            pass
        else:  
            # Or, random select prompt to start
            words = []
            for _ in range(batch_size):
                ws = [event2word['Bar_None']]
                if 'chord' in opt.dict_path:
                    tempo_classes = [v for k, v in event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in event2word.items() if 'Tempo Value' in k]
                    chords = [v for k, v in event2word.items() if 'Chord' in k]
                    ws.append(event2word['Position_1/16'])
                    ws.append(np.random.choice(chords))
                    ws.append(event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                else:
                    tempo_classes = [v for k, v in event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in event2word.items() if 'Tempo Value' in k]
                    ws.append(event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                words.append(ws)

        # generate
        original_length = len(words[0])
        initial_flag = 1
        current_generated_bar = 0
        tqdm.write('Start generating')
        bar_notes = 10
        num_note = 0
        pbar = tqdm(total=n_target_bar, position=1)
        while current_generated_bar < n_target_bar:
            # input
            if initial_flag:
                temp_x = np.zeros((batch_size, original_length))
                for b in range(batch_size):
                    for z, t in enumerate(words[b]):
                        temp_x[b][z] = t
                initial_flag = 0
            else:
                temp_x_new = np.zeros((batch_size, 1))
                for b in range(batch_size):
                    temp_x_new[b][0] = words[b][-1]
                temp_x = np.array([np.append(temp_x[0], temp_x_new[0])])
            
            temp_x = torch.Tensor(temp_x).long()

            # Pre-forward diagnostics to avoid silent CUDA asserts from embedding lookups
            vocab_size = len(event2word)
            tx_max = int(temp_x.max().item())
            tx_min = int(temp_x.min().item())
            seq_len = int(temp_x.size(1))
            tqdm.write(f'temp_x len: {temp_x.shape}')

            # warn if model embedding size doesn't match tokenizer vocab
            try:
                emb_cap = int(model.tok_emb.num_embeddings)
            except Exception:
                emb_cap = None
            if emb_cap is not None and emb_cap != vocab_size:
                tqdm.write(f"[WARN] model.tok_emb.num_embeddings={emb_cap} != tokenizer vocab_size={vocab_size}")

            # check token index ranges
            if tx_min < 0 or tx_max >= vocab_size:
                bad = sorted(set(int(x) for x in temp_x.flatten().numpy() if x < 0 or x >= vocab_size))
                tqdm.write('\n[ERROR] Input token index out of range before model forward')
                tqdm.write(f'  token range in input: min={tx_min}, max={tx_max} (allowed 0..{vocab_size-1})')
                tqdm.write(f'  offending token ids (sample): {bad[:50]}')
                try:
                    sample_map = {i: word2event.get(i, '<UNK_EVENT>') for i in bad}
                    tqdm.write('  mapping sample (id -> event):')
                    for i, ev in sample_map.items():
                        tqdm.write(f'    {i} -> {ev}')
                except Exception:
                    pass
                raise ValueError('Input token indices out of range; aborting to avoid CUDA assert')

            # check positional embedding capacity
            try:
                pos_cap = int(model.pos_emb.num_embeddings)
            except Exception:
                pos_cap = None
            if pos_cap is not None and seq_len >= pos_cap:
                # Auto-truncate to the last `pos_cap - 1` tokens (keep room for positions)
                tqdm.write('\n[WARN] Sequence length exceeds positional embedding size; auto-truncating to fit pos_emb capacity')
                tqdm.write(f'  seq_len={seq_len}, pos_emb capacity={pos_cap} (X_LEN={X_LEN})')
                # keep the last pos_cap-1 tokens to be safe
                temp_x = temp_x[:, -pos_cap:]
                seq_len = int(temp_x.size(1))
                tqdm.write(f'  new seq_len={seq_len}')

            # Run forward but catch device-side errors and reproduce on CPU to show Python exception
            try:
                output_logits = model(temp_x.to(opt.device))
            except Exception as e:
                tqdm.write('\n[ERROR] model forward raised an exception (possible device-side CUDA assert):')
                tqdm.write('  Exception:', repr(e))
                tqdm.write('  Attempting CPU reproduction of embedding lookups...')
                try:
                    cpu_model = model.cpu()
                    cpu_x = temp_x.cpu()
                    try:
                        _ = cpu_model.tok_emb(cpu_x)
                        tqdm.write('  token embedding lookup on CPU succeeded.')
                    except Exception as e_tok:
                        tqdm.write('  Token embedding lookup failed on CPU:', repr(e_tok))
                        tx_max = int(cpu_x.max().item())
                        tx_min = int(cpu_x.min().item())
                        bad = sorted(set(int(x) for x in cpu_x.flatten().numpy() if x < 0 or x >= (emb_cap or vocab_size)))
                        tqdm.write(f'    token range: min={tx_min}, max={tx_max}, allowed 0..{(emb_cap or vocab_size)-1}')
                        tqdm.write(f'    offending ids (sample): {bad[:50]}')
                        try:
                            sample_map = {i: word2event.get(i, '<UNK_EVENT>') for i in bad}
                            tqdm.write('    mapping sample (id -> event):')
                            for i, ev in sample_map.items():
                                tqdm.write(f'      {i} -> {ev}')
                        except Exception:
                            pass
                    try:
                        pos = torch.arange(seq_len).unsqueeze(0)
                        _ = cpu_model.pos_emb(pos)
                        tqdm.write('  positional embedding lookup on CPU succeeded.')
                    except Exception as e_pos:
                        tqdm.write('  Positional embedding lookup failed on CPU:', repr(e_pos))
                        tqdm.write(f'    seq_len={seq_len}, pos_emb.num_embeddings={(pos_cap if "pos_cap" in locals() else None)}')
                except Exception as e_cpu:
                    tqdm.write('  CPU reproduction failed:', repr(e_cpu))
                tqdm.write('\nAborting to avoid CUDA device-side assert. Check dictionary vs checkpoint and model config.')
                raise
            # tqdm.write(output_logits)
            # sampling
            _logit = output_logits[0, -1].to('cpu').detach().numpy()
            word = temperature_sampling(
                logits=_logit, 
                temperature=temperature,
                topk=topk)
            tqdm.write(f"{num_note}, {word2event[word]}")
            words[0].append(word)
            # print(worwwd, event2word['Bar_None'])
            if word == event2word['Bar_None']:
                current_generated_bar += 1
                num_note = 0
                pbar.update(1)
            else:
                num_note += 1
        
        midi_utils.write_midi(
            words=words[0],
            word2event=word2event,
            output_path=output_filename,
            prompt_path=None)

def main():
    for i in tqdm(range(20), position=0):
        test(32, 1.2, 60, f'{i}.mid',False)
    return

if __name__ == '__main__':
    main()

