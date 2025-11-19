# Transformer-XL: 32-Bar Piano Generation

Use the provided scripts to train Transformer-XL on `datas/*.json` event tokens and generate exactly 32 bars of music.

## Dataset Assumptions

- `datas/*.json` files come from MidiTok (REMI) export and contain either a single object with a `tokens` array or a list of such objects.
- Each `tokens` array is an event sequence (e.g., `Bar_None`, `Position_0`, `Tempo_120.0`, `Pitch_60`, `Velocity_64`, `Duration_1.0.0`, ...).

## Train

```bash
python task1/train_txl.py \
  --data_dir datas \
  --work_dir task1/runs/txl_music \
  --cuda \
  --batch_size 8 \
  --tgt_len 256 \
  --mem_len 256 \
  --epochs 10
```

Artifacts: `model.pt`, `vocab.tsv` under `task1/runs/txl_music/`.

## Generate 32 Bars

```bash
python task1/generate_txl.py \
  --work_dir task1/runs/txl_music \
  --cuda \
  --temperature 1.0 \
  --top_k 50 \
  --seed_tokens Bar_None Position_0
```

Output tokens are saved to `task1/runs/txl_music/samples/sample_<timestamp>.txt`.

## Notes

- Vocab is built from train+valid and saved as `vocab.tsv`.
- Generation counts `Bar_None` as bar starts and stops right before the 33rd one, yielding 32 bars.
- Adjust `--n_layer`, `--d_model`, `--tgt_len`, and `--mem_len` for available VRAM.
- To convert tokens to MIDI, plug your MidiTok decoding after generation.
