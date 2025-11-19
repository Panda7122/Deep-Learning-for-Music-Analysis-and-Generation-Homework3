# Deep-Learning-for-Music-Analysis-and-Generation-Homework3

## 41147009S 陳炫佑

## task 1

require model:

`https://github.com/Natooz/MidiTok.git `

`https://github.com/kimiyoung/transformer-xl.git`

you should run in python

training:3.9.19 and set up envirment of MidiTok, transformer-xl, and thie project's requirement

```shell
python task1/train_txl.py \
  --data_dir datas \
  --work_dir task1/runs/txl_music \
  --cuda \
  --batch_size 8 \
  --tgt_len 256 \
  --mem_len 256 \
  --epochs 10 \
  --lr 1e-4 \
  --warmup_steps 1000 \
  --cosine_anneal
```

generate:

```shell
python3 task1/generate_txl.py --work_dir task1/runs/txl_music --cuda
```
