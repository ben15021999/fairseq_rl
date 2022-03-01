CHECKPOINTS=drive/MyDrive/thesis/checkpoints/envi_v2
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py drive/MyDrive/thesis/data-bin/data.tokenized.en-vi.v1 \
  -s en -t vi \
  --lr 0.00001 --clip-norm 0.1 --dropout 0.1 --max-tokens 5000 \
  --arch transformer --save-dir $CHECKPOINTS \
  --optimizer adam --sampling_topk 5 --beam_size 5 --max_order 4 \
  --max-epoch 10 --criterion reward_baseline --batch-size 4 --sentence-avg \
  --save-interval-updates 1000