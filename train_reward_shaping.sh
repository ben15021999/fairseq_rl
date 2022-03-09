CHECKPOINTS=checkpoints/envi_2
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py data-bin/data.tokenized.en-vi.v6 \
  -s en -t vi \
  --lr 0.0001 --clip-norm 0.1 --dropout 0.1 --max-tokens 65480 \
  --arch transformer --save-dir $CHECKPOINTS \
  --optimizer adam --sampling_topk 5 --beam_size 5 --max_order 4 \
  --max-epoch 10 --criterion reward_sharping --batch-size 4 \
  --save-interval-updates 1000