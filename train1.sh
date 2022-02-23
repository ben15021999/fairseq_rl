CHECKPOINTS=checkpoints/envi_1
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py data-bin/data.tokenized.en-vi \
  -s en -t vi \
  --lr 0.001 --clip-norm 0.1 --dropout 0.1 --max-tokens 5000 \
  --arch transformer --save-dir $CHECKPOINTS \
  --optimizer adam \
  --max-epoch 10 --criterion reward_baseline --batch-size 5 --sentence-avg \
  --save-interval-updates 1000