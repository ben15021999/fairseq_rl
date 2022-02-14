CHECKPOINTS=checkpoints/envi
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py data-bin/data.tokenized.en-vi \
  -s en -t vi \
  --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 2048 \
  --arch transformer --save-dir checkpoints/envi \
  --optimizer adam \
  --max-epoch 10 --criterion reward_baseline --batch-size 4