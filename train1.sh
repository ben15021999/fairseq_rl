CHECKPOINTS=checkpoints/envi
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py data-bin/data.tokenized.en-vi \
  -s en -t vi --fp32-reduce-scatter \
  --lr 0.001 --clip-norm 0.1 --dropout 0.2 --max-tokens 5000 \
  --arch transformer --save-dir checkpoints/envi \
  --optimizer sgd \
  --max-epoch 10 --criterion reward_baseline --batch-size 4 --sentence-avg \
  --save-interval-updates 500