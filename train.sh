CHECKPOINTS=checkpoints/envi
CUDA_VISIBLE_DEVICES=0 python train.py data-bin/data.tokenized.en-vi \
  --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
  --arch transformer --save-dir checkpoints/envi \
  --optimizer sgd \
  --beam_size 6 --max_order 5 --gram 2 --mle_weight 0.3 --rl_weight 0.7 --modgleu True \
  --max-epoch 10 --criterion multinomial_rl --batch-size 128