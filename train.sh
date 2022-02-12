CHECKPOINTS=checkpoints/envi
CUDA_LAUNCH_BLOCKING=1 python train.py data-bin/data.tokenized.en-vi \
  -s en -t vi \
  --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 2048 \
  --arch transformer --save-dir checkpoints/envi \
  --optimizer adam \
  --multinomial_sample_train True \
  --beam_size 4 --max_order 5 --mle_weight 0.3 --rl_weight 0.7 --modgleu True \
  --max-epoch 10 --criterion multinomial_rl --batch-size 8