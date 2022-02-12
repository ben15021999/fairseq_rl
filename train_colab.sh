CHECKPOINTS=drive/MyDrive/thesis/checkpoints/envi
CUDA_VISIBLE_DEVICES="" python fairseq_rl/train.py data-bin/data.tokenized.en-vi \
  -s en -t vi \
  --lr 0.01 --clip-norm 0.1 --dropout 0.2 --max-tokens 1024 \
  --arch transformer --save-dir $CHECKPOINTS \
  --optimizer adam \
  --multinomial_sample_train True \
  --beam_size 4 --max_order 5 --mle_weight 0.3 --rl_weight 0.7 --modgleu True \
  --max-epoch 10 --criterion multinomial_rl --batch-size 16