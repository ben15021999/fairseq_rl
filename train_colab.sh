CHECKPOINTS=drive/MyDrive/thesis/checkpoints/envi
CUDA_VISIBLE_DEVICES=0 python fairseq_rl/train.py data-bin/data.tokenized.en-vi \
  -s en -t vi \
  --lr 0.001 --clip-norm 0.1 --dropout 0.1 --max-tokens 5000 \
  --arch transformer --save-dir $CHECKPOINTS \
  --optimizer adam \
  --multinomial_sample_train True \
  --beam_size 4 --max_order 5 --mle_weight 0.3 --modgleu True \
  --max-epoch 10 --criterion multinomial_rl --batch-size 22
  --save-interval-updates 500