CHECKPOINTS=drive/MyDrive/thesis/checkpoints/envi
CUDA_VISIBLE_DEVICES=0 python fairseq_rl/train.py drive/MyDrive/thesis/data-bin/data.tokenized.en-vi \
  -s en -t vi --fp32-reduce-scatter \
  --lr 0.00001 --clip-norm 0.1 --dropout 0.1 --max-tokens 10000 \
  --arch transformer --save-dir $CHECKPOINTS \
  --optimizer sgd \
  --multinomial_sample_train True \
  --beam_size 5 --max_order 5 --mle_weight 0.1 --gram 4 \
  --max-epoch 10 --criterion multinomial_rl --batch-size 21 --sampling_topk 4 \
  --save-interval-updates 1000