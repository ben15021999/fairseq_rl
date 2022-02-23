CHECKPOINTS=checkpoints/envi
CUDA_LAUNCH_BLOCKING=1 python train.py data-bin/data.tokenized.en-vi \
  -s en -t vi --fp32-reduce-scatter \
  --lr 0.00001 --clip-norm 0.1 --dropout 0.1 --max-tokens 5000 \
  --arch transformer --save-dir checkpoints/envi \
  --optimizer adam \
  --multinomial_sample_train True \
  --beam_size 4 --max_order 5 --mle_weight 0.1 --gram 3 \
  --max-epoch 10 --criterion multinomial_rl --batch-size 4 --sampling_topk 4 \
  --save-interval-updates 1000 --cpu