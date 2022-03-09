CHECKPOINTS=checkpoints/envi_3
CUDA_LAUNCH_BLOCKING=1 python train.py data-bin/data.tokenized.en-vi.v6 \
  -s en -t vi --fp32-reduce-scatter \
  --lr 0.0001 --clip-norm 0.1 --dropout 0.1 --max-tokens 65480 \
  --arch transformer --save-dir $CHECKPOINTS \
  --optimizer adam --reset-optimizer \
  --batch-size 4 \
  --multinomial_sample_train True \
  --beam_size 5 --max_order 4 --mle_weight 0.1 --gram 3 \
  --max-epoch 20 --criterion multinomial_rl --sampling_topk 4 \
  --save-interval-updates 1000