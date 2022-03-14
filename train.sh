CHECKPOINTS=checkpoints/envi_3
CUDA_LAUNCH_BLOCKING=1 python train.py data-bin/data.tokenized.en-vi\
  --lr 0.0001 --clip-norm 0.1 --dropout 0.1 --max-tokens 50464 \
  --arch transformer --save-dir $CHECKPOINTS \
  --optimizer adam --beam_size 5 --max_order 4 \
  --max-epoch 10 --criterion reward_baseline --batch-size 16 \
  --finetune-from-model checkpoints/envi/checkpoint_best.pt