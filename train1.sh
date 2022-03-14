CHECKPOINTS=checkpoints/envi_2
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py data-bin/data.tokenized.en-vi \
  -s en -t vi \
  --lr 0.0001 --clip-norm 0.1 --dropout 0.1 --max-tokens 5000 \
  --arch transformer --save-dir $CHECKPOINTS \
  --optimizer adam --max_order 4 \
  --max-epoch 10 --criterion reward_baseline_v2 --batch-size 16 \
  --save-interval-updates 1000 \
  --finetune-from-model checkpoints/envi/checkpoint_best.pt --cpu