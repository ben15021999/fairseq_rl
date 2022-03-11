CHECKPOINTS=checkpoints/envi
CUDA_VISIBLE_DEVICES=0 python train.py data-bin/data.tokenized.en-vi \
  -s en -t vi --fp32-reduce-scatter \
  --lr 0.0001 --share-decoder-input-output-embed --clip-norm 0.1 --dropout 0.1 --max-tokens 50464 \
  --arch transformer --save-dir $CHECKPOINTS \
  --optimizer adam \
  --batch-size 8 --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --max-epoch 10 --sentence-avg