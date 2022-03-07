CUDA_VISIBLE_DEVICES=0 python fairseq_cli/generate.py \
    data-bin/data.tokenized.en-vi \
    --path checkpoints/envi/checkpoint_best.pt \
    --batch-size 32 \
    --beam 4 \
    --results-path evaluation/envi --sacrebleu --cpu