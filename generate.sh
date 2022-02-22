CUDA_VISIBLE_DEVICES=0 python fairseq_cli/generate.py \
    data-bin/data.tokenized.en-vi \
    --path checkpoints/envi/checkpoint_best.pt \
    --batch-size 22 \
    --beam 5 \
    --results-path evaluation/envi --cpu