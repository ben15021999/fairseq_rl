TEXT=data/origin
python fairseq_cli/preprocess.py --source-lang en --target-lang vi \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/data.tokenized.en-vi.v2