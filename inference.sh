#!/bin/bash

python3 inference.py  \
    --base_model /project/pi_miyyer_umass_edu/ctpham/.cache/LongAlpaca-7B \
    --cache_dir /project/pi_miyyer_umass_edu/ctpham/.cache/ \
    --context_size 32768 \
    --max_gen_len 100 \
    --flash_attn True \
    --material "/work/pi_miyyer_umass_edu/ctpham/alpaca-long-dev/data/sample_20k/1.txt" \
    --temperature 0.0 \
    --top_p 0.0