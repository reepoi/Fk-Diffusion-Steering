#!/bin/bash
source ~/.bashrc
job_slot_to_gpu=(-1 1 2 3 4 5 6)

# Preflight
# env_parallel --eta -j 1 --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[{%}]}' 'python launch_eval_runs.py -c job rng_seed={rng_seed} prompt={prompt_id}' ::: $(duckdb runs.sqlite -c "copy (select * from SweepRngSeed cross join (select prompt_id from Prompt)) to '/dev/stdout'")
# Go
env_parallel --delay 5 --eta -j $((${#job_slot_to_gpu[@]} - 1)) --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[{%}]}' 'python launch_eval_runs.py rng_seed={rng_seed} prompt={prompt_id}' ::: $(duckdb runs.sqlite -c "copy (select * from SweepRngSeed cross join (select prompt_id from Prompt)) to '/dev/stdout'")
