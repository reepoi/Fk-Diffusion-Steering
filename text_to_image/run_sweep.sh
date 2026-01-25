#!/bin/bash
source ~/.bashrc
job_slot_to_gpu=(-1 1 2 4 5 6 7)

# Preflight
# env_parallel --eta -j 1 --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[{%}]}' 'python launch_eval_runs.py -c job rng_seed={rng_seed} prompt={prompt_id}' ::: $(duckdb runs.sqlite -c "copy (select * from SweepRngSeed cross join (select prompt_id from Prompt)) to '/dev/stdout'")
# Go
# env_parallel --delay 5 --eta -j $((${#job_slot_to_gpu[@]} - 1)) --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[{%}]}' 'python launch_eval_runs.py rng_seed={rng_seed} num_particles=4 num_inference_steps=999 resample_t_end=980 prompt={prompt_id}' ::: $(duckdb runs.sqlite -c "copy (select * from SweepRngSeed cross join (select prompt_id from Prompt)) to '/dev/stdout'")
env_parallel --delay 5 --eta -j $((${#job_slot_to_gpu[@]} - 1)) --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[{%}]}' 'python launch_eval_runs.py rng_seed={rng_seed} num_particles={num_particles} num_inference_steps={num_inference_steps} resample_t_start={resample_t_start} resample_frequency={resample_frequency} resample_t_end={resample_t_end} prompt={prompt_id}' ::: $(duckdb runs.sqlite -c "copy (select *, num_inference_steps // 5 as resample_frequency, resample_frequency as resample_t_start, num_inference_steps - resample_frequency as resample_t_end from SweepRngSeed cross join SweepNumParticles cross join SweepPromptSubset cross join SweepNumInferenceSteps where num_particles >= 64 order by num_inference_steps, num_particles) to '/dev/stdout'")
