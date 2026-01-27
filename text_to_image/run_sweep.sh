#!/bin/bash
source ~/.bashrc
job_slot_to_gpu=(-1 2 3 4 5)

# Preflight
# env_parallel --eta -j 1 --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[{%}]}' 'python launch_eval_runs.py -c job rng_seed={rng_seed} prompt={prompt_id}' ::: $(duckdb runs.sqlite -c "copy (select * from SweepRngSeed cross join (select prompt_id from Prompt)) to '/dev/stdout'")
# Go
# env_parallel --delay 3 --eta -j $((${#job_slot_to_gpu[@]} - 1)) --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[{%}]}' 'python launch_eval_runs.py rng_seed={rng_seed} num_particles=4 num_inference_steps=100 prompt={prompt_id}' ::: $(duckdb runs.sqlite -c "copy (select * from (select unnest([42, 43, 44]) as rng_seed) cross join (select prompt_id from Prompt)) to '/dev/stdout'")
env_parallel --delay 1 --eta -j $((${#job_slot_to_gpu[@]} - 1)) --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[{%}]}' 'python launch_eval_runs.py rng_seed={rng_seed} num_particles={num_particles} num_inference_steps={num_inference_steps} resample_t_start={resample_t_start} resample_frequency={resample_frequency} resample_t_end={resample_t_end} adaptive_resampling=true prompt={prompt_id}' ::: $(duckdb runs.sqlite -c "copy (select *, 5 as num_particles, 999 as num_inference_steps, num_inference_steps // 5 + 1 as resample_frequency, resample_frequency as resample_t_start, num_inference_steps - resample_frequency + 1 as resample_t_end from SweepRngSeed cross join (select prompt_id from Prompt)) to '/dev/stdout'")
# env_parallel --eta -j $((${#job_slot_to_gpu[@]} - 1)) --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[{%}]}' 'python clip_test.py {prompt_id} {image_path}' ::: $(duckdb runs.sqlite -c "copy (select prompt_id, format('/home/ttransue/GitHub/Fk-Diffusion-Steering/text_to_image/outputs/runs/{}/samples', alt_id) as image_path from Conf join Prompt on Conf.Prompt = Prompt.id where num_particles = 4 and num_inference_steps = 100) to '/dev/stdout'")
