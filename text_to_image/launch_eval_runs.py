# primary generation script
import csv
import pprint
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt

import argparse

from datetime import datetime

import torch
from diffusers import DDIMScheduler, UNet2DConditionModel

import sys
sys.path.append("fkd_diffusers")

from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL
from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion

from fks_utils import do_eval

import hydra
from omegaconf import OmegaConf
import conf


log = conf.getLoggerByFilename(__file__)


# load prompt data
def load_geneval_metadata(prompt_path, max_prompts=None):
    if prompt_path.endswith(".json"):
        with open(prompt_path, "r") as f:
            data = json.load(f)
    else:
        assert prompt_path.endswith(".jsonl")
        with open(prompt_path, "r") as f:
            data = [json.loads(line) for line in f]
    assert isinstance(data, list)
    prompt_key = "prompt"
    if prompt_key not in data[0]:
        assert "text" in data[0], "Prompt data should have 'prompt' or 'text' key"

        for item in data:
            item["prompt"] = item["text"]
    if max_prompts is not None:
        data = data[:max_prompts]
    return data


@hydra.main(**conf.HYDRA_INIT)
def main(cfg):
    with conf.Session() as db:
        cfg = conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        db.commit()
        log.info('Command: python %s', ' '.join(sys.argv[:-1]))
        log.info(pprint.pformat(cfg))
        log.info('Output directory: %s', cfg.run_dir)


    # seed everything
    torch.manual_seed(cfg.rng_seed)
    torch.cuda.manual_seed(cfg.rng_seed)
    torch.cuda.manual_seed_all(cfg.rng_seed)

    # configure pipeline
    if "xl" in cfg.model_name and "dpo" not in cfg.model_name:
        print("Using SDXL")
        pipe = FKDStableDiffusionXL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        )
    elif "mhdang/dpo" in cfg.model_name and "xl" in cfg.model_name:
        pipe = FKDStableDiffusionXL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )

        unet_id = "mhdang/dpo-sdxl-text2image-v1"
        unet = UNet2DConditionModel.from_pretrained(
            unet_id, subfolder="unet", torch_dtype=torch.float16
        )
        pipe.unet = unet

    elif "mhdang/dpo" in cfg.model_name and "xl" not in cfg.model_name:
        pipe = FKDStableDiffusion.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        # load finetuned model
        unet_id = "mhdang/dpo-sd1.5-text2image-v1"
        unet = UNet2DConditionModel.from_pretrained(
            unet_id, subfolder="unet", torch_dtype=torch.float16
        )
        pipe.unet = unet
    else:
        print("Using SD")
        pipe = FKDStableDiffusion.from_pretrained(
            cfg.model_name, torch_dtype=torch.float16
        )

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)


    metrics_to_compute = ["ImageReward", "HumanPreference"]

    # cache metric fns
    # do_eval(
    #     prompt=["test"],
    #     images=[Image.new("RGB", (224, 224))],
    #     metrics_to_compute=metrics_to_compute,
    # )

    metrics_arr = {
        metric: dict(mean=0, max=0, min=0, std=0) for metric in metrics_to_compute
    }
    n_samples = 0
    average_time = 0

    prompt = [cfg.prompt.text] * cfg.num_particles
    start_time = datetime.now()

    prompt_path = cfg.run_dir

    fkd_args = dict(
        lmbda=cfg.lambduh,
        num_particles=cfg.num_particles,
        use_smc=cfg.use_smc,
        adaptive_resampling=cfg.adaptive_resampling,
        resample_frequency=cfg.resample_frequency,
        time_steps=cfg.num_inference_steps,
        resampling_t_start=cfg.resample_t_start,
        resampling_t_end=cfg.resample_t_end,
        guidance_reward_fn=cfg.guidance_reward_fn,
        potential_type=cfg.potential,
    )

    images = pipe(
        prompt,
        num_inference_steps=cfg.num_inference_steps,
        eta=cfg.eta,
        fkd_args=fkd_args,
    )
    images = images[0]
    if cfg.use_smc:
        end_time = datetime.now()

    results = do_eval(
        prompt=prompt, images=images, metrics_to_compute=metrics_to_compute
    )
    if not cfg.use_smc:
        end_time = datetime.now()
    time_taken = end_time - start_time

    results["time_taken"] = time_taken.total_seconds()
    results["prompt"] = prompt

    n_samples += 1

    average_time += time_taken.total_seconds()
    print(f"Time taken: {average_time / n_samples}")

    # sort images by reward
    guidance_reward = np.array(results[fkd_args['guidance_reward_fn']]["result"])
    sorted_idx = np.argsort(guidance_reward)[::-1]
    images = [images[i] for i in sorted_idx]
    for metric in metrics_to_compute:
        results[metric]["result"] = [
            results[metric]["result"][i] for i in sorted_idx
        ]

    # {'ImageReward': {'result': [0.8297791481018066, 0.8297791481018066, 0.8297791481018066, 0.5484601259231567], 'mean': 0.7594493627548218, 'std': 0.14065951108932495, 'max': 0.8297791481018066, 'min': 0.5484601259231567}, 'HumanPreference': {'result': [0.269775390625, 0.269775390625, 0.269775390625, 0.266845703125], 'mean': 0.26904296875, 'std': 0.00146484375, 'max': 0.269775390625, 'min': 0.266845703125}, 'time_taken': 24.903365, 'prompt': ['an underwater rollercoaster, cinematic, dramatic, -', 'an underwater rollercoaster, cinematic, dramatic, -', 'an underwater rollercoaster, cinematic, dramatic, -', 'an underwater rollercoaster, cinematic, dramatic, -']}
    with open(cfg.run_dir/'metrics_best_sample.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, ['prompt_id', *metrics_to_compute])
        writer.writeheader()
        to_write = {
            'prompt_id': cfg.prompt.prompt_id,
            **{
                metric: results[metric]['result'][0] for metric in metrics_to_compute
            }
        }
        writer.writerow(to_write)


    for metric in metrics_to_compute:
        metrics_arr[metric]["mean"] += results[metric]["mean"]
        metrics_arr[metric]["max"] += results[metric]["max"]
        metrics_arr[metric]["min"] += results[metric]["min"]
        metrics_arr[metric]["std"] += results[metric]["std"]

    for metric in metrics_to_compute:
        print(
            metric,
            metrics_arr[metric]["mean"] / n_samples,
            metrics_arr[metric]["max"] / n_samples,
        )

    if True:
    # if args.save_individual_images:
        sample_path = os.path.join(prompt_path, "samples")
        os.makedirs(sample_path, exist_ok=True)
        for image_idx, image in enumerate(images):
            image.save(os.path.join(sample_path, f"{image_idx:05}.png"))

        best_of_n_sample_path = os.path.join(prompt_path, "best_of_n_samples")
        os.makedirs(best_of_n_sample_path, exist_ok=True)
        for image_idx, image in enumerate(images[:1]):
            image.save(os.path.join(best_of_n_sample_path, f"{image_idx:05}.png"))

    with open(os.path.join(prompt_path, "results.json"), "w") as f:
        json.dump(results, f)

    _, ax = plt.subplots(1, cfg.num_particles, figsize=(cfg.num_particles * 5, 5))
    for i, image in enumerate(images):
        ax[i].imshow(image)
        ax[i].axis("off")

    plt.suptitle(prompt[0])
    image_fpath = os.path.join(prompt_path, f"grid.png")
    plt.savefig(image_fpath)
    plt.close()

    # save final metrics
    for metric in metrics_to_compute:
        metrics_arr[metric]["mean"] /= n_samples
        metrics_arr[metric]["max"] /= n_samples
        metrics_arr[metric]["min"] /= n_samples
        metrics_arr[metric]["std"] /= n_samples

    with open(cfg.run_dir/'final_metrics.json', "w") as f:
        json.dump(metrics_arr, f)


if __name__ == "__main__":
    last_override, run_dir = conf.get_run_dir()
    conf.set_run_dir(last_override, run_dir)
    main()
