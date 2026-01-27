import csv
from pathlib import Path
import sys

import torch
import clip
from PIL import Image
import duckdb



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
    model.eval()

    k = 4

    duckdb.sql("""
        attach 'runs.sqlite';
        use runs;
    """)
    prompts_and_dirs = duckdb.sql("""
        select prompt_id, format('/home/ttransue/GitHub/Fk-Diffusion-Steering/text_to_image/outputs/runs/{}/samples', alt_id) as image_path from Conf join Prompt on Conf.Prompt = Prompt.id where adaptive_resampling and num_particles <= 32 and num_inference_steps = 50
    """).fetchall()

    for prompt_id, image_path in prompts_and_dirs:
        image_path = Path(image_path)
        image_features = []

        with torch.no_grad():
            for path in image_path.iterdir():
                image = preprocess(Image.open(path).convert("RGB"))
                image = image.unsqueeze(0).to(device)

                feat = model.encode_image(image)
                feat = feat / feat.norm(dim=1, keepdim=True)  # L2 normalize (required for clip according to what i read)

                image_features.append(feat)

        image_features = torch.cat(image_features, dim=0)


        # set it to float so that it doesn't complain
        image_features = image_features.float()

        # get pairwise distances between features
        dist_sq = torch.cdist(image_features, image_features, p=2) ** 2

        # sum over i < j
        clip_div = dist_sq.triu(diagonal=1).sum()
        # normalize over batch size (see equation 9 in the paper )
        clip_div = (2.0 / (k* (k - 1))) * clip_div

        with open(image_path.parent/'clip_best_sample.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, ['prompt_id', 'clip_div'])
            writer.writeheader()
            to_write = {
                'prompt_id': prompt_id,
                'clip_div': clip_div.item(),
            }
            writer.writerow(to_write)

        print(to_write)
