# %%
import os
import json
import numpy as np

# base

zs_list = [
    "meta_mvtec",
    "meta_visa"
]
zs_list = [
    "meta_continual_ad_test_total"
]

score_dir = "/workspace/MegaInspection/HGAD/scores"
scenarios = os.listdir(score_dir)
scenarios = sorted(scenarios)[2:]

zs_scores = {}

zs_scores = {}

for scenario in scenarios:
    zs_scores[scenario] = {}
    for zs in zs_list:
        zs_scores[scenario][zs] = {}

        cases = os.listdir(os.path.join(score_dir, scenario))
        cases = [case for case in cases if "base" not in case]
        cases = sorted(cases, key=lambda x: int(x[:-13]))

        for case in cases:
            zs_dir = os.path.join(score_dir, scenario, case, zs)

            file_list = os.listdir(zs_dir)
            if not file_list:
                print(f"[WARNING] No files in {zs_dir}")
                continue

            file = file_list[0]
            with open(os.path.join(zs_dir, file), "r") as f:
                data = json.load(f)

            auroc_list = []
            ap_list = []

            for key, value in data.items():
                auroc_list.append(value["image_auroc"])
                ap_list.append(value["pixel_ap"])

            auroc_value = sum(auroc_list) / len(auroc_list)
            ap_value = sum(ap_list) / len(ap_list)

            zs_scores[scenario][zs][case] = {
                "image_auroc": auroc_value,
                "pixel_ap": ap_value,
            }
# %%
average_scores = {}

for scenario, zs_dict in zs_scores.items():
    average_scores[scenario] = {}
    for zs, cases_dict in zs_dict.items():
        auroc_list = []
        ap_list = []

        for case, score in cases_dict.items():
            auroc_list.append(score["image_auroc"])
            ap_list.append(score["pixel_ap"])

        avg_auroc = sum(auroc_list) / len(auroc_list) if auroc_list else 0.0
        avg_ap = sum(ap_list) / len(ap_list) if ap_list else 0.0

        average_scores[scenario][zs] = {
            "avg_image_auroc": np.round(avg_auroc * 100, 1),
            "avg_pixel_ap": np.round(avg_ap * 100, 1)
        }

average_scores
# %%
