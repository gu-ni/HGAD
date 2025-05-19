# %%
import os
import json
import numpy as np

# base

score_dir = "/workspace/MegaInspection/HGAD/scores"
scenarios = os.listdir(score_dir)
scenario_base_scores = {}

for scenario in scenarios:
    scenario_base_dir = os.path.join(score_dir, scenario, "base")
    if not os.path.isdir(scenario_base_dir):
        continue
    print(f"Processing {scenario} base")
    file = os.listdir(scenario_base_dir)[0]
    with open(os.path.join(scenario_base_dir, file), "r") as f:
        data = json.load(f)
    print(len(data))
    # Calculate final scores
    auroc_list = []
    ap_list = []
    for key, value in data.items():
        auroc_list.append(value["image_auroc"])
        ap_list.append(value["pixel_ap"])
    auroc_value = sum(auroc_list) / len(auroc_list)
    ap_value = sum(ap_list) / len(ap_list)
    auroc_value = np.round(auroc_value * 100, 1)
    ap_value = np.round(ap_value * 100, 1)
    
    scenario_base_scores[scenario] = {
        "image_auroc": np.round(auroc_value, 1),
        "pixel_ap": np.round(ap_value, 1),
    }
scenario_base_scores = dict(sorted(scenario_base_scores.items()))
scenario_base_scores
# %%
# continual

import os
import json
import numpy as np

score_dir = "/workspace/MegaInspection/HGAD/scores"
scenarios = sorted(os.listdir(score_dir))

scenario_continual_scores_fm = {}
scenario_continual_scores_acc = {}

for scenario in scenarios:
    cases = os.listdir(os.path.join(score_dir, scenario))
    cases = [case for case in cases if "base" not in case]
    cases = sorted(cases, key=lambda x: int(x[:-13]))

    fm_scores_of_dataset_per_case = {}
    acc_scores_of_dataset_per_case = {}

    for case in cases:
        case_dir = os.path.join(score_dir, scenario, case)
        datasets = sorted(os.listdir(case_dir), key=lambda x: int(x[7:]))

        for dataset in datasets:
            model_scores = {}
            dataset_dir = os.path.join(case_dir, dataset)
            models = sorted(os.listdir(dataset_dir), key=lambda x: int(x.split(".")[0][5:]))

            for model in models:
                with open(os.path.join(dataset_dir, model), "r") as f:
                    data = json.load(f)

                auroc_list = [v["image_auroc"] for v in data.values()]
                ap_list = [v["pixel_ap"] for v in data.values()]

                avg_auroc = sum(auroc_list) / len(auroc_list)
                avg_ap = sum(ap_list) / len(ap_list)

                model_scores[model] = {
                    "image_auroc": avg_auroc,
                    "pixel_ap": avg_ap
                }

            # 모델명 순으로 정렬
            sorted_models = sorted(model_scores.keys(), key=lambda x: int(x.split(".")[0][5:]))

            # 중간 모델 최고 성능
            max_auroc = max(model_scores[m]["image_auroc"] for m in sorted_models)
            max_ap = max(model_scores[m]["pixel_ap"] for m in sorted_models)

            # 마지막 모델 성능
            last_model = sorted_models[-1]
            last_auroc = model_scores[last_model]["image_auroc"]
            last_ap = model_scores[last_model]["pixel_ap"]

            # FM 계산
            fm_scores = {
                "auroc_fm": max_auroc - last_auroc,
                "ap_fm": max_ap - last_ap
            }
            
            acc_scores = {
                "auroc_acc": last_auroc,
                "ap_acc": last_ap
            }

            if case not in fm_scores_of_dataset_per_case:
                fm_scores_of_dataset_per_case[case] = {}
            if case not in acc_scores_of_dataset_per_case:
                acc_scores_of_dataset_per_case[case] = {}

            fm_scores_of_dataset_per_case[case][dataset] = fm_scores
            acc_scores_of_dataset_per_case[case][dataset] = acc_scores

    scenario_continual_scores_fm[scenario] = fm_scores_of_dataset_per_case
    scenario_continual_scores_acc[scenario] = acc_scores_of_dataset_per_case

# %%
avg_fm_scores = {}
avg_acc_scores = {}
for scenario, case_scores in scenario_continual_scores_acc.items():
    avg_acc_scores[scenario] = {}
    for case, dataset_scores in case_scores.items():
        
        auroc_list = []
        ap_list = []
        for dataset, acc_scores in dataset_scores.items():
            auroc_list.append(acc_scores["auroc_acc"])
            ap_list.append(acc_scores["ap_acc"])
        avg_auroc = sum(auroc_list) / len(auroc_list)
        avg_ap = sum(ap_list) / len(ap_list)

        avg_acc_scores[scenario][case] = {
            "avg_auroc_acc": np.round(avg_auroc*100, 1),
            "avg_ap_acc": np.round(avg_ap*100, 1)
        }
for scenario, case_scores in scenario_continual_scores_fm.items():
    avg_fm_scores[scenario] = {}
    for case, dataset_scores in case_scores.items():
        
        auroc_list = []
        ap_list = []
        for dataset, fm_scores in dataset_scores.items():
            auroc_list.append(fm_scores["auroc_fm"])
            ap_list.append(fm_scores["ap_fm"])
        avg_auroc = sum(auroc_list) / len(auroc_list)
        avg_ap = sum(ap_list) / len(ap_list)

        avg_fm_scores[scenario][case] = {
            "avg_auroc_fm": np.round(avg_auroc*100, 1),
            "avg_ap_fm": np.round(avg_ap*100, 1)
        }
avg_acc_scores

# %%
