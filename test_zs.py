import os
import argparse
import re
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import pandas as pd
import json

import torch
import torch
import torch.nn.functional as F
import gc

from models.model import HGAD
from models.utils import load_model
import models.nf_model as nfs
from datasets.json_loader import prepare_loader_from_json, prepare_loader_from_json_by_chunk
from train import load_class_mapping_length


def parse_args():
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--backbone_arch', default='tf_efficientnet_b6', type=str, 
                        help='feature extractor: (default: efficientnet_b6)')
    parser.add_argument('--flow_arch', default='conditional_flow_model', type=str, 
                        help='normalizing flow model (default: cnflow)')
    parser.add_argument('--feature_levels', default=3, type=int, 
                        help='nudmber of feature layers (default: 3)')
    parser.add_argument('--coupling_layers', default=12, type=int, 
                        help='number of coupling layers used in normalizing flow (default: 8)')
    parser.add_argument('--clamp_alpha', default=1.9, type=float, 
                        help='clamp alpha hyperparameter in normalizing flow (default: 1.9)')
    parser.add_argument('--pos_embed_dim', default=256, type=int,
                        help='dimension of positional enconding (default: 128)')
    parser.add_argument('--lambda1', default=1.0, type=float, 
                        help='hyperparameter lambad_1 in the loss (default: 1.0)')
    parser.add_argument('--lambda2', default=100.0, type=float, 
                        help='hyperparameter lambad_2 in the loss (default: 100.0)')
    parser.add_argument('--label_smoothing', default=0.02, type=float, 
                        help='smoothing the class labels (default: 0.02)')
    parser.add_argument('--n_intra_centers', default=10, type=int, 
                        help='number of intra-class centers (default: 10)')
    
    # Data configures
    parser.add_argument('--img_size', default=336, type=int, 
                        help='image size (default: 1024)')
    parser.add_argument('--msk_size', default=336, type=int, 
                        help='mask size (default: 256)')
    parser.add_argument('--batch_size', default=8, type=int, 
                        help='train batch size (default: 32)')
    parser.add_argument('--num_workers', default=2, type=int,)
    
    # training configures
    parser.add_argument('--lr', type=float, default=2e-4, 
                        help='learning rate (default: 2e-4)')
    parser.add_argument('--lr_decay_epochs', nargs='+', default=[50, 75, 90],
                        help='learning rate decay epochs (default: [50, 75, 90])')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, 
                        help='learning rate decay rate (default: 0.1)')
    parser.add_argument('--lr_warm', type=bool, default=True, 
                        help='learning rate warm up (default: True)')
    parser.add_argument('--lr_warm_epochs', type=int, default=2, 
                        help='learning rate warm up epochs (default: 2)')
    parser.add_argument('--lr_cosine', type=bool, default=True, 
                        help='cosine learning rate schedular (default: True)')
    parser.add_argument('--temp', type=float, default=0.5, 
                        help='temp of cosine learning rate scheduler (default: 0.5)')                    
    parser.add_argument('--meta_epochs', type=int, default=25, 
                        help='number of meta epochs to train (default: 25)')
    parser.add_argument('--sub_epochs', type=int, default=1, 
                        help='number of sub epochs to train (default: 8)')
    
    parser.add_argument('--device', default='cuda', type=str,)
    parser.add_argument('--score_dir', default='/workspace/MegaInspection/HGAD/scores', type=str,)
    parser.add_argument("--json_path", type=str, default="meta_mvtec", help="Name of the task JSON file (e.g., 5classes_tasks)")
    parser.add_argument('--scenario', default='scenario_1', type=str)
    parser.add_argument('--case', default='5_classes_tasks', type=str)

    # Parse arguments
    args = parser.parse_args()
    
    return args


def model_anomaly_score(model, x, y, scale=0):
    """
    Compute pixel-wise and image-level anomaly score from HGAD model.

    Args:
        model: trained HGAD model
        x (Tensor): input image tensor, shape (1, 3, H, W)
        y (Tensor): class index, shape (1,)
        scale (int): feature level to use (default: 0)
    
    Returns:
        anomaly_map (Tensor): shape (H, W)
        image_score (float): max value of anomaly map
    """

    # 1. Extract feature
    features = model.encoder(x)
    feat = features[scale]  # shape (1, C, Hf, Wf)
    B, C, Hf, Wf = feat.shape
    
    # 2. Flatten features and expand label
    feat = feat.permute(0, 2, 3, 1).reshape(-1, C)  # (B*Hf*Wf, C)
    label_r = y.view(-1, 1, 1).repeat(1, Hf, Wf).reshape(-1)  # (B*Hf*Wf,)
    y_onehot = F.one_hot(label_r, num_classes=model.n_classes).float()  # (B*Hf*Wf, num_classes)
    
    # 3. Forward flow
    
    # (bs, 256, h, w)
    pos_embed = nfs.positionalencoding2d(args.pos_embed_dim, Hf, Wf).to(args.device).unsqueeze(0).repeat(B, 1, 1, 1)
    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
    
    z, log_jac_det = model.nfs[scale](feat, [pos_embed, ])
    log_jac_det = log_jac_det.reshape(-1)
    
    # 4. Get intra-class centers
    main_centers = model.mus[scale].detach()  # (n_classes, dim)
    mu_delta = model.mu_deltas[scale].detach()  # (n_classes, n_centers - 1, dim)
    mu_all = torch.cat([main_centers.unsqueeze(1), main_centers.unsqueeze(1) + mu_delta], dim=1)  # (n_classes, n_centers, dim)
    mu_intra = mu_all[label_r]  # (N, n_centers, dim)
    
    # 5. Get weights for intra-class mixture
    phi = model.phi_intras[scale].detach()
    log_py = torch.log_softmax(phi, dim=1)[label_r]  # (N, n_centers)

    # 6. Compute likelihood score (S_l)
    S_l = 1.0 - model.get_logps(z, mu_intra, log_py, log_jac_det, C)  # (N,)

    # 7. Compute entropy (S_e)
    zz = model.calculate_distances_to_inter_class_centers(z, model.mus[scale].detach())  # (N, n_classes)
    probs = torch.softmax(-0.5 * zz, dim=1)
    S_e = torch.sum(-probs * torch.log(probs + 1e-8), dim=1)  # (N,)

    # 8. Final anomaly score
    anomaly_score = S_l * S_e  # (N,)

    # 9. Reshape to spatial map
    anomaly_map = anomaly_score.reshape(B, Hf, Wf)
    anomaly_map = F.interpolate(anomaly_map.unsqueeze(1),  # (B, 1, Hf, Wf)
                                size=(x.shape[2], x.shape[3]),
                                mode='bilinear',
                                align_corners=True)
    anomaly_map = anomaly_map.squeeze().cpu()

    # 10. Image-level score
    image_score = anomaly_map.max().item()

    return anomaly_map, image_score


def model_anomaly_score_batch(model, x, y, scale=0):
    """
    Batch-aware anomaly scoring function for HGAD.

    Args:
        model: trained HGAD model
        x (Tensor): input image tensor, shape (B, 3, H, W)
        y (Tensor): class indices, shape (B,)
        args: argparse arguments (for pos_embed_dim, device)
        scale (int): feature level to use

    Returns:
        anomaly_maps (Tensor): (B, H, W)
        image_scores (Tensor): (B,)
    """
    features = model.encoder(x)                     # list of feature levels
    feat = features[scale]                          # (B, C, Hf, Wf)
    B, C, Hf, Wf = feat.shape

    feat = feat.permute(0, 2, 3, 1).reshape(B * Hf * Wf, C)  # (B*Hf*Wf, C)
    label_r = y.view(B, 1, 1).expand(B, Hf, Wf).reshape(-1)  # (B*Hf*Wf,)
    y_onehot = F.one_hot(label_r, num_classes=model.n_classes).float()  # (B*Hf*Wf, n_classes)

    # Positional embedding
    pos_embed = nfs.positionalencoding2d(args.pos_embed_dim, Hf, Wf).to(args.device)
    pos_embed = pos_embed.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, C, Hf, Wf)
    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(B * Hf * Wf, -1)

    z, log_jac_det = model.nfs[scale](feat, [pos_embed])  # z: (B*Hf*Wf, C), log_jac_det: (B*Hf*Wf,)
    log_jac_det = log_jac_det.reshape(B * Hf * Wf)

    # Compute intra-class centers
    main_centers = model.mus[scale].detach()                        # (n_classes, dim)
    mu_delta = model.mu_deltas[scale].detach()                      # (n_classes, n_centers-1, dim)
    mu_all = torch.cat([main_centers.unsqueeze(1),                 # (n_classes, 1, dim)
                        main_centers.unsqueeze(1) + mu_delta], 1)  # (n_classes, n_centers, dim)
    mu_intra = mu_all[label_r]                                     # (B*Hf*Wf, n_centers, dim)

    phi = model.phi_intras[scale].detach()                         # (n_classes, n_centers)
    log_py = torch.log_softmax(phi, dim=1)[label_r]                # (B*Hf*Wf, n_centers)

    # Likelihood score
    S_l = 1.0 - model.get_logps(z, mu_intra, log_py, log_jac_det, C)  # (B*Hf*Wf,)

    # Entropy score
    zz = model.calculate_distances_to_inter_class_centers(z, main_centers)  # (B*Hf*Wf, n_classes)
    probs = torch.softmax(-0.5 * zz, dim=1)
    S_e = torch.sum(-probs * torch.log(probs + 1e-8), dim=1)  # (B*Hf*Wf,)

    # Final anomaly score
    anomaly_score = S_l * S_e                                  # (B*Hf*Wf,)
    anomaly_maps = anomaly_score.reshape(B, Hf, Wf)            # (B, Hf, Wf)
    anomaly_maps = F.interpolate(anomaly_maps.unsqueeze(1),    # (B, 1, Hf, Wf)
                                  size=(x.shape[2], x.shape[3]),
                                  mode='bilinear', align_corners=True).squeeze(1)  # (B, H, W)

    image_scores = anomaly_maps.view(B, -1).max(dim=1).values  # (B,)

    return anomaly_maps, image_scores


def compute_anomaly_metrics(gt_labels, pred_scores, gt_masks, pred_maps):
    """
    Args:
        gt_labels (np.ndarray): (N,) 이미지 단위 정상(0)/이상(1) 레이블
        pred_scores (np.ndarray): (N,) 이미지 단위 예측 점수 (e.g., max anomaly score)
        gt_masks (np.ndarray): (N, H, W) 픽셀 단위 레이블 (0 or 1)
        pred_maps (np.ndarray): (N, H, W) 픽셀 단위 예측 anomaly score

    Returns:
        dict with 'image_auroc', 'pixel_auroc', 'pixel_ap'
    """

    # flatten pixel-wise arrays
    gt_masks_flat = gt_masks.flatten()
    pred_maps_flat = pred_maps.flatten()
    
    gt_masks_flat_bin = (gt_masks_flat > 0).astype(np.uint8)

    print("\nStart computing metrics...")
    image_auroc = roc_auc_score(gt_labels, pred_scores)
    # pixel_auroc = roc_auc_score(gt_masks_flat, pred_maps_flat)
    pixel_ap = average_precision_score(gt_masks_flat_bin, pred_maps_flat)
    print("End computing metrics!\n")

    return {
        'image_auroc': image_auroc,
        # 'pixel_auroc': pixel_auroc,
        'pixel_ap': pixel_ap
    }


def calculate_scores(model, dataloader, device):
    """
    Evaluate HGAD model over a dataset using anomaly scoring.

    Args:
        model: trained HGAD model
        dataloader: test dataloader (yields image, label, mask, name, type)
        device: torch device

    Returns:
        Dictionary with AUROC and AP scores
    """
    model.to(device).eval()
    
    image_scores = []
    pixel_maps = []
    gt_labels = []
    gt_masks = []

    normal_count = 0
    anomaly_count = 0
    for i, (image, label, mask, anomaly) in tqdm(enumerate(dataloader)):
        B = image.shape[0]
        image = image.to(device)
        label = label.to(device)

        if B == 1:
            if int(anomaly.item()) == 0:
                normal_count += 1
            else:
                anomaly_count += 1
            
            anomaly_map, image_score = model_anomaly_score(model, image, label)
            
            # Save results
            image_scores.append(image_score)                       # scalar
            pixel_maps.append(anomaly_map.cpu().numpy())           # (H, W)
            gt_labels.append(int(anomaly.item()))                              # label: 0 (정상), 1 (이상)
            gt_masks.append(mask.squeeze().cpu().numpy())          # (H, W)
        else:
            anomaly_maps, image_scores_batch = model_anomaly_score_batch(model, image, label)
            for j in range(B):
                a = int(anomaly[j].item())
                if a == 0:
                    normal_count += 1
                else:
                    anomaly_count += 1

                # Save one sample from batch
                image_scores.append(image_scores_batch[j].item())
                pixel_maps.append(anomaly_maps[j].cpu().numpy())
                gt_labels.append(a)
                gt_masks.append(mask[j].squeeze().cpu().numpy())
        
    # Convert to numpy arrays
    image_scores = np.array(image_scores)                     # (N,)
    pixel_maps = np.stack(pixel_maps, axis=0)                 # (N, H, W)
    gt_labels = np.array(gt_labels)                           # (N,)
    gt_masks = np.stack(gt_masks, axis=0)                     # (N, H, W)

    # Normalize pixel-wise maps to [0, 1]
    image_scores = (image_scores - image_scores.min()) / (image_scores.max() - image_scores.min() + 1e-8)
    pixel_maps = (pixel_maps - pixel_maps.min()) / (pixel_maps.max() - pixel_maps.min() + 1e-8)

    # Compute AUROC, AP
    results = compute_anomaly_metrics(gt_labels, image_scores, gt_masks, pixel_maps)
    results["normal_count"] = normal_count
    results["anomaly_count"] = anomaly_count
    
    # results = {
    #     "image_scores": image_scores.tolist(),
    #     "gt_labels": gt_labels.tolist(),
    #     "pixel_maps": pixel_maps.tolist(),
    #     "gt_masks": gt_masks.tolist()
    # }
    
    return results


if __name__ == '__main__':
    args = parse_args()
    
    model_weight_path = "/workspace/MegaInspection/HGAD/outputs"
     
    json_path = os.path.join("/workspace/meta_files", f"{args.json_path}.json")
    with open(json_path, 'r') as f:
        data_dict = json.load(f)
    data_dict = data_dict['test']

    weights_list = os.listdir(os.path.join(model_weight_path, args.scenario, args.case))
    weights_list = [weight for weight in weights_list if weight.endswith("pt")]
    weights_list = sorted(weights_list, key=lambda x: int(x.split(".")[0][9:-4]))
    last_model = weights_list[-1]
    last_model_num = int(last_model.split(".")[0][9:-4])
    
    class_mapping_json_path = os.path.join(model_weight_path, 
                                        args.scenario, 
                                        args.case, 
                                        f"class_mapping_task_{last_model_num}.json")
    args.n_classes = load_class_mapping_length(class_mapping_json_path)
    
    pretrained_path = os.path.join(model_weight_path, 
                                        args.scenario, 
                                        args.case, 
                                        last_model)
    final_score_path = os.path.join(args.score_dir,
                                    args.scenario, 
                                    args.case, 
                                    args.json_path,
                                    f"model{last_model_num}.json")

    args.final_score_path = final_score_path
    
    model = HGAD(args)
    load_model(pretrained_path, model)
    
    os.makedirs(os.path.dirname(final_score_path), exist_ok=True)
    
    for i, (cls_name, samples) in enumerate(data_dict.items()):
        len_samples = len(samples)
        print(f"[{i+1}/{len(data_dict)}] {cls_name}")
        print("length of samples:", len_samples)
        print()
        # if len_samples > 1500:
        #     print(f"Sample size {len_samples} is larger than 1500, passing...")
        #     continue
        
        if os.path.exists(final_score_path):
            with open(final_score_path, 'r') as f:
                cumm_score = json.load(f)
            if cls_name in cumm_score:
                print(f"json_path: {json_path}")
                print(f"Already evaluated {cls_name} class")
                continue
        else:
            cumm_score = {}
        
        sub_data_dict = {}
        sub_data_dict[cls_name] = samples
        
        test_loader = prepare_loader_from_json_by_chunk(
            sub_data_dict,
            batch_size=args.batch_size,
            img_size=args.img_size, msk_size=args.img_size, 
            num_workers=args.num_workers, train=False,
            class_mapping_json_path=class_mapping_json_path,
            zero_shot_category=None,
        )        
        
        with torch.no_grad():
            results = calculate_scores(model, test_loader, args.device)
        
        cumm_score[cls_name] = results
        
        with open(final_score_path, 'w') as f:
            json.dump(cumm_score, f, indent=4)
        
        del test_loader
        torch.cuda.empty_cache()
        gc.collect()