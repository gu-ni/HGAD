import os
import json
import argparse
from typing import List
from tqdm import tqdm
import numpy as np
import re
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter

import torch.optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import models.nf_model as nfs
from models.model import HGAD
from models.utils import save_model, load_model, load_model_partial
from datasets.mvtec import MVTEC, MVTEC_CLASS_NAMES
from datasets.btad import BTAD, BTAD_CLASS_NAMES
from datasets.mvtec_3d import MVTEC3D, MVTEC3D_CLASS_NAMES
from datasets.visa import VISA, VISA_CLASS_NAMES
from datasets.union import UnionDataset
from utils import adjust_learning_rate, warmup_learning_rate, onehot

from datasets.json_loader import prepare_loader_from_json


def load_class_mapping_and_set_n_classes(class_mapping_path):
        if not os.path.exists(class_mapping_path):
            raise FileNotFoundError(f"[ERROR] Class mapping file not found: {class_mapping_path}")
        
        with open(class_mapping_path, 'r') as f:
            class_to_idx = json.load(f)

        return len(class_to_idx)


def train(args):
    
    if args.phase == "base":
        args.n_classes = args.num_classes_per_task
        model = HGAD(args)
        model.to(args.device)
        
        print("Starting base training...")
        train_loader = prepare_loader_from_json(args.json_path, task_id=None,
                                                    batch_size=args.batch_size,
                                                    img_size=args.img_size, msk_size=args.img_size, 
                                                    num_workers=args.num_workers, train=True,
                                                    output_dir=args.output_dir,
                                                    save_class_mapping=True)
        
        optimizer = model.optimizer
        for epoch in range(args.meta_epochs):
            model.train()
            
            for idx, (image, label) in enumerate(train_loader):
            
                # x: (N, 3, 256, 256) y: (N, )
                image, label = image.to(args.device), label.to(args.device)  # (N, num_classes)
                
                with torch.no_grad():
                    features = model.encoder(image)
                
                for lvl in range(args.feature_levels):
                    e = features[lvl].detach()  
                    bs, dim, h, w = e.size()
                    e = e.permute(0, 2, 3, 1).reshape(-1, dim)  # (bs*h*w, dim)
                    
                    label_r = label.view(-1, 1, 1).repeat([1, h, w])
                    
                    label_onehot = onehot(label_r.reshape(-1), args.num_classes_per_task, args.label_smoothing)
                    
                    # (bs, 128, h, w)
                    pos_embed = nfs.positionalencoding2d(args.pos_embed_dim, h, w).to(args.device).unsqueeze(0).repeat(bs, 1, 1, 1)
                    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
            
                    # losses: all loss items, L_x_tr, logits_tr, L_cNLL_tr, L_y_tr, acc_tr
                    losses = model(e, (label_r, label_onehot), pos_embed, scale=lvl, epoch=epoch)

                    if epoch < 2:  # only training with inter-class loss
                        loss = args.lambda1 * losses['L_g'] - args.lambda2 * losses['L_mi'] + losses['L_e']
                        losses['L_g_intra'] = torch.tensor([-1])
                        losses['L_z'] = torch.tensor([-1])
                    else:
                        loss = args.lambda1 * losses['L_g'] - args.lambda2 * losses['L_mi'] + losses['L_g_intra']  + losses['L_z'] + losses['L_e'] 
                    losses['loss'] = loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            print(f'[Base] Epoch {epoch+1}/{args.meta_epochs} completed.')
            
        save_model(args.output_dir, model, args.meta_epochs, 'base', flag='img')
        print(f'[Base] Training complete. Model saved.')
    
    elif args.phase == 'continual':
        print("Starting continual learning...")  
        # 1. 이전까지 학습된 클래스 매핑 경로
        if args.task_id == 1:
            prev_class_mapping_path = os.path.join(os.path.dirname(args.output_dir), "base", "class_mapping_base.json")
        else:
            prev_class_mapping_path = os.path.join(args.output_dir, f"class_mapping_task_{args.task_id - 1}.json")

        # 2. 현재 task까지 반영된 매핑 경로
        curr_class_mapping_path = os.path.join(args.output_dir, f"class_mapping_task_{args.task_id}.json")

        # 3. 현재 데이터 로딩 및 mapping 업데이트
        train_loader = prepare_loader_from_json(
            json_path=args.json_path,
            task_id=args.task_id,
            batch_size=args.batch_size,
            img_size=args.img_size,
            msk_size=args.img_size,
            num_workers=args.num_workers,
            train=True,
            output_dir=args.output_dir,
            save_class_mapping=True
        )

        # 4. 전체 클래스 수 로드
        args.n_classes = load_class_mapping_and_set_n_classes(curr_class_mapping_path)

        # 5. 모델 초기화 및 weight load
        model = HGAD(args)
        pretrained_path = (
            f"/workspace/MegaInspection/HGAD/outputs/{args.scenario}/base/HGAD_base_img.pt" if args.task_id == 1
            else os.path.join(args.output_dir, f"HGAD_task{args.task_id - 1}_img.pt")
        )
        load_model_partial(pretrained_path, model, curr_num_classes=args.n_classes)
        model.to(args.device)
            
            



        
        print(f'[Task {args.task_id}] Loading task data...')
        train_loader = prepare_loader_from_json(args.json_path, task_id=args.task_id,
                                                batch_size=args.batch_size,
                                                img_size=args.img_size, msk_size=args.img_size, 
                                                num_workers=args.num_workers, train=True,
                                                output_dir=args.output_dir,
                                                save_class_mapping=True)
        
        
        class_mapping_path = os.path.join(args.output_dir, f"class_mapping_task_{args.task_id}.json")

        num_all_classes = load_class_mapping_and_set_n_classes(class_mapping_path)
        print(f"[INFO] Loaded class_to_idx with {args.n_classes} classes from {class_mapping_path}")
        
        args.n_classes = args.num_classes_per_task
        model = HGAD(args)
        load_model_partial(
            path=args.pretrained_path,
            model=model,
            curr_num_classes=num_all_classes,
        )
        model.to(args.device)
        
        optimizer = model.optimizer
        
        
        
        
        for epoch in range(args.meta_epochs):
            model.train()
            for idx, (image, label, *_ ) in enumerate(train_loader):
                image, label = image.to(args.device), label.to(args.device)
                with torch.no_grad():
                    features = model.encoder(image)

                for lvl in range(args.feature_levels):
                    e = features[lvl].detach()
                    bs, dim, h, w = e.size()
                    e = e.permute(0, 2, 3, 1).reshape(-1, dim)

                    label_r = label.view(-1, 1, 1).repeat([1, h, w])
                    label_onehot = onehot(label_r.reshape(-1), args.num_classes_per_task, args.label_smoothing)

                    pos_embed = nfs.positionalencoding2d(args.pos_embed_dim, h, w).to(args.device).unsqueeze(0).repeat(bs, 1, 1, 1)
                    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)

                    losses = model(e, (label_r, label_onehot), pos_embed, scale=lvl, epoch=epoch)

                    if epoch < 2:
                        loss = args.lambda1 * losses['L_g'] - args.lambda2 * losses['L_mi'] + losses['L_e']
                    else:
                        loss = args.lambda1 * losses['L_g'] - args.lambda2 * losses['L_mi'] + losses['L_g_intra'] + losses['L_z'] + losses['L_e']

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            print(f'[Task {args.task_id}] Epoch {epoch+1}/{args.meta_epochs} done.')

        save_model(args.output_dir, model, args.meta_epochs, f'task{args.task_id}', flag='img')
        print(f'[Task {args.task_id}] Finished and model saved.')

    else:
        raise ValueError("Unknown phase. Choose 'base' or 'continual'.")