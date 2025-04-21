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
from models.utils import save_model, load_model
from datasets.mvtec import MVTEC, MVTEC_CLASS_NAMES
from datasets.btad import BTAD, BTAD_CLASS_NAMES
from datasets.mvtec_3d import MVTEC3D, MVTEC3D_CLASS_NAMES
from datasets.visa import VISA, VISA_CLASS_NAMES
from datasets.union import UnionDataset
from utils import adjust_learning_rate, warmup_learning_rate, onehot

from datasets.json_loader import prepare_loader_from_json


def train(args):
    args.n_classes = args.num_classes_per_task
    
    model = HGAD(args)
    if args.phase == "continual":
        args.pretrained_path = f"./outputs/{args.scenario}/base/HGAD_base_img.pt"
        load_model(args.pretrained_path, model)
    model.to(args.device)
    
    if args.phase == "base":
        
        print("Starting base training...")
        train_loader = prepare_loader_from_json(args.json_path, task_id=None,
                                                    batch_size=args.batch_size,
                                                    img_size=args.img_size, msk_size=args.img_size, train=True)
        
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
        
        optimizer = model.optimizer

        for task_id in range(1, args.num_tasks + 1):
            print(f'[Task {task_id}] Loading task data...')
            train_loader = prepare_loader_from_json(args.json_path, task_id=task_id,
                                                    batch_size=args.batch_size,
                                                    img_size=args.img_size, msk_size=args.img_size, train=True)
            
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

                print(f'[Task {task_id}] Epoch {epoch+1}/{args.meta_epochs} done.')

            save_model(args.output_dir, model, args.meta_epochs, f'task{task_id}', flag='img')
            print(f'[Task {task_id}] Finished and model saved.')

    else:
        raise ValueError("Unknown phase. Choose 'base' or 'continual'.")