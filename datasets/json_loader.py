import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.utils.data import DataLoader

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class JSONDataset(Dataset):
    dataset_root_map = {
        "continual_ad": "/datasets/MegaInspection/megainspection",
        "mvtec_anomaly_detection": "/datasets/MegaInspection/non_megainspection/MVTec",
        "VisA_20220922": "/datasets/MegaInspection/non_megainspection/VisA",
        "Real-IAD-512": "/datasets/MegaInspection/non_megainspection/Real-IAD",
        "VIADUCT": "/datasets/MegaInspection/non_megainspection/VIADUCT",
        "BTAD": "/datasets/MegaInspection/non_megainspection/BTAD",
        "MPDD": "/datasets/MegaInspection/non_megainspection/MPDD"
    }

    def resolve_path(self, relative_path):
        if not relative_path:
            return None
        if os.path.isabs(relative_path):
            return relative_path
        parts = relative_path.split("/", 1)
        if len(parts) != 2:
            return None
        prefix, sub_path = parts
        root = self.dataset_root_map.get(prefix, "")
        return os.path.normpath(os.path.join(root, sub_path))
    
    def __init__(self, json_data, img_size=336, crp_size=336, msk_size=336, train=True,
                 output_path=None, prev_class_mapping_path=None, save_class_mapping=False):
        
        self.samples = []
        self.num_all_samples = 0
        for cls_name, samples in json_data.items():
            for sample in samples:
                self.num_all_samples += 1
                img_path = self.resolve_path(sample["img_path"])
                anomaly = sample.get("anomaly", 0)

                if train:
                    if anomaly != 0:
                        continue  # ⛔ skip abnormal sample during training
                    mask_path = ""  # not used
                else:
                    mask_path = self.resolve_path(sample["mask_path"]) if sample.get("mask_path") else ""

                self.samples.append((img_path, cls_name, mask_path, anomaly))
                
        self.data = json_data
        self.train = train
        self.masksize = msk_size

        self.transform = T.Compose([
            T.Resize(img_size),
            T.CenterCrop(crp_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

        self.target_transform = T.Compose([
            T.Resize(msk_size, Image.NEAREST),
            T.CenterCrop(msk_size),
            T.ToTensor()
        ])
        
        # === 1. 기존 매핑 불러오기
        if prev_class_mapping_path and os.path.exists(prev_class_mapping_path):
            with open(prev_class_mapping_path, 'r') as f:
                self.class_to_idx = json.load(f)
                self.class_to_idx = {k: int(v) for k, v in self.class_to_idx.items()}
            print(f"[INFO] Loaded class_to_idx mapping from {prev_class_mapping_path}")
        else:
            self.class_to_idx = {}
        
        # === 2. 현재 json에서 등장한 클래스 추가
        current_class_names = sorted(set(json_data.keys()))
        max_index = max(self.class_to_idx.values(), default=-1)
        for cls in current_class_names:
            if cls not in self.class_to_idx:
                max_index += 1
                self.class_to_idx[cls] = max_index
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        # === 3. 필요한 경우 저장
        if save_class_mapping:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(self.class_to_idx, f, indent=4)
            print(f"[INFO] Saved class_to_idx mapping to {output_path}")
        
        
        print(f"All samples: {self.num_all_samples}")
        print(f"All samples: {self.num_all_samples}")
        print(f"Loaded {len(self.samples)} samples from {len(current_class_names)} classes.")
        print(f"Loaded {len(self.samples)} samples from {len(current_class_names)} classes.")
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls_name, mask_path, anomaly = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        if mask_path:
            mask = Image.open(mask_path)
            mask = self.target_transform(mask)
        else:
            mask = torch.zeros((1, self.masksize, self.masksize), dtype=torch.float32)

        label = self.class_to_idx[cls_name]
        
        return image, label, mask, anomaly


def prepare_loader_from_json(json_path, task_id=None, batch_size=8, img_size=336, 
                             msk_size=336, num_workers=8, train=True,
                             output_dir=None, save_class_mapping=False):
    json_path = os.path.join("/workspace/meta_files", f"{json_path}.json")
    with open(json_path, 'r') as f:
        data_dict = json.load(f)
    if task_id is None or task_id == 0:
        data = data_dict['train'] if train else data_dict['test']
        json_file_name = "class_mapping_base.json"
        prev_class_mapping_path = None
    else:
        task_key = f'task_{task_id}'
        data = data_dict[task_key]['train'] if train else data_dict[task_key]['test']
        json_file_name = f"class_mapping_{task_key}.json"
        if task_id == 1:
            prev_output_dir = os.path.dirname(output_dir)
            prev_class_mapping_path = os.path.join(prev_output_dir, "base", "class_mapping_base.json")
        else:
            prev_class_mapping_path = os.path.join(output_dir, f"class_mapping_task_{task_id-1}.json")

    output_path = os.path.join(output_dir, json_file_name)
    save_class_mapping = save_class_mapping and train
    
    dataset = JSONDataset(
        data,
        img_size=img_size,
        crp_size=img_size,
        msk_size=msk_size,
        train=train,
        output_path=output_path,
        prev_class_mapping_path=prev_class_mapping_path,
        save_class_mapping=save_class_mapping,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)
    return loader



class JSONDatasetForChunk(Dataset):
    dataset_root_map = {
        "continual_ad": "/datasets/MegaInspection/megainspection",
        "mvtec_anomaly_detection": "/datasets/MegaInspection/non_megainspection/MVTec",
        "VisA_20220922": "/datasets/MegaInspection/non_megainspection/VisA",
        "Real-IAD-512": "/datasets/MegaInspection/non_megainspection/Real-IAD",
        "VIADUCT": "/datasets/MegaInspection/non_megainspection/VIADUCT",
        "BTAD": "/datasets/MegaInspection/non_megainspection/BTAD",
        "MPDD": "/datasets/MegaInspection/non_megainspection/MPDD"
    }

    def resolve_path(self, relative_path):
        if not relative_path:
            return None
        if os.path.isabs(relative_path):
            return relative_path
        parts = relative_path.split("/", 1)
        if len(parts) != 2:
            return None
        prefix, sub_path = parts
        root = self.dataset_root_map.get(prefix, "")
        return os.path.normpath(os.path.join(root, sub_path))
    
    def __init__(self, json_data, img_size=336, crp_size=336, msk_size=336, train=False,
                 class_mapping_json_path=None):
        
        self.samples = []
        self.num_all_samples = 0
        for cls_name, samples in json_data.items():
            for sample in samples:
                self.num_all_samples += 1
                img_path = self.resolve_path(sample["img_path"])
                anomaly = sample.get("anomaly", 0)

                if train:
                    if anomaly != 0:
                        continue  # ⛔ skip abnormal sample during training
                    mask_path = ""  # not used
                else:
                    mask_path = self.resolve_path(sample["mask_path"]) if sample.get("mask_path") else ""

                self.samples.append((img_path, cls_name, mask_path, anomaly))
                
        self.train = train
        self.masksize = msk_size

        self.transform = T.Compose([
            T.Resize(img_size),
            T.CenterCrop(crp_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

        self.target_transform = T.Compose([
            T.Resize(msk_size, Image.NEAREST),
            T.CenterCrop(msk_size),
            T.ToTensor()
        ])
        
        # === 1. 기존 매핑 불러오기
        with open(class_mapping_json_path, 'r') as f:
            self.class_to_idx = json.load(f)
            self.class_to_idx = {k: int(v) for k, v in self.class_to_idx.items()}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        print(f"[INFO] Loaded class_to_idx mapping from {class_mapping_json_path}")

        print(f"All samples: {self.num_all_samples}")
        print(f"All samples: {self.num_all_samples}")
        print(f"Loaded {len(self.samples)} samples from {len(self.class_to_idx)} classes.")
        print(f"Loaded {len(self.samples)} samples from {len(self.class_to_idx)} classes.")
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls_name, mask_path, anomaly = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        if mask_path:
            mask = Image.open(mask_path)
            mask = self.target_transform(mask)
        else:
            mask = torch.zeros((1, self.masksize, self.masksize), dtype=torch.float32)

        label = self.class_to_idx[cls_name]
        
        return image, label, mask, anomaly



def prepare_loader_from_json_by_chunk(json_data, batch_size=8, img_size=336, 
                             msk_size=336, num_workers=8, train=False,
                             class_mapping_json_path=None):
    
    dataset = JSONDatasetForChunk(
        json_data,
        img_size=img_size,
        crp_size=img_size,
        msk_size=msk_size,
        train=train,
        class_mapping_json_path=class_mapping_json_path,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)
    return loader