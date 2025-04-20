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
    
    def __init__(self, json_data, img_size=336, crp_size=336, msk_size=336, train=True):
        
        self.samples = []
        self.num_all_samples = 0
        for cls_name, samples in json_data.items():
            for sample in samples:
                self.num_all_samples += 1
                if train:
                    if not sample["mask_path"]:
                        img_path = self.resolve_path(sample["img_path"])
                        self.samples.append((img_path, cls_name, None, None))
                else:
                    img_path = self.resolve_path(sample["img_path"])
                    if sample["mask_path"]:
                        mask_path = self.resolve_path(sample["mask_path"])
                    else:
                        mask_path = None
                    anomaly = sample["anomaly"]
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

        self.class_to_idx = {}
        self.idx_to_class = {}

        class_names = sorted(set(json_data.keys()))
        for i, cls in enumerate(class_names):
            self.class_to_idx[cls] = i
            self.idx_to_class[i] = cls
        
        print(f"All samples: {self.num_all_samples}")
        print(f"All samples: {self.num_all_samples}")
        print(f"Loaded {len(self.samples)} samples from {len(class_names)} classes.")
        print(f"Loaded {len(self.samples)} samples from {len(class_names)} classes.")
        
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


def prepare_loader_from_json(json_path, task_id=None, batch_size=8, img_size=336, msk_size=336, train=True):
    json_path = os.path.join("/workspace/meta_files", f"{json_path}.json")
    with open(json_path, 'r') as f:
        data_dict = json.load(f)
    if task_id is None:
        data = data_dict['train'] if train else data_dict['test']
    else:
        task_key = f'task_{task_id}'
        data = data_dict[task_key]['train'] if train else data_dict[task_key]['test']

    dataset = JSONDataset(data, img_size=img_size, crp_size=img_size, msk_size=msk_size, train=train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=True)
    return loader