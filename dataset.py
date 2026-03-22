import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os

def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    dataset_key = dataset_name.lower()

    if dataset_key in {'mvtec', 'mvtec-ad'}:
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    elif dataset_key == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_key == 'mpdd':
        obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    elif dataset_key == 'btad':
        obj_list = ['01', '02', '03']
    elif dataset_key == 'dagm':
        obj_list = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
    elif dataset_key in {'sdd', 'kolektorsdd'}:
        obj_list = ['electrical commutators']
    elif dataset_key in {'dtd', 'dtd-synthetic'}:
        obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
    elif dataset_key == 'colon':
        obj_list = ['colon']
    elif dataset_key == 'isbi':
        obj_list = ['skin']
    elif dataset_key == 'chest':
        obj_list = ['chest']
    elif dataset_key == 'thyroid':
        obj_list = ['thyroid']
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id

class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.requested_mode = mode
        self.mode = mode
        self.data_all = []
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        if mode not in meta_info:
            if 'train' in meta_info:
                self.mode = 'train'
            elif 'test' in meta_info:
                self.mode = 'test'
            else:
                self.mode = next(iter(meta_info.keys()))
            print(f"[Dataset] split '{mode}' not found in {self.root}/meta.json, fallback to '{self.mode}'")
        meta_info = meta_info[self.mode]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)
        self.dataset_name = dataset_name
        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                # just for classification not report error
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        # transforms
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(   
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        if self.dataset_name == 'mvtec' and self.mode == 'train':
            prefix_name, _ = os.path.splitext(img_path)
            pt_path = os.path.join(self.root, prefix_name + '.pt')
            if os.path.exists(pt_path):
                llm_embedding = torch.load(pt_path)
                llm_embedding = llm_embedding.unsqueeze(0)  # Add a batch dimension
                llm_embedding = torch.nn.functional.interpolate(llm_embedding, size=(77, 768), mode='bilinear',
                                                                align_corners=False)

                return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                        'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name], "llm_embedding": llm_embedding}
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}
