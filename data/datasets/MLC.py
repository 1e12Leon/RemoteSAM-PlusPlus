import torch
import os
from PIL import Image
import json
from tqdm import tqdm
import numpy as np


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


class COCO_style(torch.utils.data.Dataset):

    classes = None

    def __init__(self, args, max_tokens=20) :

        self.args = args
        self.split = args.split
        self.max_tokens = max_tokens

        self.image_path = args.imageFolder
        self.anno_path = args.annoFolder

        self.has_groundtruth = True
        self.image_list, self.image2groundtruth = self.load_data()


    def load_data(self):

        image_list = []
        image2groundtruth = {}

        with open(self.anno_path, 'r') as f:
            data = json.load(f)
        
        id2image = {}
        for item in data['images']:
            id2image[item["id"]] = item["file_name"]

        for item in tqdm(data['annotations'], desc="Collecting groundtruth", total=len(data['annotations'])):
            image_name = id2image[item["image_id"]]
            if image_name not in image2groundtruth:
                image2groundtruth[image_name] = [0] * len(self.classes)
                image_list.append(image_name)
            
            image2groundtruth[image_name][int(item["category_id"])-1] = 1

        return sorted(image_list), image2groundtruth

            
    def __getitem__(self, idx):

        image_name = self.image_list[idx]
        origin_image = Image.open(os.path.join(self.image_path, image_name)).convert('RGB')
        
        # return image, gt, image_name
        return torch.tensor(np.array(origin_image)), torch.tensor(self.image2groundtruth[image_name]), image_name


    def __len__(self):
        return len(self.image_list)


class DOTA_style(COCO_style):

    convert_name = None

    def load_data(self):

        image_list = [
            name for name in os.listdir(self.image_path)
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
        ]
        image2groundtruth = {}

        for image_name in image_list:
            image2groundtruth[image_name] = [0] * len(self.classes)

        self.has_groundtruth = (
            os.path.isdir(self.anno_path)
            and any(name.endswith('.txt') for name in os.listdir(self.anno_path))
        )
        if not self.has_groundtruth:
            if self.split != 'test':
                raise FileNotFoundError(f"No DOTA annotation txt files found under {self.anno_path}")
            return sorted(image_list), image2groundtruth

        parsed_objects = 0
        unknown_classes = {}

        for image_name in tqdm(image_list, desc="Collecting groundtruth", total=len(image_list)):
            anno_name = os.path.splitext(image_name)[0] + ".txt"
            anno_file = os.path.join(self.anno_path, anno_name)
            if not os.path.isfile(anno_file):
                continue

            with open(anno_file, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) < 9:
                        continue
                    classname = self.convert_name.get(data[8], data[8])
                    if classname in self.classes:
                        image2groundtruth[image_name][self.classes.index(classname)] = 1
                        parsed_objects += 1
                    else:
                        unknown_classes[data[8]] = unknown_classes.get(data[8], 0) + 1

        if unknown_classes:
            print(f"Warning: ignored unknown DOTA classes: {unknown_classes}")

        if self.split != 'test' and parsed_objects == 0:
            raise ValueError(
                f"DOTA annotations were found under {self.anno_path}, "
                "but no objects matched the configured MLC classes."
            )

        print(f"Loaded DOTA groundtruth: {parsed_objects} objects matched {len(self.classes)} classes.")

        return sorted(image_list), image2groundtruth


class DIOR_MLC(COCO_style):

    classes = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam', 'expressway-service-area', 'expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill']


class DOTAv2_MLC(DOTA_style):

    classes = ['airplane', 'ship', 'storagetank', 'baseballfield', 'tenniscourt', 'basketballcourt', 'groundtrackfield', 'harbor', 'bridge', 'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool', "container-crane", "airport", "helipad"]
    
    # unify classname
    convert_name = {
        'plane': 'airplane',
        'storage-tank': 'storagetank',
        'baseball-diamond': 'baseballfield',
        'tennis-court': 'tenniscourt',
        'basketball-court': 'basketballcourt',
        'ground-track-field': 'groundtrackfield',
    }
