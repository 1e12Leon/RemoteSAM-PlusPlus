import torch
import os
from PIL import Image
import json
from tqdm import tqdm
import xml.etree.cElementTree as ET
import numpy as np


class RSVG_VG(torch.utils.data.Dataset):

    def __init__(self, args, max_tokens=20):

        self.args = args
        self.split = args.split
        self.max_tokens = max_tokens
        
        self.image_path = args.imageFolder
        self.anno_path = args.annoFolder

        self.image_list, self.groundtruth, self.sentence = self.load_data()
    

    def load_data(self):

        image_list = []
        groundtruth = []
        sentence = []
        
        split_txt = os.path.join(os.path.dirname(self.anno_path), self.split + ".txt")
        with open(split_txt, "r", encoding="utf-8") as f:
            lines = f.readlines()
            Index = [ int(line.strip()) for line in lines ]
        
        count = 0
        anno_list = sorted(os.listdir(self.anno_path))
        for anno in tqdm(anno_list, desc="Generating sentence embeddings", total=len(anno_list)):
            root = ET.parse(os.path.join(self.anno_path, anno)).getroot()
            image_name = root.find('filename').text
            objects = []
            for obj in root.findall('object'):
                if count in Index:

                    bndbox = obj.find('bndbox')
                    xmin = bndbox.find('xmin').text
                    ymin = bndbox.find('ymin').text
                    xmax = bndbox.find('xmax').text
                    ymax = bndbox.find('ymax').text
                    box = [int(xmin), int(ymin), int(xmax), int(ymax)]

                    description = obj.find('description').text

                    image_list.append(image_name)
                    groundtruth.append(box)
                    sentence.append(description)
                
                count += 1

        return image_list, groundtruth, sentence


    def __getitem__(self, idx):

        image_name = self.image_list[idx]
        origin_image = Image.open(os.path.join(self.image_path, image_name)).convert('RGB')

        # return image, gt, image_name, sentence
        return torch.tensor(np.array(origin_image)), torch.tensor(self.groundtruth[idx]), image_name, self.sentence[idx]


    def __len__(self):
        return len(self.image_list)


class SARVG_VG(torch.utils.data.Dataset):

    def __init__(self, args, max_tokens=20):

        self.args = args
        self.split = args.split
        self.max_tokens = max_tokens

        self.image_path = args.imageFolder
        self.anno_path = args.annoFolder
        self.dataset_root = os.path.dirname(os.path.normpath(self.anno_path))
        self.split_path = os.path.join(self.dataset_root, "ImageSets", self.split + ".txt")

        self.image_list, self.groundtruth, self.sentence = self.load_data()

    def load_split_ids(self):
        if not os.path.isfile(self.split_path):
            raise FileNotFoundError(f"SARVG split file not found: {self.split_path}")

        with open(self.split_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def load_data(self):

        image_list = []
        groundtruth = []
        sentence = []

        split_ids = self.load_split_ids()
        skipped_ids = []
        loaded_xml = 0

        for sample_id in tqdm(split_ids, desc="Loading SARVG annotations", total=len(split_ids)):
            anno_file = os.path.join(self.anno_path, sample_id + ".xml")
            if not os.path.isfile(anno_file):
                skipped_ids.append(sample_id)
                continue

            root = ET.parse(anno_file).getroot()
            image_name = root.findtext("filename")
            if not image_name:
                image_name = sample_id + ".jpg"

            image_file = os.path.join(self.image_path, image_name)
            if not os.path.isfile(image_file):
                skipped_ids.append(sample_id)
                continue

            loaded_xml += 1
            for obj in root.findall("object"):
                bndbox = obj.find("bndbox")
                description = obj.findtext("description")
                if bndbox is None or not description:
                    continue

                xmin = int(float(bndbox.findtext("xmin")))
                ymin = int(float(bndbox.findtext("ymin")))
                xmax = int(float(bndbox.findtext("xmax")))
                ymax = int(float(bndbox.findtext("ymax")))

                image_list.append(image_name)
                groundtruth.append([xmin, ymin, xmax, ymax])
                sentence.append(description)

        print(
            "SARVG split '{}' loaded: {} samples from {} XML files; skipped {} split ids without local XML/image.".format(
                self.split, len(image_list), loaded_xml, len(skipped_ids)
            )
        )
        if skipped_ids:
            print("SARVG skipped id examples:", skipped_ids[:5])

        if not image_list:
            raise RuntimeError(
                f"No SARVG samples loaded from {self.split_path}. Check imageFolder/annoFolder and split ids."
            )

        return image_list, groundtruth, sentence

    def __getitem__(self, idx):

        image_name = self.image_list[idx]
        origin_image = Image.open(os.path.join(self.image_path, image_name)).convert("RGB")

        return torch.tensor(np.array(origin_image)), torch.tensor(self.groundtruth[idx]), image_name, self.sentence[idx]

    def __len__(self):
        return len(self.image_list)
