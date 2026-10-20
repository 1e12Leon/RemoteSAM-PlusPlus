import torch
import os
from PIL import Image
import json
from tqdm import tqdm
import numpy as np


class UCM_CAP(torch.utils.data.Dataset):

    # potential classes
    classes = ["road", "highway", "bank", "tenniscourt", "roof", "vehicle", "parking lot", "airplane", "stand", "farmland", "waste land", "basketballcourt", "storagetank", "pedestrian", "villa", "beach", "boat", "tree", "grass", "water", "trail", "park", "bush", "golffield", "house", "airport", "people", "overpass", "baseballfield", "swimming-pool", "bunker", "building", "harbor", "residential", "sand", "stadium", "intersection"]

    def __init__(self, args, max_tokens=20) :

        self.args = args
        self.split = args.split
        self.max_tokens = max_tokens

        self.image_path = args.imageFolder
        self.anno_path = args.annoFolder

        self.image_list, self.image2groundtruth, self.image2id = self.load_data()

    
    def load_data(self):

        image_list = []
        image2groundtruth = {}
        image2id = {}

        with open(self.anno_path, 'r') as f:
            data = json.load(f)
        
        for item in tqdm(data['images'], desc="Collecting groundtruth", total=len(data['images'])):
            if item['split'] != self.split:
                continue

            image_name = item["filename"]
            if image_name not in image2groundtruth:
                image2groundtruth[image_name] = []
                image_list.append(image_name)
                image2id[image_name] = item["imgid"]

            for gt in item["sentences"]:
                caption = gt["raw"]
                image2groundtruth[image_name].append(caption)

        return sorted(image_list), image2groundtruth, image2id


    def __getitem__(self, idx):

        image_name = self.image_list[idx]
        origin_image = Image.open(os.path.join(self.image_path, image_name)).convert('RGB')

        # return image, gt, image_name
        return torch.tensor(np.array(origin_image)), self.image2groundtruth[image_name], image_name


    def __len__(self):
        return len(self.image_list)


class SARLANG_CAP(torch.utils.data.Dataset):

    # potential SAR captioning classes
    classes = [
        "building", "road", "bridge", "river", "water", "lake", "sea",
        "coastline", "harbor", "ship", "airport", "runway", "airplane",
        "vehicle", "railway", "farmland", "field", "forest", "tree",
        "vegetation", "mountain", "residential area", "urban area",
        "industrial area", "parking lot", "stadium", "sports field",
        "bare land", "grassland", "island", "beach", "canal",
    ]

    def __init__(self, args, max_tokens=20):
        self.args = args
        self.split = args.split
        self.max_tokens = max_tokens
        self.image_path = args.imageFolder
        self.anno_path = args.annoFolder

        self.image_list, self.image2groundtruth, self.image2id = self.load_data()

    def _caption_files(self):
        if os.path.isfile(self.anno_path):
            return [self.anno_path]

        candidates = []
        roots = [
            self.anno_path,
            os.path.join(self.anno_path, "Text", "Caption", self.split),
            os.path.join(self.anno_path, "Caption", self.split),
            os.path.join(self.anno_path, self.split),
        ]
        names = [
            f"Caption_{self.split}.json",
            f"Caption_{self.split}1.json",
        ]
        for root in roots:
            for name in names:
                path = os.path.join(root, name)
                if os.path.isfile(path) and path not in candidates:
                    candidates.append(path)
        if not candidates:
            raise FileNotFoundError(
                f"No SARLANG caption json found under {self.anno_path!r} for split {self.split!r}"
            )
        return candidates

    @staticmethod
    def _assistant_caption(item):
        messages = item.get("messages", [])
        for message in messages:
            if message.get("role") == "assistant":
                return message.get("content", "").strip()
        if messages:
            return messages[-1].get("content", "").strip()
        return ""

    def load_data(self):
        image2groundtruth = {}
        for caption_file in self._caption_files():
            with open(caption_file, "r", encoding="utf-8-sig") as f:
                data = json.load(f)

            for item in tqdm(data, desc=f"Collecting groundtruth from {os.path.basename(caption_file)}"):
                images = item.get("images") or []
                if not images:
                    continue
                image_name = os.path.basename(images[0])
                caption = self._assistant_caption(item)
                if not image_name or not caption:
                    continue
                image2groundtruth.setdefault(image_name, [])
                if caption not in image2groundtruth[image_name]:
                    image2groundtruth[image_name].append(caption)

        image_list = sorted(image2groundtruth)
        missing = [
            image_name for image_name in image_list
            if not os.path.isfile(os.path.join(self.image_path, image_name))
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} SARLANG images referenced by captions are missing under "
                f"{self.image_path!r}; first missing: {missing[:5]}"
            )

        image2id = {image_name: idx for idx, image_name in enumerate(image_list)}
        return image_list, image2groundtruth, image2id

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        origin_image = Image.open(os.path.join(self.image_path, image_name)).convert("RGB")
        return torch.tensor(np.array(origin_image)), self.image2groundtruth[image_name], image_name

    def __len__(self):
        return len(self.image_list)


class FUSARCAP_CAP(torch.utils.data.Dataset):

    # potential SAR captioning classes for FSAR-Cap
    classes = [
        "ship", "cargo ship", "large ship", "small ship", "vessel",
        "water", "sea", "open sea", "river", "coastline", "harbor",
        "port", "dock", "bridge", "building", "road", "runway",
        "airport", "vehicle", "farmland", "forest", "mountain",
        "urban area", "residential area", "industrial area",
    ]

    def __init__(self, args, max_tokens=20):
        self.args = args
        self.split = args.split
        self.max_tokens = max_tokens
        self.image_path = args.imageFolder
        self.anno_path = args.annoFolder

        self.image_list, self.image2groundtruth, self.image2id = self.load_data()

    def load_data(self):
        image_list = []
        image2groundtruth = {}
        image2id = {}

        with open(self.anno_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        for item in tqdm(data["images"], desc="Collecting FSAR-Cap groundtruth", total=len(data["images"])):
            if item.get("split") != self.split:
                continue

            image_name = item["filename"]
            captions = [
                sentence.get("raw", "").strip()
                for sentence in item.get("sentences", [])
                if sentence.get("raw", "").strip()
            ]
            if not captions:
                continue

            if image_name not in image2groundtruth:
                image2groundtruth[image_name] = []
                image_list.append(image_name)
                image2id[image_name] = item.get("imgid", len(image2id))

            for caption in captions:
                if caption not in image2groundtruth[image_name]:
                    image2groundtruth[image_name].append(caption)

        image_list = sorted(image_list)
        missing = [
            image_name for image_name in image_list
            if not os.path.isfile(os.path.join(self.image_path, image_name))
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} FSAR-Cap images referenced by captions are missing under "
                f"{self.image_path!r}; first missing: {missing[:5]}. "
                "If only all_images.rar exists, extract it first."
            )

        return image_list, image2groundtruth, image2id

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        origin_image = Image.open(os.path.join(self.image_path, image_name)).convert("RGB")
        return torch.tensor(np.array(origin_image)), self.image2groundtruth[image_name], image_name

    def __len__(self):
        return len(self.image_list)
