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


class SARDet_MLC(COCO_style):

    classes = ['ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor']


class MSAR_MLC(COCO_style):

    classes = ['ship', 'aircraft', 'oil tank', 'bridge']

    def load_data(self):
        image_list = []
        image2groundtruth = {}

        with open(self.anno_path, 'r') as f:
            data = json.load(f)

        id2image = {item["id"]: item["file_name"] for item in data['images']}
        id2category = {item["id"]: item["name"] for item in data.get('categories', [])}
        class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        for item in tqdm(data['annotations'], desc="Collecting groundtruth", total=len(data['annotations'])):
            image_name = id2image[item["image_id"]]
            if image_name not in image2groundtruth:
                image2groundtruth[image_name] = [0] * len(self.classes)
                image_list.append(image_name)

            category_name = id2category.get(item["category_id"])
            if category_name in class_to_idx:
                image2groundtruth[image_name][class_to_idx[category_name]] = 1

        return sorted(image_list), image2groundtruth


class BigEarthNetS1_MLC(torch.utils.data.Dataset):

    classes = [
        "Agro-forestry areas",
        "Arable land",
        "Beaches, dunes, sands",
        "Broad-leaved forest",
        "Coastal wetlands",
        "Complex cultivation patterns",
        "Coniferous forest",
        "Industrial or commercial units",
        "Inland waters",
        "Inland wetlands",
        "Land principally occupied by agriculture, with significant areas of natural vegetation",
        "Marine waters",
        "Mixed forest",
        "Moors, heathland and sclerophyllous vegetation",
        "Natural grassland and sparsely vegetated areas",
        "Pastures",
        "Permanent crops",
        "Transitional woodland, shrub",
        "Urban fabric",
    ]

    def __init__(self, args, max_tokens=20):
        self.args = args
        self.split = "validation" if args.split == "val" else args.split
        self.max_tokens = max_tokens
        self.root = args.imageFolder
        self.metadata_path = args.annoFolder or os.path.join(self.root, "metadata.parquet")
        self.s1_root = os.path.join(self.root, "BigEarthNet-S1")
        self.limit = int(os.environ.get("BIGEARTHNET_LIMIT", "0") or 0)
        self.filter_bad_patches = os.environ.get("BIGEARTHNET_FILTER_BAD", "0") == "1"
        self.subset_fraction = float(os.environ.get("BIGEARTHNET_SUBSET_FRACTION", "0") or 0)
        self.subset_seed = int(os.environ.get("BIGEARTHNET_SUBSET_SEED", "2026") or 2026)
        self.focus_classes = [
            item.strip() for item in os.environ.get("BIGEARTHNET_FOCUS_CLASSES", "").split("|")
            if item.strip()
        ]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.has_groundtruth = True
        self.samples = self.load_data()

    @staticmethod
    def acquisition_name(s1_name):
        # Example: S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57
        # Folder:  S1B_IW_GRDH_1SDV_20170612T165809
        return "_".join(str(s1_name).split("_")[:5])

    def load_data(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "BigEarthNetS1_MLC requires pandas and pyarrow. Install them in the active env with: "
                "python -m pip install pandas pyarrow"
            ) from exc

        if not os.path.isdir(self.s1_root):
            raise FileNotFoundError(
                f"Missing extracted BigEarthNet-S1 directory: {self.s1_root}. "
                "Run tasks/MLC_BigEarthNetS1.sh once to extract BigEarthNet-S1.tar.zst."
            )
        if not os.path.isfile(self.metadata_path):
            raise FileNotFoundError(f"Missing BigEarthNet metadata parquet: {self.metadata_path}")

        dataframe = pd.read_parquet(self.metadata_path)
        dataframe = dataframe[dataframe["split"] == self.split]
        if self.filter_bad_patches:
            dataframe = dataframe[
                (~dataframe["contains_seasonal_snow"])
                & (~dataframe["contains_cloud_or_shadow"])
            ]
        if self.subset_fraction > 0:
            frac = min(max(self.subset_fraction, 0.0), 1.0)
            target_count = max(1, int(round(len(dataframe) * frac)))
            if self.focus_classes:
                focus_set = set(self.focus_classes)
                focus_mask = dataframe["labels"].map(lambda labels: bool(focus_set.intersection(labels)))
                focus_df = dataframe[focus_mask]
                if len(focus_df) >= target_count:
                    dataframe = focus_df.sample(n=target_count, random_state=self.subset_seed)
                else:
                    rest_df = dataframe[~focus_mask]
                    need = target_count - len(focus_df)
                    rest_sample = rest_df.sample(n=min(need, len(rest_df)), random_state=self.subset_seed)
                    dataframe = pd.concat([focus_df, rest_sample], ignore_index=True)
                    dataframe = dataframe.sample(frac=1.0, random_state=self.subset_seed)
                print(
                    "BigEarthNet-S1 subset: fraction={} target={} focus_classes={}".format(
                        frac, len(dataframe), ", ".join(self.focus_classes)
                    )
                )
            else:
                dataframe = dataframe.sample(n=target_count, random_state=self.subset_seed)
                print(f"BigEarthNet-S1 random subset: fraction={frac} target={len(dataframe)} seed={self.subset_seed}")
        if self.limit > 0:
            dataframe = dataframe.head(self.limit)

        samples = []
        skipped_missing = 0
        for row in tqdm(dataframe.itertuples(index=False), desc="Loading BigEarthNet-S1 metadata", total=len(dataframe)):
            s1_name = row.s1_name
            patch_dir = os.path.join(self.s1_root, self.acquisition_name(s1_name), s1_name)
            vv_path = os.path.join(patch_dir, f"{s1_name}_VV.tif")
            vh_path = os.path.join(patch_dir, f"{s1_name}_VH.tif")
            if not os.path.isfile(vv_path) or not os.path.isfile(vh_path):
                skipped_missing += 1
                continue

            target = np.zeros(len(self.classes), dtype=np.int64)
            for label in row.labels:
                if label in self.class_to_idx:
                    target[self.class_to_idx[label]] = 1
            samples.append((s1_name, vv_path, vh_path, target))

        if skipped_missing:
            print(f"Warning: skipped {skipped_missing} BigEarthNet-S1 samples with missing VV/VH files.")
        if not samples:
            raise RuntimeError(
                f"No BigEarthNet-S1 samples loaded for split={self.split} from {self.s1_root}."
            )
        print(f"Loaded BigEarthNet-S1 split={self.split}: {len(samples)} samples, {len(self.classes)} labels.")
        return samples

    @staticmethod
    def sar_to_uint8(array):
        array = np.asarray(array, dtype=np.float32)
        finite = np.isfinite(array)
        if not finite.any():
            return np.zeros(array.shape, dtype=np.uint8)
        valid = array[finite]
        low, high = np.percentile(valid, [2.0, 98.0])
        if high <= low:
            low, high = float(valid.min()), float(valid.max())
        if high <= low:
            return np.zeros(array.shape, dtype=np.uint8)
        array = np.clip((array - low) / (high - low), 0.0, 1.0)
        return (array * 255.0).astype(np.uint8)

    def read_s1_image(self, vv_path, vh_path):
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError(
                "BigEarthNetS1_MLC requires tifffile. Install it with: python -m pip install tifffile"
            ) from exc

        vv = self.sar_to_uint8(tifffile.imread(vv_path))
        vh = self.sar_to_uint8(tifffile.imread(vh_path))
        avg = ((vv.astype(np.uint16) + vh.astype(np.uint16)) // 2).astype(np.uint8)
        return np.stack([vv, vh, avg], axis=-1)

    def __getitem__(self, idx):
        image_name, vv_path, vh_path, target = self.samples[idx]
        image = self.read_s1_image(vv_path, vh_path)
        return torch.tensor(image), torch.tensor(target), image_name

    def __len__(self):
        return len(self.samples)
