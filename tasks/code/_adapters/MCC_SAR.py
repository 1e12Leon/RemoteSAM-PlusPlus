import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import torch
import torch.utils.data
from torch.utils.data import Sampler
import utils
from PIL import Image
import numpy as np
from tasks.code.model import RemoteSAM
from tasks.code.runtime import load_model_and_text_encoder
import warnings
import re

warnings.filterwarnings("ignore")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DATASETS = {
    "FUSAR-SHIP": {
        "folder": "New_FUSAR",
        "split": "Val",
        "table_name": "FUSAR-SHIP",
    },
    "MSTAR": {
        "folder": "MSTAR_SOC",
        "split": "TEST",
        "table_name": "MSTAR",
    },
    "SAR-ACD": {
        "folder": "SAR_ACD",
        "split": "images",
        "table_name": "SAR-ACD",
    },
    "SAR-VSA": {
        "folder": "MyDataset",
        "split": "TEST",
        "table_name": "SAR-VSA",
    },
    "New_FUSAR": {
        "folder": "New_FUSAR",
        "split": "Val",
        "table_name": "FUSAR-SHIP",
    },
    "MSTAR_SOC": {
        "folder": "MSTAR_SOC",
        "split": "TEST",
        "table_name": "MSTAR",
    },
    "SAR_ACD": {
        "folder": "SAR_ACD",
        "split": "images",
        "table_name": "SAR-ACD",
    },
    "SAR-VSA-MyDataset": {
        "folder": "MyDataset",
        "split": "TEST",
        "table_name": "SAR-VSA",
    },
    "MyDataset": {
        "folder": "MyDataset",
        "split": "TEST",
        "table_name": "SAR-VSA",
    },
}


CLASSNAME_ALIASES = {
    "CoastalLands_island": "Coastal Lands island",
    "LandPatches": "Land Patches",
    "OtherShip": "Other Ship",
    "SeaClutterWaves": "Sea Clutter Waves",
    "SeaPatches": "Sea Patches",
    "StrongFalseAlarms": "Strong False Alarms",
    "Boeing737": "Boeing 737",
    "Boeing787": "Boeing 787",
}


def prompt_classname(classname):
    if classname in CLASSNAME_ALIASES:
        return CLASSNAME_ALIASES[classname]
    classname = classname.replace("_", " ").replace("-", " ")
    classname = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", classname)
    return " ".join(classname.split())


class DistributedSequentialNoPadSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = utils.get_world_size()
        if rank is None:
            rank = utils.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.indices = list(range(rank, len(dataset), num_replicas))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class SAR_MCC_Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        if self.dataset not in DATASETS:
            raise ValueError(
                "MCC_SAR dataset must be one of {}, but got {}".format(
                    sorted(DATASETS), self.dataset
                )
            )
        self.dataset_info = DATASETS[self.dataset]
        self.root = args.imageFolder
        self.image_root = self._resolve_image_root()
        self.raw_classes = self._load_classes()
        self.classes = [prompt_classname(classname) for classname in self.raw_classes]
        self.image_list, self.image2groundtruth = self._load_data()

    def _resolve_image_root(self):
        folder = self.dataset_info["folder"]
        expected_leaf = self.dataset_info["split"]
        if os.path.basename(os.path.normpath(self.root)) == expected_leaf:
            return self.root

        candidate = os.path.join(self.root, folder, expected_leaf)
        if os.path.isdir(candidate):
            return candidate

        candidate = os.path.join(self.root, expected_leaf)
        if os.path.isdir(candidate):
            return candidate

        raise FileNotFoundError(
            "Cannot find image root for {} under {}. Expected leaf directory '{}'.".format(
                self.dataset, self.root, expected_leaf
            )
        )

    def _load_classes(self):
        classes = [
            name
            for name in os.listdir(self.image_root)
            if os.path.isdir(os.path.join(self.image_root, name))
        ]
        classes = sorted(classes)
        if not classes:
            raise RuntimeError("No class folders found under {}".format(self.image_root))
        return classes

    def _load_data(self):
        image_list = []
        image2groundtruth = {}

        for index, classname in enumerate(self.raw_classes):
            class_dir = os.path.join(self.image_root, classname)
            for root, _, filenames in os.walk(class_dir):
                for filename in sorted(filenames):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in IMAGE_EXTENSIONS:
                        continue
                    image_path = os.path.relpath(os.path.join(root, filename), self.image_root)
                    image_list.append(image_path)
                    image2groundtruth[image_path] = index

        image_list = sorted(image_list)
        if not image_list:
            raise RuntimeError("No images found under {}".format(self.image_root))
        return image_list, image2groundtruth

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_root, image_name)
        origin_image = Image.open(image_path).convert("RGB")
        groundtruth = self.image2groundtruth[image_name]
        return torch.tensor(np.array(origin_image)), torch.tensor(groundtruth), image_name

    def __len__(self):
        return len(self.image_list)


def eval(metric_total):
    correct_top1, correct_top5, total = metric_total.cpu().double()
    if correct_top1 > total or correct_top5 > total:
        raise RuntimeError(
            "Invalid MCC_SAR accuracy counters: top1={}, top5={}, total={}".format(
                int(correct_top1.item()), int(correct_top5.item()), int(total.item())
            )
        )
    top1_accuracy = correct_top1 / (total + 1e-15)
    top5_accuracy = correct_top5 / (total + 1e-15)

    print("\n########## Evaluation Summary ##########")
    print(
        "Top-1 Accuracy: {:.3f}% ({}/{})".format(
            top1_accuracy.item() * 100.0,
            int(correct_top1.item()),
            int(total.item()),
        )
    )
    print(
        "Top-5 Accuracy: {:.3f}% ({}/{})".format(
            top5_accuracy.item() * 100.0,
            int(correct_top5.item()),
            int(total.item()),
        )
    )
    print()


def classification(model, data_loader, bert_model, args, device):
    model = RemoteSAM(
        model,
        device,
        use_EPOC=args.EPOC,
        args=args,
        text_encoder=bert_model,
        EPOC_threshold=0.25,
    )
    metric_logger = utils.MetricLogger(delimiter="  ")

    classnames = data_loader.dataset.classes
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, "Test:"):
            origin_image, groundtruth, image_name = data

            groundtruth = groundtruth.item()
            _, prob_ins = model.multi_class_cls(
                image=origin_image.squeeze(0).numpy(),
                classnames=classnames,
                return_prob=True,
            )

            pred = max(classnames, key=lambda classname: prob_ins[classname])
            pred_index = classnames.index(pred)
            topk = min(5, len(classnames))
            topk_predictions = sorted(
                classnames,
                key=lambda classname: prob_ins[classname],
                reverse=True,
            )[:topk]
            topk_indices = {classnames.index(classname) for classname in topk_predictions}
            correct_top1 += int(pred_index == groundtruth)
            correct_top5 += int(groundtruth in topk_indices)
            total += 1

            del groundtruth, origin_image, image_name, prob_ins

    if utils.is_dist_avail_and_initialized():
        gathered = [None for _ in range(utils.get_world_size())]
        torch.distributed.all_gather_object(
            gathered,
            (correct_top1, correct_top5, total),
        )
        correct_top1 = sum(item[0] for item in gathered)
        correct_top5 = sum(item[1] for item in gathered)
        total = sum(item[2] for item in gathered)

    return torch.tensor([correct_top1, correct_top5, total], dtype=torch.long)


def main(args):
    utils.init_distributed_mode(args)

    dataset_test = SAR_MCC_Dataset(args)
    if utils.get_rank() == 0:
        print("Dataset: {}".format(dataset_test.dataset_info["table_name"]))
        print("Dataset arg: {}".format(args.dataset))
        print("Image root: {}".format(dataset_test.image_root))
        print("Class folders: {}".format(", ".join(dataset_test.raw_classes)))
        print("Prompt classes: {}".format(", ".join(dataset_test.classes)))
        print("Images: {}".format(len(dataset_test)))

    if utils.is_dist_avail_and_initialized():
        test_sampler = DistributedSequentialNoPadSampler(
            dataset_test,
            num_replicas=utils.get_world_size(),
            rank=utils.get_rank(),
        )
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
    )

    model, bert_model, device = load_model_and_text_encoder(
        args,
        distributed=utils.is_dist_avail_and_initialized(),
    )
    metric_total = classification(
        model,
        data_loader_test,
        bert_model,
        args=args,
        device=device,
    )

    if utils.get_rank() == 0:
        eval(metric_total)


if __name__ == "__main__":
    from args import get_parser

    parser = get_parser()
    args = parser.parse_args()
    main(args)
