import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))


import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import functional as F1

from torchvision.transforms import InterpolationMode
import transforms as T
from lib.transformers_compat import patch_optional_sklearn

patch_optional_sklearn()
from transformers import AutoTokenizer

import numpy as np
import torch
import warnings
import random

import torch.backends.cudnn as cudnn

from lib import segmentation
from tasks.code.runtime import load_model_and_text_encoder

warnings.filterwarnings("ignore")


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def resolve_image_path(directory, stem):
    for suffix in (".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"):
        path = os.path.join(directory, stem + suffix)
        if os.path.isfile(path):
            return path
    return os.path.join(directory, stem + ".png")


def compute_mIoU(args, model, bert_model, cat_dictionary, refToInput, refToAttention, gt_dir, pred_dir, num_classes,
                 name_classes):
    print('Num classes', num_classes)
    hist = np.zeros((num_classes, num_classes))
    png_name_list = []
    for file in os.listdir(gt_dir):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            file_name_without_extension = os.path.splitext(file)[0]
            png_name_list.append(file_name_without_extension)

    gt_imgs = [resolve_image_path(gt_dir, x) for x in png_name_list]
    pred_imgs = [resolve_image_path(pred_dir, x[0:5]) for x in png_name_list]


    for ind in range(len(gt_imgs)):

        catToid = {}

        for index, (category, color) in enumerate(cat_dictionary.items()):
            catToid[category] = index + 1  # ??0???
        img = Image.open(pred_imgs[ind]).convert('RGB')
        target = Image.open(gt_imgs[ind]).convert('RGB')
        target = F1.resize(target, (896, 896), interpolation=InterpolationMode.NEAREST)
        o_W, o_H = target.size

        label = change_label(target, cat_dictionary)

        transformimg = get_transform(args)
        img, img2 = transformimg(img, img)
        output_list = []
        img = img.unsqueeze(0).to(args.device)

        for cat, cat_value in cat_dictionary.items():
            emb = refToInput[cat].to(img.device)
            att_mask = refToAttention[cat].to(img.device)
            modal = torch.tensor([1 if getattr(args, "modal", "rgb") == "sar" else 0], device=img.device)
            output = model(img, emb, att_mask, modal=modal)

            # output = output
            # output = F.interpolate(output, (o_H, o_W), align_corners=True, mode='bilinear')

            output = output.argmax(1).data
            # output=torch.sigmoid(output)


            output = class_colors_transform(output, catToid, cat)
            output = np.array(output.cpu()).astype(int)
            output_list.append(output)

        # merge_output = np.zeros((o_H, o_H), dtype=output_list[0].dtype)

        merge_output = np.maximum.reduce(output_list)
        pred = merge_output


        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue


        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIou-{:0.2f}; mPA-{:0.2f}'.format(ind, len(gt_imgs),
                                                                  100 * np.nanmean(per_class_iu(hist)),
                                                                  100 * np.nanmean(per_class_PA(hist))))
    # ------------------------------------------------#
    #   mIoU
    # ------------------------------------------------#
    mIoUs = per_class_iu(hist)
    mPA = per_class_PA(hist)
    for ind_class in range(num_classes):
        # ind_class=ind_class+1
        print('===>' + name_classes[ind_class] + ':\tmIou-' + str(round(mIoUs[ind_class] * 100, 2)) + '; mPA-' + str(
            round(mPA[ind_class] * 100, 2)))

    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(mPA) * 100, 2)))
    return mIoUs


def class_colors_transform(predict, catToid, cat):
    color_value = catToid[cat]
    zeros_tensor = torch.zeros_like(predict)

    transformed_predict = torch.where(predict > 0, torch.full_like(predict, color_value), zeros_tensor)
    # transformed_predict = torch.where(predict > 0.3, torch.full_like(predict, color_value), predict)

    return transformed_predict


def get_sent_embedding(cat_dictionary, tokenizer, refToInput, refToAttention, refToExist, max_tokens=20):
    for cat, _ in cat_dictionary.items():
        attention_mask = [0] * max_tokens
        padded_input_id = [0] * max_tokens
        catnew = cat + " in the image."
        input_id = tokenizer.encode(text=catnew, add_special_tokens=True)
        input_id = input_id[:max_tokens]

        padded_input_id[:len(input_id)] = input_id
        attention_mask[:len(input_id)] = [1] * len(input_id)

        input_id = torch.tensor(padded_input_id).unsqueeze(0)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0)
        exist = torch.Tensor([True])
        refToInput[cat] = input_id
        refToAttention[cat] = attention_mask
        refToExist[cat] = exist
    return refToInput, refToAttention, refToExist


def change_label(label, cat_dictionary):
    label = np.array(label).astype(int)
    label_mapped = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)

    categories = list(cat_dictionary.keys())

    for index, (category, color) in enumerate(cat_dictionary.items()):
        mask = np.all(label == color, axis=-1)
        label_mapped[mask] = index + 1

    return label_mapped


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)



def main(args):
    cudnn.benchmark = True
    cudnn.deterministic = True

    model, bert_model, device = load_model_and_text_encoder(args, distributed=False)

    model.eval()
    img_path = args.imageFolder
    label_path = args.annoFolder

    max_tokens = 20
    tokenizer_path = args.modernbert_path if getattr(args, "text_encoder", "modernbert") == "modernbert" else args.bert_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    refToInput = {}
    refToAttention = {}
    refToExist = {}
    cat_dictionary = {
        "ship": (0, 0, 63),
        "storagetank": (0, 63, 63),
        "baseballfield": (0, 63, 0),
        "tenniscourt": (0, 63, 127),
        "basketballcourt": (0, 63, 191),
        "groundtrackfield": (0, 63, 255),
        "bridge": (0, 127, 63),
        "large-vehicle": (0, 127, 127),
        "small-vehicle": (0, 0, 127),
        "helicopter": (0, 0, 191),
        "swimming-pool": (0, 0, 255),
        "roundabout": (0, 191, 127),
        "soccer-ball-field": (0, 127, 191),
        "airplane": (0, 127, 255),
        "harbor": (0, 100, 155)
    }
    name_classes = []
    name_classes.append("background")
    for keys, _ in cat_dictionary.items():
        name_classes.append(keys)
    refToInput, refToAttention, refToExist = get_sent_embedding(cat_dictionary, tokenizer, refToInput, refToAttention,
                                                                refToExist, max_tokens=20)
    with torch.no_grad():
        compute_mIoU(args, model, bert_model, cat_dictionary, refToInput, refToAttention, label_path, img_path,
                     len(cat_dictionary.keys()) + 1, name_classes)

if __name__ == '__main__':
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    main(args)
