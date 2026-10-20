import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

import transforms as T
from lib.modal_inference import infer_modal_tensor_from_image
from tasks.code.runtime import load_model_and_text_encoder
from lib.transformers_compat import patch_optional_sklearn

patch_optional_sklearn()
from transformers import AutoTokenizer


CLASSES = OrderedDict([
    ('impervious_surface', (255, 255, 255)),
    ('building', (0, 0, 255)),
    ('low_vegetation', (0, 255, 255)),
    ('tree', (0, 255, 0)),
    ('car', (255, 204, 0)),
    ('clutter', (255, 0, 0)),
])


def label_map(image):
    rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
    out = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for idx, color in enumerate(CLASSES.values()):
        out[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = idx
    return out


def main(args):
    image_dir, label_dir = Path(args.imageFolder), Path(args.annoFolder)
    labels = sorted(label_dir.glob('*.png'))
    if len(labels) != 113:
        raise RuntimeError(f'Expected 113 Vaihingen test labels, found {len(labels)} in {label_dir}')
    image_names = {p.stem: p for p in image_dir.glob('*.tif')}
    missing = [p.stem for p in labels if p.stem not in image_names]
    if missing:
        raise FileNotFoundError(f'Missing matching Vaihingen images ({len(missing)}): {missing[:5]}')

    model, bert_model, device = load_model_and_text_encoder(args, distributed=False)
    model.eval()
    if bert_model is not None:
        bert_model.eval()
    tokenizer_path = args.modernbert_path if getattr(args, 'text_encoder', 'modernbert') == 'modernbert' else args.bert_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    prompts = {}
    for name in CLASSES:
        tok = tokenizer(f'{name.replace("_", " ")} in the image.', max_length=20, padding='max_length', truncation=True, return_tensors='pt')
        prompts[name] = (tok['input_ids'].to(device), tok['attention_mask'].to(device))
    transform = T.Compose([T.Resize(args.img_size, args.img_size), T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])
    hist = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    with torch.no_grad():
        for label_path in labels:
            image = Image.open(image_names[label_path.stem]).convert('RGB')
            gt = label_map(Image.open(label_path))
            image_tensor, _ = transform(image, image)
            image_tensor = image_tensor.unsqueeze(0).to(device)
            modal = infer_modal_tensor_from_image(image, device=device)
            probs = []
            for ids, attn in prompts.values():
                if bert_model is not None:
                    hidden = bert_model(ids, attention_mask=attn)[0].permute(0, 2, 1).to(dtype=image_tensor.dtype)
                    out = model(image_tensor, hidden, l_mask=attn.unsqueeze(-1), modal=modal)
                else:
                    out = model(image_tensor, ids, l_mask=attn, modal=modal)
                probs.append(torch.softmax(out, dim=1)[:, 1])
            # argmax over classes leaves (N, H, W); add only the channel dimension.
            pred = torch.stack(probs).argmax(0).unsqueeze(1).float()
            pred = F.interpolate(pred, size=gt.shape, mode='nearest').squeeze().cpu().numpy().astype(np.uint8)
            valid = gt != 255
            hist += np.bincount(len(CLASSES) * gt[valid].astype(np.int64) + pred[valid], minlength=len(CLASSES) ** 2).reshape(hist.shape)
    tp = np.diag(hist).astype(float); denom = hist.sum(1) + hist.sum(0) - tp
    iou = tp / np.maximum(denom, 1); f1 = 2 * tp / np.maximum(hist.sum(1) + hist.sum(0), 1)
    mean_indices = [i for i, name in enumerate(CLASSES) if name != 'clutter']
    print('Vaihingen semantic segmentation results:')
    for name, a, b in zip(CLASSES, iou, f1): print(f'{name}: IoU={a*100:.2f} F1={b*100:.2f}')
    print(f'mIoU (excluding clutter)={iou[mean_indices].mean()*100:.2f}; mF1 (excluding clutter)={f1[mean_indices].mean()*100:.2f}')
    print(f'mIoU (all 6 classes)={iou.mean()*100:.2f}; mF1 (all 6 classes)={f1.mean()*100:.2f}; pixel_acc={tp.sum()/max(hist.sum(),1)*100:.2f}')
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        json.dump({'mIoU': float(iou[mean_indices].mean()*100), 'mF1': float(f1[mean_indices].mean()*100), 'mIoU_all_classes': float(iou.mean()*100), 'mF1_all_classes': float(f1.mean()*100), 'pixel_acc': float(tp.sum()/max(hist.sum(),1)*100)}, open(os.path.join(args.save_path, 'seg_vaihingen_results.json'), 'w'), indent=2)


if __name__ == '__main__':
    from args import get_parser
    parser = get_parser()
    main(parser.parse_args())
