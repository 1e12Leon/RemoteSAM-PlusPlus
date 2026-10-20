"""Evaluate RemoteSAM++ on the labeled DeepGlobe Land Cover train folder."""
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from lib.transformers_compat import patch_optional_sklearn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
patch_optional_sklearn()

import transforms as T
from lib.modal_inference import infer_modal_tensor_from_image
from tasks.code.runtime import load_model_and_text_encoder
from transformers import AutoTokenizer


CLASSES = ("urban land", "agriculture land", "rangeland", "forest land", "water", "barren land")
COLORS = {
    (0, 255, 255): 0,
    (255, 255, 0): 1,
    (255, 0, 255): 2,
    (0, 255, 0): 3,
    (0, 0, 255): 4,
    (255, 255, 255): 5,
    (0, 0, 0): 6,
}
IGNORE_LABEL = 6


def decode_mask(mask):
    rgb = np.asarray(mask.convert("RGB"), dtype=np.uint8)
    labels = np.full(rgb.shape[:2], IGNORE_LABEL, dtype=np.uint8)
    for color, label in COLORS.items():
        labels[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=2)] = label
    return labels


def main(args):
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    pairs = []
    for image_path in sorted(image_dir.glob("*_sat.jpg")):
        mask_path = mask_dir / image_path.name.replace("_sat.jpg", "_mask.png")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
        pairs.append((image_path, mask_path))
    if not pairs:
        raise FileNotFoundError(f"No *_sat.jpg files found under {image_dir}")
    print(f"Found {len(pairs)} paired DeepGlobe images and masks.")
    start_idx = (len(pairs) * args.shard_index) // args.num_shards
    end_idx = (len(pairs) * (args.shard_index + 1)) // args.num_shards
    pairs = pairs[start_idx:end_idx]
    print(f"Processing shard {args.shard_index}/{args.num_shards}: {len(pairs)} images.")

    model, bert_model, device = load_model_and_text_encoder(args, distributed=False)
    model.eval()
    if bert_model is not None:
        bert_model.eval()
    tokenizer_path = args.modernbert_path if args.text_encoder == "modernbert" else args.bert_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenized = []
    for classname in CLASSES:
        encoded = tokenizer(
            f"{classname} in the image.", max_length=args.max_tokens,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        tokenized.append((encoded["input_ids"].to(device), encoded["attention_mask"].to(device)))

    transform = T.Compose([
        T.Resize(args.img_size, args.img_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    hist = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    with torch.no_grad():
        for index, (image_path, mask_path) in enumerate(pairs, 1):
            image = Image.open(image_path).convert("RGB")
            gt = decode_mask(Image.open(mask_path))
            x, _ = transform(image, image)
            x = x.unsqueeze(0).to(device)
            modal = infer_modal_tensor_from_image(image, device=device)
            probs = []
            for ids, attention in tokenized:
                if bert_model is not None:
                    hidden = bert_model(ids, attention_mask=attention)[0].permute(0, 2, 1).to(dtype=x.dtype)
                    output = model(x, hidden, l_mask=attention.unsqueeze(-1), modal=modal)
                else:
                    output = model(x, ids, l_mask=attention, modal=modal)
                probs.append(torch.softmax(output, dim=1)[:, 1])
            prediction = torch.stack(probs).argmax(0).unsqueeze(1).float()
            prediction = F.interpolate(prediction, size=gt.shape, mode="nearest").squeeze().cpu().numpy().astype(np.uint8)
            valid = (gt != IGNORE_LABEL)
            hist += np.bincount(
                len(CLASSES) * gt[valid].astype(np.int64) + prediction[valid],
                minlength=len(CLASSES) ** 2,
            ).reshape(hist.shape)
            if index % 25 == 0 or index == len(pairs):
                print(f"Processed {index}/{len(pairs)}", flush=True)

    if args.num_shards > 1:
        stats_dir = Path(args.stats_dir)
        stats_dir.mkdir(parents=True, exist_ok=True)
        path = stats_dir / f"deepglobe.shard{args.shard_index}of{args.num_shards}.json"
        path.write_text(json.dumps({"hist": hist.tolist(), "count": len(pairs)}), encoding="utf-8")
        print(f"Saved shard statistics to {path}")
        return

    tp = np.diag(hist).astype(float)
    iou = tp / np.maximum(hist.sum(1) + hist.sum(0) - tp, 1)
    f1 = 2 * tp / np.maximum(hist.sum(1) + hist.sum(0), 1)
    print("DeepGlobe Land Cover train-folder results (unknown excluded):")
    for classname, class_iou, class_f1 in zip(CLASSES, iou, f1):
        print(f"{classname}: IoU={class_iou * 100:.2f} F1={class_f1 * 100:.2f}")
    print(f"mIoU={iou.mean() * 100:.2f}; mF1={f1.mean() * 100:.2f}; pixel_acc={tp.sum() / max(hist.sum(), 1) * 100:.2f}; samples={len(pairs)}")


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--stats_dir", default="./logs/deepglobe_shards")
    main(parser.parse_args())
