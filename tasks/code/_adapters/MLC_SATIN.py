"""Zero-shot multi-label classification on locally downloaded SATIN parquet files.

SATIN publishes one ``train`` split per constituent dataset; this evaluator
reports results on that split and never claims it is an official test split.
"""
import argparse
import glob
import io
import os
import time
import json

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm import tqdm

import utils
from tasks.code.model_auto import RemoteSAM
from tasks.code.runtime import load_model_and_text_encoder


# These are the label orders used by the original constituent datasets.
CLASSNAMES = {
    "UC_Merced_LandUse_MultiLabel": [
        "airplane", "bare soil", "buildings", "cars", "chaparral", "court",
        "dock", "field", "grass", "mobile home", "pavement", "sand",
        "sea", "ship", "tanks", "trees", "water",
    ],
    "AID_MultiLabel": [
        "airplane", "bare soil", "buildings", "cars", "chaparral", "court",
        "dock", "field", "grass", "mobile home", "pavement", "sand",
        "sea", "ship", "tanks", "trees", "water",
    ],
    "MultiScene": [
        "apron", "baseball field", "basketball field", "beach", "bridge", "cemetery",
        "commercial", "farmland", "woodland", "golf course", "greenhouse", "helipad",
        "lake or pond", "oil field", "orchard", "parking lot", "park", "pier", "port",
        "quarry", "railway", "residential", "river", "roundabout", "runway", "soccer",
        "solar panel", "sparse shrub", "stadium", "storage tank", "tennis court",
        "train station", "wastewater plant", "wind turbine", "works", "sea",
    ],
    "MLRSNet": [
        "airplane", "airport", "bare soil", "baseball diamond", "basketball court", "beach",
        "bridge", "buildings", "cars", "chaparral", "cloud", "containers", "crosswalk",
        "dense residential area", "desert", "dock", "factory", "field", "football field",
        "forest", "freeway", "golf course", "grass", "greenhouse", "gully", "habor",
        "intersection", "island", "lake", "mobile home", "mountain", "overpass", "park",
        "parking lot", "parkway", "pavement", "railway", "railway station", "river", "road",
        "roundabout", "runway", "sand", "sea", "ships", "snow", "snowberg",
        "sparse residential area", "stadium", "swimming pool", "tanks", "tennis court", "terrace",
        "track", "trail", "transmission tower", "trees", "water", "wetland", "wind turbine",
    ],
}


def parquet_files(root, name):
    files = sorted(glob.glob(os.path.join(root, name, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found for {name} under {root}")
    return files


def evaluate_dataset(model, bert, args, device, name):
    files = parquet_files(args.data_root, name)
    parquet = [pq.ParquetFile(path) for path in files]
    classes = CLASSNAMES[name]
    if len(classes) != args.class_counts[name]:
        raise AssertionError("internal class-count mismatch")
    total = sum(p.metadata.num_rows for p in parquet)

    # Contiguous exact sharding gives every rank a predictable amount of work.
    # Unlike modulo sharding, this also makes progress/straggler diagnosis clear.
    if args.standalone_shard:
        rank, world = args.shard_index, args.num_shards
    else:
        rank, world = utils.get_rank(), utils.get_world_size()
    shard_start = (total * rank) // world
    shard_end = (total * (rank + 1)) // world
    local_total = shard_end - shard_start
    stats = torch.zeros((4, len(classes)), dtype=torch.double, device=device)
    exact = torch.zeros(1, dtype=torch.double, device=device)
    start = time.time()
    wrapper = RemoteSAM(model, device, use_EPOC=args.EPOC, args=args,
                        text_encoder=bert, EPOC_threshold=0.25)
    progress = tqdm(
        total=local_total,
        desc=f"{name} (rank {rank} shard)",
        unit="img",
        disable=rank != 0,
    )
    with torch.no_grad():
        global_idx = 0
        for pf in parquet:
            for batch in pf.iter_batches(columns=["image", "label"], batch_size=32):
                for record in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
                    idx = global_idx
                    global_idx += 1
                    if idx < shard_start or idx >= shard_end:
                        continue
                    payload = record[0].get("bytes") if isinstance(record[0], dict) else None
                    labels_row = record[1] or []
                    max_id = max(labels_row, default=-1)
                    if max_id >= len(classes):
                        raise ValueError(f"{name}: row {idx} label id {max_id} exceeds {len(classes)} class names")
                    if payload is None:
                        raise ValueError(f"{name}: row {idx} has no image.bytes payload")
                    image = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
                    pred_names, _ = wrapper.multi_label_cls(image=image, classnames=classes, return_prob=True)
                    gt_ids = set(int(x) for x in labels_row)
                    gt = torch.tensor([i in gt_ids for i in range(len(classes))], dtype=torch.bool, device=device)
                    pred = torch.tensor([c in pred_names for c in classes], dtype=torch.bool, device=device)
                    stats[0] += (gt & pred).double()
                    stats[1] += ((~gt) & (~pred)).double()
                    stats[2] += ((~gt) & pred).double()
                    stats[3] += (gt & (~pred)).double()
                    exact += (gt == pred).all().double()
                    if rank == 0:
                        progress.update(1)
    progress.close()
    if args.standalone_shard:
        stats_dir = os.path.abspath(args.stats_dir)
        os.makedirs(stats_dir, exist_ok=True)
        stats_path = os.path.join(stats_dir, f"{name}.shard{rank}of{world}.json")
        payload = {
            "dataset": name,
            "total": total,
            "rank": rank,
            "world": world,
            "stats": stats.cpu().tolist(),
            "exact": float(exact.item()),
            "elapsed_minutes": (time.time() - start) / 60,
        }
        with open(stats_path, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Saved shard statistics: {stats_path}")
        return

    torch.distributed.all_reduce(stats)
    torch.distributed.all_reduce(exact)
    if rank == 0:
        tp, tn, fp, fn = stats
        acc = (tp.sum() + tn.sum()) / (stats.sum() + 1e-15)
        print(f"\n{name} (SATIN train split; samples={total}):")
        print(f"  Accuracy (micro over labels): {acc.item() * 100:.2f}%")
        print(f"  Exact-match sample accuracy: {exact.item() / total * 100:.2f}%")
        print(f"  elapsed: {(time.time() - start) / 60:.1f} min")


def main(args):
    if args.standalone_shard:
        args.device = f"cuda:{args.device_index}"
        model, bert, device = load_model_and_text_encoder(args, distributed=False)
    else:
        utils.init_distributed_mode(args)
        model, bert, device = load_model_and_text_encoder(args, distributed=True)
    args.class_counts = {k: len(v) for k, v in CLASSNAMES.items()}
    for name in args.datasets:
        evaluate_dataset(model, bert, args, device, name)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(CLASSNAMES))
    parser.add_argument("--standalone_shard", action="store_true",
                        help="Run one independent shard without NCCL collectives.")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--stats_dir", default="./logs/mlc_satin_shards")
    args = parser.parse_args()
    main(args)
