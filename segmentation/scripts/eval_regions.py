
from pathlib import Path
import argparse
import torch
from omegaconf import OmegaConf

from cross_view_transformer.common import setup_network, setup_data_module
from cross_view_transformer.model.geometry_kernel_transformer_encoder import bev2image_sampling


class RegionalIoU:
    def __init__(self, min_visibility=None):
        self.min_visibility = min_visibility
        self.masks = None
        self.points = None
        self.counts = {}

    @torch.no_grad()
    def update(self, outputs, batch, indices):
        pred = outputs["bev"].sigmoid() >= 0.5
        target = batch["bev"][:, indices].any(dim=1, keepdim=True)
        visibility = batch["visibility"][:, None]

        # Original GKT: keep only the front 50 meters
        if pred.shape[-2:] == (200, 200):
            pred = pred[..., :100, :]
            target = target[..., :100, :]
            visibility = visibility[..., :100, :]

        assert pred.shape == target.shape
        assert pred.shape[-2:] == (100, 200)

        B, _, H, W = pred.shape

        if self.masks is None:
            rows = torch.arange(H, device=pred.device)[:, None]
            cols = torch.arange(W, device=pred.device)[None, :]

            # GKT: ego at bottom, front direction at top
            x = ((H - rows - 0.5) * 0.5).expand(H, W)
            y = ((cols + 0.5) * 0.5 - 50.0).expand(H, W)

            self.masks = {
                "Complete BEV": torch.ones(
                    (H, W), dtype=torch.bool, device=pred.device
                ),
                "X 0-10 m": (x >= 0) & (x < 10),
                "X 10-25 m": (x >= 10) & (x < 25),
                "X 25-50 m": (x >= 25) & (x < 50),
                "Center |Y|<=20": y.abs() <= 20,
                "Lateral |Y|>20": y.abs() > 20,
                "Close and lateral": (
                    (x < 10) & (y.abs() > 20)
                ),
            }

            # BEV pixel centers in ego coordinates
            self.points = torch.stack(
                [x, -y, torch.ones_like(x)], dim=-1
            ).reshape(-1, 3)

            self.counts = {
                name: torch.zeros(
                    3, dtype=torch.long, device=pred.device
                )
                for name in [
                    *self.masks,
                    "Covered (3cam)",
                    "Uncovered (3cam)"
                ]
            }

        # Project all BEV pixels onto the three front cameras
        _, mask = bev2image_sampling(
            self.points,
            batch["intrinsics"][:, :3],
            batch["extrinsics"][:, :3],
            batch["image"].shape[-2],
            batch["image"].shape[-1]
        )

        covered = mask.squeeze(-1).any(dim=1).reshape(B, 1, H, W)

        # Static regions + sample-dependent camera coverage
        masks = {
            **self.masks,
            "Covered (3cam)": covered,
            "Uncovered (3cam)": ~covered
        }

        for name, region in masks.items():
            valid = region[None, None] if region.ndim == 2 else region

            if self.min_visibility is not None:
                valid = valid & (visibility >= self.min_visibility)

            self.counts[name] += torch.stack([
                (pred & target & valid).sum(),
                (pred & ~target & valid).sum(),
                (~pred & target & valid).sum()
            ])

    def report(self):
        lines = [
            "\n========== REGION METRICS (@0.50) ==========",
            f"{'Region':<23} {'IoU':>8} {'Precision':>10} "
            f"{'Recall':>10} {'TP':>12} {'FP':>12} {'FN':>12}"
        ]

        for name, counts in self.counts.items():
            tp, fp, fn = counts.cpu().tolist()

            iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0

            lines.append(
                f"{name:<23} "
                f"{iou:>8.4f} "
                f"{precision:>10.4f} "
                f"{recall:>10.4f} "
                f"{tp:>12d} "
                f"{fp:>12d} "
                f"{fn:>12d}"
            )

        # Check that coverage masks partition the complete BEV
        assert torch.equal(
            self.counts["Covered (3cam)"] +
            self.counts["Uncovered (3cam)"],
            self.counts["Complete BEV"]
        ), "Coverage counts do not match Complete BEV"

        lines.append("\nCoverage partition verification: OK")
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--pretrained", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument(
        "-l", "--logdir",
        default="./logs/lightning_logs/eval_regions"
    )
    parser.add_argument("-b", "--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--min_visibility", type=int, default=None)

    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")

    checkpoint = torch.load(args.pretrained, map_location="cpu")
    cfg = OmegaConf.create(checkpoint["hyper_parameters"])

    cfg.data.dataset_dir = args.dataset_dir
    cfg.data.labels_dir = args.labels_dir
    cfg.loader.batch_size = args.batch_size
    cfg.loader.num_workers = args.num_workers
    cfg.loader.prefetch_factor = 2

    # Both variants must use cameras 0, 1, 2 as the front cameras
    assert list(cfg.data.cameras[0][:3]) == [0, 1, 2]

    model = setup_network(cfg)

    weights = {
        key[len("backbone."):]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("backbone.")
    }

    model.load_state_dict(weights, strict=True)
    model = model.to(device).eval()

    loader = setup_data_module(cfg).val_dataloader(shuffle=False)
    indices = list(cfg.data.label_indices[0])
    regional = RegionalIoU(args.min_visibility)

    print("Checkpoint:", args.pretrained, flush=True)
    print(
        "BEV:", cfg.data.bev.h, "x", cfg.data.bev.w,
        "| Cameras:", cfg.data.cameras, flush=True
    )

    with torch.no_grad():
        for step, batch in enumerate(loader):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            if step == 0:
                assert batch["bev"].shape[-2:] == (
                    cfg.data.bev.h, cfg.data.bev.w
                ), "Incompatible BEV labels"

                assert batch["image"].shape[1] == len(
                    cfg.data.cameras[0]
                ), "Incompatible camera count"

            regional.update(model(batch), batch, indices)

            if (step + 1) % 100 == 0:
                print("Processed batches:", step + 1, flush=True)

    report = regional.report()
    print(report)

    output_path = (
        Path(args.logdir) /
        Path(args.pretrained).stem /
        "regional_metrics.txt"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "Checkpoint: " + args.pretrained + "\n" + report + "\n",
        encoding="utf-8"
    )

    print("\nResults saved in:", output_path)


if __name__ == "__main__":
    main()
