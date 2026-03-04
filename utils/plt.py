import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

import matplotlib.pyplot as plt
from pathlib import Path

def save_visualization_samples(samples_for_vis, epoch, examples_dir):

    examples_dir = Path(examples_dir)
    epoch_dir = examples_dir / f"Epoch_{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples_for_vis:
        input_sat, target_cloud, target_ghi, pred_cloud, pred_ghi, event_name = sample
        
        event_dir = epoch_dir / str(event_name)
        event_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------
        # 1. 历史卫星输入（4 通道 × 6 时间帧）
        # input_sat shape = (4,6,H,W)
        # -------------------------------------------
        fig, axes = plt.subplots(4, 6, figsize=(18, 12))

        for c in range(4):
            for t in range(6):
                axes[c, t].imshow(input_sat[c, t], cmap='gray')
                axes[c, t].set_title(f"Ch={c}, T={t}")
                axes[c, t].axis('off')

        plt.tight_layout()
        plt.savefig(event_dir / "input_satellite.png", dpi=150, bbox_inches='tight')
        plt.close()

        # -------------------------------------------
        # Helper for 24-frame (cloud/GHI)
        # -------------------------------------------
        def plot_24_frames(frames, title_prefix, cmap, save_path, vmin=None, vmax=None):
            fig, axes = plt.subplots(4, 6, figsize=(18, 12))
            for i in range(24):
                r = i // 6
                c = i % 6
                axes[r, c].imshow(frames[i], cmap=cmap, vmin=vmin, vmax=vmax)
                axes[r, c].set_title(f"{title_prefix} {i}", fontsize=8)
                axes[r, c].axis('off')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

        # -------------------------------------------
        # 2. Cloud Target (24,H,W)
        # -------------------------------------------
        plot_24_frames(
            target_cloud, "Cloud_GT", "viridis",
            event_dir / "cloud_target.png",
            vmin=0, vmax=100
        )

        # 3. Cloud Pred
        plot_24_frames(
            pred_cloud, "Cloud_Pred", "viridis",
            event_dir / "cloud_pred.png",
            vmin=0, vmax=100
        )

        # 4. GHI Target
        plot_24_frames(
            target_ghi, "GHI_GT", "inferno",
            event_dir / "ghi_target.png"
        )

        # 5. GHI Pred
        plot_24_frames(
            pred_ghi, "GHI_Pred", "inferno",
            event_dir / "ghi_pred.png"
        )


