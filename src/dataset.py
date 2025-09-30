import os
import torch
from torch.utils.data import Dataset
from pathlib import Path

class XrayCTDataset(Dataset):
    """
    X線画像（正面・側面）と3D CT画像をロードするためのカスタムデータセット。

    ディレクトリ構造：
    - 入力 (X線): <base_dir>/<pt_dir>/AP.pt, <base_dir>/<pt_dir>/LAT.pt
    - 出力 (CT):  <base_dir2>/<pt_dir>.pt
    """
    def __init__(self, base_dir, base_dir2, transform=None):
        """
        Args:
            base_dir (str): X線画像が格納されている親ディレクトリのパス。
            base_dir2 (str): CT画像が格納されている親ディレクトリのパス。
            transform (callable, optional): データに適用されるオプションの変換。
        """
        self.base_dir = Path(base_dir)
        self.base_dir2 = Path(base_dir2)
        self.transform = transform

        # base_dir内のすべての患者ディレクトリ（pt_dir）をリストアップ
        self.patient_dirs = sorted([d for d in self.base_dir.iterdir() if d.is_dir()])

    def __len__(self):
        """
        データセットのサンプル数を返す。
        """
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        """
        指定されたインデックスのデータサンプルをロードして返す。

        Args:
            idx (int): サンプルのインデックス。

        Returns:
            dict: {
                'conditioning_images': torch.Tensor, # (2, H, W)
                'target_image': torch.Tensor         # (D, H, W)
            }
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pt_dir_path = self.patient_dirs[idx]
        pt_id = pt_dir_path.name

        ap_path = pt_dir_path / "AP.pt"
        lat_path = pt_dir_path / "LAT.pt"
        ct_path = self.base_dir2 / f"{pt_id}.pt"

        try:
            ap_image = torch.load(ap_path, map_location="cpu")
            lat_image = torch.load(lat_path, map_location="cpu")
            ct_image = torch.load(ct_path, map_location="cpu")

            # X線画像をスタック
            conditioning_images = torch.stack([ap_image, lat_image], dim=0)

            # CT画像が (D, H, W) 形式でない場合は調整が必要
            # ここでは (D, H, W) と仮定
            # チャンネル次元を追加して (1, D, H, W) にする
            if ct_image.dim() == 3:
                ct_image = ct_image.unsqueeze(0)

            sample = {'conditioning_images': conditioning_images, 'target_image': ct_image}

            if self.transform:
                sample = self.transform(sample)

            return sample

        except FileNotFoundError as e:
            print(f"Warning: File not found, skipping index {idx}. Path: {e.filename}")
            return None

def custom_collate_fn(batch):
    """
    __getitem__がNoneを返す可能性のあるバッチを処理するためのcollate_fn。
    """
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)