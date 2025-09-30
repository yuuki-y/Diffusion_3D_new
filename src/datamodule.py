import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from src.dataset import XrayCTDataset, custom_collate_fn

class CTDataModule(pl.LightningDataModule):
    """
    X線-CTデータセット用のLightningDataModule。
    データセットの準備、分割、データローダーの作成をカプセル化する。
    """
    def __init__(self,
                 base_dir: str,
                 base_dir2: str,
                 batch_size: int = 1,
                 num_workers: int = 4,
                 train_val_split_ratio: float = 0.9):
        super().__init__()
        self.base_dir = base_dir
        self.base_dir2 = base_dir2
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split_ratio = train_val_split_ratio

        # LightningModuleで保存したいハイパーパラメータ
        self.save_hyperparameters()

    def setup(self, stage: str = None):
        """
        データのダウンロード、分割、前処理などを実行する。
        Trainer.fit()やTrainer.test()が呼ばれると、対応するstageで実行される。
        """
        if stage == "fit" or stage is None:
            full_dataset = XrayCTDataset(self.base_dir, self.base_dir2)
            dataset_size = len(full_dataset)
            train_size = int(dataset_size * self.train_val_split_ratio)
            val_size = dataset_size - train_size

            # データセットを訓練用と検証用にランダムに分割
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        if stage == "test" or stage is None:
            # ここでは検証用データセットをテスト用として再利用する
            # 実際のシナリオでは、専用のテストデータセットをロードする
            if not hasattr(self, 'val_dataset'):
                 full_dataset = XrayCTDataset(self.base_dir, self.base_dir2)
                 dataset_size = len(full_dataset)
                 train_size = int(dataset_size * self.train_val_split_ratio)
                 val_size = dataset_size - train_size
                 _, self.test_dataset = random_split(full_dataset, [train_size, val_size])
            else:
                self.test_dataset = self.val_dataset


    def train_dataloader(self):
        """
        訓練用データローダーを作成して返す。
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        """
        検証用データローダーを作成して返す。
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        """
        テスト用データローダーを作成して返す。
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
        )