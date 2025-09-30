import argparse
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.datamodule import CTDataModule
from src.model import DiffusionLitModule

def main(config):
    # --- 1. シードの設定 ---
    if config['training']['seed'] is not None:
        seed_everything(config['training']['seed'], workers=True)

    # --- 2. データモジュールの初期化 ---
    datamodule = CTDataModule(**config['data'])

    # --- 3. LightningModuleの初期化 ---
    model = DiffusionLitModule(
        model_config=config['model'],
        optimizer_config=config['optimizer'],
        scheduler_config=config['scheduler']
    )

    # --- 4. コールバックとロガーの設定 ---
    # チェックポイントコールバック：検証損失に基づいて最良のモデルを保存
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['trainer']['default_root_dir'],
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    # 学習率ロガー
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # TensorBoardロガー
    logger = TensorBoardLogger(
        save_dir=config['trainer']['default_root_dir'],
        name="logs"
    )

    # --- 5. Trainerの設定 ---
    # PyTorch LightningのTrainerが分散学習、混合精度などをすべて管理する
    trainer = Trainer(
        # --- ハードウェア/分散学習設定 ---
        accelerator=config['trainer']['accelerator'],  # "gpu" or "cpu"
        devices=config['trainer']['devices'],          # 使用するGPUの数 (e.g., 3)
        strategy=config['trainer']['strategy'],        # "fsdp_native" for model parallelism

        # --- 精度とパフォーマンス設定 ---
        precision=config['trainer']['precision'],      # "bf16-mixed" for Ampere GPUs

        # --- 学習設定 ---
        max_epochs=config['trainer']['max_epochs'],

        # --- ロギングとチェックポイント ---
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=config['trainer']['default_root_dir'],

        # --- その他 ---
        log_every_n_steps=config['trainer']['log_every_n_steps'],
        accumulate_grad_batches=config['trainer']['accumulate_grad_batches']
    )

    # --- 6. 学習の開始 ---
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 3D conditional diffusion model using PyTorch Lightning.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training configuration YAML file.",
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)