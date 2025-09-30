import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from diffusers.models.unets.unet_3d_condition import UNet3DConditionModel
from diffusers import DDPMScheduler

class XrayConditioningEncoder(nn.Module):
    """
    2枚のX線画像(AP, LAT)を条件ベクトルにエンコードする2D CNN。
    入力: (B, 2, H, W) のテンソル
    出力: (B, N, D) のテンソル (N=sequence_length, D=cross_attention_dim)
    """
    def __init__(self, out_dim=768, sequence_length=16):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),  # 256 -> 128
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 128 -> 64
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 64 -> 32
            nn.SiLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # 32 -> 16
            nn.SiLU(),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1), # 16 -> 8
        )
        feature_size = 1024 * 8 * 8
        self.projection = nn.Linear(feature_size, sequence_length * out_dim)
        self.out_dim = out_dim
        self.sequence_length = sequence_length

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.convnet(x)
        x = x.view(batch_size, -1)
        x = self.projection(x)
        x = x.view(batch_size, self.sequence_length, self.out_dim)
        return x

class DiffusionLitModule(pl.LightningModule):
    """
    PyTorch Lightningで3D Conditional Diffusion Modelをカプセル化する。
    """
    def __init__(self, model_config: dict, optimizer_config: dict, scheduler_config: dict):
        super().__init__()
        self.save_hyperparameters() # 設定を保存し、チェックポイントからロード可能にする

        # --- モデルの初期化 ---
        encoder_config = self.hparams.model_config['encoder_config']
        unet_config = self.hparams.model_config['unet_config']

        # U-Netのcross_attention_dimはエンコーダの出力次元と一致させる
        unet_config['cross_attention_dim'] = encoder_config.get('out_dim', 768)

        self.condition_encoder = XrayConditioningEncoder(**encoder_config)
        self.unet = UNet3DConditionModel(**unet_config)

        # --- メモリ最適化 ---
        # 勾配チェックポインティングを有効化してVRAM使用量を削減
        if self.hparams.model_config.get("gradient_checkpointing", False):
            self.unet.enable_gradient_checkpointing()

        # --- ノイズスケジューラの初期化 ---
        self.noise_scheduler = DDPMScheduler(**scheduler_config)

    def forward(self, sample, timestep, conditioning_images):
        """
        モデルのフォワードパスを定義
        """
        encoder_hidden_states = self.condition_encoder(conditioning_images)
        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states
        ).sample
        return noise_pred

    def training_step(self, batch, batch_idx):
        """
        1回のトレーニングステップ
        """
        clean_images = batch['target_image']
        conditioning_images = batch['conditioning_images']

        # ノイズとタイムステップをサンプリング
        noise = torch.randn_like(clean_images)
        bsz = clean_images.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device)
        timesteps = timesteps.long()

        # クリーンな画像にノイズを追加
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

        # ノイズを予測
        noise_pred = self(noisy_images, timesteps, conditioning_images)

        # 損失を計算
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        1回の検証ステップ
        """
        clean_images = batch['target_image']
        conditioning_images = batch['conditioning_images']

        noise = torch.randn_like(clean_images)
        bsz = clean_images.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device)
        timesteps = timesteps.long()

        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        noise_pred = self(noisy_images, timesteps, conditioning_images)
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """
        オプティマイザと（オプションで）学習率スケジューラを設定
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.optimizer_config['learning_rate'],
            betas=(self.hparams.optimizer_config['adam_beta1'], self.hparams.optimizer_config['adam_beta2']),
            weight_decay=self.hparams.optimizer_config['adam_weight_decay'],
            eps=self.hparams.optimizer_config['adam_epsilon'],
        )
        return optimizer