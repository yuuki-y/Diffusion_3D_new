import argparse
import torch
from pathlib import Path
from tqdm.auto import tqdm
from diffusers import DDIMScheduler # DDIM is often better for fast inference

from src.model import DiffusionLitModule

def main(args):
    # --- 1. デバイスの設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. モデルのロード ---
    # PyTorch Lightningのチェックポイントからモデルをロード
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    model = DiffusionLitModule.load_from_checkpoint(
        args.checkpoint_path,
        map_location=device
    )
    model.eval() # 推論モードに設定

    # --- 3. スケジューラの準備 ---
    # 推論時にはDDIMSchedulerの方が高速で高品質な場合がある
    # scheduler = DDIMScheduler.from_config(model.hparams.scheduler_config)
    scheduler = DDIMScheduler(
        num_train_timesteps=model.hparams.scheduler_config['num_train_timesteps'],
        beta_schedule=model.hparams.scheduler_config['beta_schedule']
    )
    scheduler.set_timesteps(args.num_inference_steps)
    timesteps = scheduler.timesteps

    # --- 4. 入力データの準備 ---
    print("Loading conditioning images...")
    ap_image = torch.load(args.ap_path, map_location="cpu").to(device)
    lat_image = torch.load(args.lat_path, map_location="cpu").to(device)

    # (H, W) -> (1, H, W)
    if ap_image.dim() == 2:
        ap_image = ap_image.unsqueeze(0)
    if lat_image.dim() == 2:
        lat_image = lat_image.unsqueeze(0)

    # (1, H, W) + (1, H, W) -> (1, 2, H, W)
    conditioning_images = torch.stack([ap_image, lat_image], dim=1)

    # --- 5. 推論ループ ---
    # 生成する画像の形状をモデルの設定から取得
    unet_config = model.hparams.model_config['unet_config']
    image_shape = (
        1, # batch_size
        unet_config.get('in_channels', 1),
        args.depth,
        args.height,
        args.width,
    )

    # ランダムノイズから開始
    image = torch.randn(image_shape, device=device)

    print("Starting denoising process...")
    progress_bar = tqdm(total=args.num_inference_steps)
    progress_bar.set_description("Generating CT image")

    with torch.no_grad():
        for t in timesteps:
            # タイムステップをテンソルに変換
            timestep_tensor = torch.tensor([t], device=device)

            # モデルでノイズを予測
            noise_pred = model(
                sample=image,
                timestep=timestep_tensor,
                conditioning_images=conditioning_images
            )

            # スケジューラで1ステップ前の画像を計算
            image = scheduler.step(noise_pred, t, image).prev_sample
            progress_bar.update(1)

    progress_bar.close()

    # --- 6. 出力の保存 ---
    # バッチ次元とチャンネル次元を削除 (1, 1, D, H, W) -> (D, H, W)
    output_image = image.squeeze()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_image.cpu(), output_path)

    print(f"Successfully generated and saved CT image to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a 3D CT image from a pair of X-ray images using a trained Lightning model.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained model .ckpt checkpoint file.")
    parser.add_argument("--ap_path", type=str, required=True, help="Path to the anterior-posterior (AP) X-ray .pt file.")
    parser.add_argument("--lat_path", type=str, required=True, help="Path to the lateral (LAT) X-ray .pt file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated 3D CT .pt file.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps.")
    # 生成する画像のサイズを指定
    parser.add_argument("--depth", type=int, default=256, help="Depth of the generated 3D image.")
    parser.add_argument("--height", type=int, default=256, help="Height of the generated 3D image.")
    parser.add_argument("--width", type=int, default=256, help="Width of the generated 3D image.")

    args = parser.parse_args()
    main(args)