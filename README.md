# 3D Conditional Diffusion Model for CT Generation (PyTorch Lightning)

## 1. 概要 (Overview)

このプロジェクトは、2枚の2D X線画像（正面: AP, 側面: LAT）を条件として、高解像度の3D CT画像 (256x256x256) を生成する条件付き3D Diffusion Modelを実装します。

このバージョンは **PyTorch Lightning** をフレームワークとして使用しており、ボイラープレートコードを削減し、可読性と再現性を高めています。Hugging Faceの `diffusers` ライブラリをモデルアーキテクチャの基盤としつつ、PyTorch Lightningの `Trainer` が分散学習、混合精度、チェックポイント管理などをすべて引き受けます。

**主な特徴:**
- **Conditional Generation:** 2D X線画像ペアから3D CTを生成。
- **High Resolution:** 256x256x256のボクセル解像度に対応。
- **PyTorch Lightning Framework:** クリーンで構造化されたコード。
- **Multi-GPU Model Parallelism:** PyTorch Lightningの `Trainer` を通じて、Fully Sharded Data Parallelism (FSDP) を簡単に設定可能。
- **Memory Efficiency:** 勾配チェックポインティングと混合精度学習（`bf16`）をサポートし、メモリ要求の厳しい大規模モデルの学習を実現。

## 2. セットアップ (Setup)

### 2.1. 依存関係のインストール
まず、リポジトリをクローンし、必要なPythonライブラリをインストールします。

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```
PyTorchは、ご自身のCUDAバージョンに合ったものをインストールしてください。
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### 2.2. データセットの準備
このモデルは、特定のディレクトリ構造を持つデータセットを想定しています。

- **入力 (X線画像):**
  - `<base_dir>/<patient_id>/AP.pt`
  - `<base_dir>/<patient_id>/LAT.pt`
- **ターゲット (CT画像):**
  - `<base_dir2>/<patient_id>.pt`

各 `.pt` ファイルは、単一の `torch.Tensor` を含む必要があります。

### 2.3. 設定ファイルの編集
`configs/config.yaml` を開き、ご自身の環境に合わせてパスや設定を更新します。

- **`trainer` セクション:**
  - `default_root_dir`: モデルのチェックポイントやログを保存するディレクトリ。
  - `devices`: 使用するGPUの数（例: `3`）。
  - `strategy`: モデル並列学習には `"fsdp_native"` を指定します。
  - `precision`: Ampere世代以降のGPU（RTX 30x0, 40x0, A100, 6000 Ada）では `"bf16-mixed"` を推奨します。

- **`data` セクション:**
  - `base_dir`: 入力X線画像のベースディレクトリパス。
  - `base_dir2`: ターゲットCT画像のベースディレクトリパス。

## 3. 学習の実行 (Training)

設定が完了したら、以下のコマンドで学習スクリプトを実行します。PyTorch Lightningが設定ファイルに基づいて分散学習を自動的に処理します。

```bash
python train.py --config configs/config.yaml
```

学習の進捗はTensorBoardで確認できます。

```bash
tensorboard --logdir ./output_pl/logs
```
（`trainer.default_root_dir`で指定したパスに合わせてください）

## 4. 推論の実行 (Inference)

学習が完了すると、`default_root_dir` にチェックポイントファイル (`.ckpt`) が保存されます。このファイルを使用して、新しいX線画像ペアから3D CT画像を生成できます。

```bash
python inference.py \
    --checkpoint_path ./output_pl/best-model-....ckpt \
    --ap_path /path/to/new/AP.pt \
    --lat_path /path/to/new/LAT.pt \
    --output_path ./generated_ct.pt \
    --num_inference_steps 50
```

- `--checkpoint_path`: 使用する学習済みモデルの `.ckpt` ファイルを指定します。
- `--ap_path`, `--lat_path`: 入力となるX線画像のパス。
- `--output_path`: 生成されたCT画像を保存するファイルパス。
- `--num_inference_steps`: 推論時のノイズ除去ステップ数。大きいほど高品質になる可能性がありますが、時間もかかります。
- `--depth`, `--height`, `--width`: 生成する画像のサイズを指定します（デフォルトは256）。