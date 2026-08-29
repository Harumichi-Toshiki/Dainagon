# Dainagon AI 学習基盤 (ROCm / DirectML / CUDA 対応)

将棋AI **Dainagon (ChunagonNetV3)** のディープラーニングモデルを学習するための基盤コードです。
環境に応じて **AMD ROCm (HIP)**、**Windows DirectML**、**NVIDIA CUDA**、**CPU** を自動判定してアクセラレーションを適用します。

---

## 💻 対応ハードウェア・バックエンド

| バックエンド | 対象環境 | 状態 / 備考 |
| :--- | :--- | :--- |
| **ROCm (HIP)** | Linux / Windows (AMD Radeon RX 6000/7000/9000シリーズ) | **動作確認済み(RX9070XT)** (AMD GPU推奨) |
| **DirectML** | Windows (AMD / Intel / NVIDIA GPU) | **動作確認済み** (Windows手軽設定) |
| **CUDA** | Linux / Windows (NVIDIA GeForce/RTXシリーズ) | **未確認** (理論上動作可能) |
| **CPU** | 汎用環境 | **未確認** (フォールバック用) |

> ⚠️ **注意**: 本リポジトリは AMD ROCm および Windows DirectML 環境での動作確認を行っています。NVIDIA CUDA 環境および CPU 環境での学習・動作は未確認です。

### ⚙️ 実機動作検証済み環境

| 項目 | ROCm 推奨環境 | DirectML 環境 |
| :--- | :--- | :--- |
| **OS** | Windows 11 / Linux | Windows 10 / 11 |
| **GPU** | AMD Radeon RX 9070 XT (RDNA4) / RX 6000・7000シリーズ | DirectX 12 対応 GPU (AMD / Intel / NVIDIA) |
| **Python** | 3.10 / 3.12 | 3.10 / 3.12 |
| **PyTorch** | `2.9.1+rocm7.2.1` | `2.4.1` + `torch-directml 0.2.5` |
| **cshogi** | `0.9.7` | `0.9.7` |
| **実行スクリプト** | `python train_rocm.py` | `python train.py` |

---

## 🛠️ 環境構築手順

### 方式A: Windows + DirectML 環境 (AMD/Intel/NVIDIA GPU)

Windows環境で手軽にGPU学習を行いたい場合に推奨されます。

```bash
# 1. 基本パッケージのインストール
pip install torch torchvision

# 2. DirectML プラグインのインストール
pip install torch-directml

# 3. 将棋エンジン・補助ライブラリのインストール
pip install cshogi numpy tqdm matplotlib
```

---

### 方式B: AMD ROCm (Native HIP) 環境 (AMD Radeon GPU)

AMD Radeon GPUで最大の学習速度を出したい場合に推奨されます。

```bash
# 1. ROCm対応版 PyTorch のインストール (例: ROCm 6.1 / 7.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1

# 2. 将棋エンジン・補助ライブラリのインストール
pip install cshogi numpy tqdm matplotlib
```

※ `requirements.txt` から一括インストールすることも可能です:
```bash
pip install -r requirements.txt
```

---

## 📁 ディレクトリ構造

```text
train/
├── network/                        # ネットワークモデル定義 (v25, v26, v27, v28)
│   ├── network_chunagon_v28.py               # メインモデル
│   └── ...
├── data/                           # 教師データ格納用 (Git管理対象外)
│   ├── traindata/                  # 訓練データ (*.hcpe)
│   └── train_test/                 # 検証データ (*.hcpe)
├── features.py                     # 55ch 局面特徴量抽出エンジン
├── train.py                        # メイン学習スクリプト (ROCm / DirectML 自動判定)
├── train_rocm.py                   # 学習スクリプト (同上)
├── run_experiment.py               # 壁時計時間ベースの実験ランナー
├── exportonnx.py                   # ONNX 形式へのエクスポート
├── exporttorchscript.py            # TorchScript 形式へのエクスポート
├── make_swa.py                     # SWA (Stochastic Weight Averaging)
├── requirements.txt                # 依存パッケージ一覧
├── .gitignore                      # Git管理対象外設定
└── README.md                       # 本ドキュメント
```

---

## 🚀 使い方

### 1. 教師データの配置

将棋の教師データ (`*.hcpe` 形式) を以下のディレクトリに配置します。

- **訓練用データ**: `./data/traindata/` (例: `0.hcpe`, `1.hcpe` ...)
- **検証用データ**: `./data/train_test/` (例: `test.hcpe`)

※独自のパスにあるデータを使用する場合は、コマンドライン引数 `--data-dir` / `--val-dir` または環境変数 `DAINAGON_DATA_DIR` / `DAINAGON_VAL_DIR` を指定してください。

---

### 2. 学習の実行

#### 通常学習 (自動バックエンド選択)
```bash
python train.py
```
実行時に自動的に利用可能なデバイスが検出・表示されます：
- `[INFO] DEVICE = cuda` (ROCm または CUDA)
- `[INFO] DEVICE = privateuseone:0` (DirectML)

#### パスを明示して実行する場合
```bash
python train.py --data-dir "C:/path/to/traindata" --val-dir "C:/path/to/valdata"
```

#### 時間指定・実験管理ランナー (`run_experiment.py`)
```bash
python run_experiment.py --test-id exp_01 --max-hours 6
```
- 指定した時間（例: 6時間）で自動的に評価ログ (`model/logs/exp_01.jsonl`) とチェックポイント (`model/checkpoints/exp_01.pt`) を出力して終了します。

---

### 3. モデルのエクスポート (ONNX / TorchScript)

学習したモデル (`model/chunagon_v30.pt` 等) を推論エンジン向けにエクスポートできます。

#### ONNX エクスポート (FP16 / 半精度)
```bash
python exportonnx.py --name chunagon_v28 --fp16
```

#### TorchScript エクスポート
```bash
python exporttorchscript.py --name chunagon_v28
```

---

## ❓ トラブルシューティング & FAQ

#### Q. DirectML でメモリ不足 (OOM) になる
`train.py` 内の `PHYSICAL_BATCH_SIZE` (デフォルト `256`) を `128` や `64` に下げて実行してください。勾配累積 (`ACCUM_STEPS`) により実質的なバッチサイズ `TARGET_BATCH_SIZE` (2048) は維持されます。

#### Q. ROCm と DirectML の速度差は？
AMD Radeon GPU環境では、ROCm (hipBLAS + TunableOp) を使用することで DirectML 比で約 1.1〜1.3 倍程度の高速化が期待できます。環境構築の容易さを重視する場合は DirectML、速度を追求する場合は ROCm をご使用ください。
