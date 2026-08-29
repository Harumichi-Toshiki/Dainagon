# Dainagon AI 学習基盤 (ROCm / PyTorch)

将棋AI **Dainagon (ChunagonNetV3)** のディープラーニングモデルを学習するためのトレーニング基盤です。
AMD ROCm (HIP) / GPU 環境において並列学習を行います。

---

## 🛠️ 環境要件

- **Python**: 3.10 以上
- **PyTorch**: 2.4.0 以上 (ROCm 7.x 以上対応版)
- **主要依存パッケージ**: `cshogi`, `numpy`, `tqdm`, `matplotlib`

```bash
pip install -r requirements.txt
```

---

## 📁 ディレクトリ構造

```text
train/
├── network/                        # ネットワークアーキテクチャ定義 (v25, v26, v27, v28)
│   ├── network_chunagon_v28.py               # 公開用メインモデル
│   └── ...
├── features.py                     # 55ch 局面特徴量抽出エンジン
├── train_rocm.py                   # ROCm メイン学習スクリプト (train.py と同一)
├── run_experiment.py               # 壁時計時間ベースの実験ランナー
├── exportonnx.py                   # ONNX 形式へのエクスポート
├── exporttorchscript.py            # TorchScript 形式へのエクスポート
├── make_swa.py                     # SWA (Stochastic Weight Averaging)
├── requirements.txt                # 依存パッケージ定義
├── .gitignore                      # Git管理対象外設定
└── README.md                       # 本ドキュメント
```

---

## 🚀 使い方

### 1. 教師データの配置

教師データ (`*.hcpe` 形式) を以下のディレクトリに配置してください。
- 訓練用: `./data/traindata/`
- 検証用: `./data/train_test/`

※任意のパスを指定したい場合は、コマンドライン引数 `--data-dir` / `--val-dir` で指定可能です。

### 2. 学習の実行

#### 通常学習
```bash
python train_rocm.py
```
または
```bash
python train_rocm.py --data-dir ./path/to/traindata --val-dir ./path/to/valdata
```

#### 時間指定での実験実行
```bash
python run_experiment.py --test-id test1 --max-hours 6
```
