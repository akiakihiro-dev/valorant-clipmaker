# valorant-clipmaker

Valorantのプレイ録画（mp4）から、自分がキルしたシーンだけを自動で検出し、
1本のハイライトクリップとして書き出すツールです。

NVIDIA ShadowPlay等の「インスタントリプレイ」機能で保存済みの録画クリップを対象に、
画面上のキルフィード表示を画像処理で解析してキル区間を検出します（OCRや外部APIは
使用せず、完全にローカルで動作します）。詳しい設計方針は [plans/plan1.md](plans/plan1.md)
を参照してください。

## 現状のステータス

キル検出・区間決定・クリップ書き出しまでの一連のパイプラインが動作するPoC（概念実証）
段階です。以下の制約があります。

- キルフィードのROI（検出対象領域）はフレーム比率で持っていますが、デフォルト値は
  1920x1080の録画を前提に調整したものです。異なる解像度・アスペクト比・HUDスケール
  設定では `src/kill_detector.py` の `ROIConfig` を調整する必要があります。
- 自分のキル判定は、自分のプレイヤー名をキルフィード内でテンプレートマッチングする
  方式です。**利用する際は `assets/templates/own_name.png` に、自分のプレイヤー名を
  キルフィードから切り出した画像を自分で用意して配置してください**（個人が特定できる
  情報のためリポジトリには含めておらず、`.gitignore`対象です。配置しないと動作しません）。

## 必要環境

- Python 3.11以上
- [ffmpeg](https://ffmpeg.org/)（コマンドラインから `ffmpeg` を実行できること。区間の
  切り出し・結合に使用します）

## セットアップ

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 使い方

1. 自分のプレイヤー名をキルフィードから切り出した画像を用意し、
   `assets/templates/own_name.png` として配置する（切り出しのコツは
   [assets/templates/README.md](assets/templates/README.md)を参照）。
2. 処理したい録画クリップ（`.mp4`）を `clipsample/` ディレクトリに置く。
3. 実行する。

   ```bash
   python main.py
   ```

4. `clipsample/` 内の各動画について自分のキル区間が検出され、
   `output/{動画名}_highlight.mp4` としてハイライトクリップが出力される。
   キルが検出されなかった動画はスキップされる。

## ディレクトリ構成

```
valorant-clipmaker/
├── assets/templates/    # キル検出用のテンプレート画像（own_name.pngは各自配置、gitignore対象）
├── clipsample/          # 入力動画を置く場所（gitignore対象）
├── output/              # 出力先（gitignore対象）
├── plans/               # 設計ドキュメント
├── src/
│   ├── kill_detector.py       # フレーム抽出・ROI切り出し・キル検出
│   ├── highlight_detector.py  # キル区間からハイライト切り出し範囲を決定
│   └── clipper.py             # ffmpegによるクリップ切り出し・結合
├── tests/               # 単体テスト
└── main.py              # パイプライン実行エントリーポイント
```

## テスト

外部の動画ファイルを必要としない範囲のロジックについて単体テストがあります。

```bash
python -m unittest discover -s tests
```

## ライセンス

[MIT License](LICENSE)
