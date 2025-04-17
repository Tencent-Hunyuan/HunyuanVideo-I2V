<!-- ## **HunyuanVideo** -->

[中文阅读](./README_zh.md)
[English Version](./README.md)

<p align="center">
  <img src="./assets/logo.png"  height=100>
</p>

# **HunyuanVideo-I2V** 🌅

<div align="center">
  <a href="https://github.com/Tencent/HunyuanVideo-I2V"><img src="https://img.shields.io/static/v1?label=HunyuanVideo-I2V コード&message=Github&color=blue"></a> &ensp;
  <a href="https://aivideo.hunyuan.tencent.com"><img src="https://img.shields.io/static/v1?label=プロジェクトページ&message=Web&color=green"></a> &ensp;
  <a href="https://video.hunyuan.tencent.com"><img src="https://img.shields.io/static/v1?label=プレイグラウンド&message=Web&color=green"></a>
</div>
<div align="center">
  <a href="https://arxiv.org/abs/2412.03603"><img src="https://img.shields.io/static/v1?label=技術レポート&message=Arxiv&color=red"></a> &ensp;
  <a href="https://aivideo.hunyuan.tencent.com/hunyuanvideo.pdf"><img src="https://img.shields.io/static/v1?label=技術レポート&message=高品質バージョン (~350M)&color=red"></a>
</div>
<div align="center">
  <a href="https://huggingface.co/tencent/HunyuanVideo-I2V"><img src="https://img.shields.io/static/v1?label=HunyuanVideo-I2V&message=HuggingFace&color=yellow"></a> &ensp;
</div>

<p align="center">
    👋 私たちの<a href="assets/WECHAT.md" target="_blank">WeChat</a>と<a href="https://discord.gg/tv7FkG4Nwf" target="_blank">Discord</a>に参加してください 
</p>

-----

私たちの[HunyuanVideo](https://github.com/Tencent/HunyuanVideo)の大成功のオープンソース化に続き、オープンソースコミュニティの探索を加速するための新しい画像からビデオ生成フレームワークである[HunyuanVideo-I2V](https://github.com/Tencent/HunyuanVideo-I2V)を誇らしげに発表します！

このリポジトリには、公式のPyTorchモデル定義、事前トレーニング済みの重み、および推論/サンプリングコードが含まれています。詳細なビジュアライゼーションは[プロジェクトページ](https://aivideo.hunyuan.tencent.com)でご覧いただけます。同時に、カスタマイズ可能な特殊効果のためのLoRAトレーニングコードも公開しました。これを使用して、より興味深いビデオ効果を作成できます。

> [**HunyuanVideo: A Systematic Framework For Large Video Generation Model**](https://arxiv.org/abs/2412.03603)

## 🔥🔥🔥 最新情報!!
* 2025年3月6日: 👋 HunyuanVideo-I2Vの推論コードとモデルの重みをリリースしました。[ダウンロード](https://github.com/Tencent/HunyuanVideo-I2V/blob/main/ckpts/README.md)。

## 🎥 デモ
### I2V デモ
<div align="center">
  <video src="https://github.com/user-attachments/assets/442afb73-3092-454f-bc46-02361c285930" width="80%" poster="./assets/video_poster.jpg"> </video>
  <p>共同制作者 @D-aiY 監督 丁一</p>
</div>

### カスタマイズ可能なI2V LoRAデモ

| I2V Lora 効果 |  参照画像 | 生成されたビデオ  |
|:---------------:|:--------------------------------:|:----------------:|
|   髪の成長   |        <img src="./assets/demo/i2v_lora/imgs/hair_growth.png" width="40%">         |       <video src="https://github.com/user-attachments/assets/06b998ae-bbde-4c1f-96cb-a25a9197d5cb" width="100%"> </video>        |
|     抱擁     |      <img src="./assets/demo/i2v_lora/imgs/embrace.png" width="40%">          |       <video src="https://github.com/user-attachments/assets/f8c99eb1-2a43-489a-ba02-6bd50a6dd260" width="100%" > </video>        |

## 📑 オープンソース計画
- HunyuanVideo-I2V（画像からビデオへのモデル）
  - [x] LoRAトレーニングスクリプト
  - [x] 推論コード
  - [x] モデルの重み
  - [x] ComfyUIサポート
  - [ ] マルチGPUシーケンス並列推論（複数のGPUでの推論速度の向上）
  - [ ] Diffusers統合 
  - [ ] FP8量子化重み

## 目次
- [**HunyuanVideo-I2V** 🌅](#hunyuanvideo-i2v-)
  - [🔥🔥🔥 最新情報!!](#-最新情報)
  - [🎥 デモ](#-デモ)
    - [I2V デモ](#i2v-デモ)
    - [カスタマイズ可能なI2V LoRAデモ](#カスタマイズ可能なi2v-loraデモ)
  - [📑 オープンソース計画](#-オープンソース計画)
  - [目次](#目次)
  - [**HunyuanVideo-I2V 全体アーキテクチャ**](#hunyuanvideo-i2v-全体アーキテクチャ)
  - [📜 要件](#-要件)
  - [🛠️ 依存関係とインストール](#️-依存関係とインストール)
    - [Linux用インストールガイド](#linux用インストールガイド)
  - [🧱 事前トレーニング済みモデルのダウンロード](#-事前トレーニング済みモデルのダウンロード)
  - [🔑 シングルGPU推論](#-シングルgpu推論)
    - [画像からビデオへのモデルの使用に関するヒント](#画像からビデオへのモデルの使用に関するヒント)
    - [コマンドラインの使用](#コマンドラインの使用)
    - [その他の設定](#その他の設定)
  - [🎉 カスタマイズ可能なI2V LoRA効果のトレーニング](#カスタマイズ可能なi2v-lora効果のトレーニング)
    - [要件](#要件)
    - [環境](#環境)
    - [トレーニングデータの構築](#トレーニングデータの構築)
    - [トレーニング](#トレーニング)
    - [推論](#推論)
  - [🔗 BibTeX](#-bibtex)
  - [謝辞](#謝辞)

---

## **HunyuanVideo-I2V 全体アーキテクチャ**
[HunyuanVideo](https://github.com/Tencent/HunyuanVideo)の高度なビデオ生成機能を活用して、画像からビデオへの生成タスクにその応用を拡張しました。これを実現するために、画像の潜在空間の連結技術を使用して、参照画像情報を効果的に再構築し、ビデオ生成プロセスに組み込みます。

私たちは、テキストエンコーダーとしてDecoder-Onlyアーキテクチャを持つ事前トレーニング済みのマルチモーダル大規模言語モデル（MLLM）を使用しているため、入力画像の意味内容を理解する能力を大幅に向上させ、画像とそ��関連キャプションの情報をシームレスに統合できます。具体的には、入力画像はMLLMによって処理され、意味的な画像トークンが生成されます。これらのトークンはビデオの潜在トークンと連結され、結合データ全体にわたる包括的なフルアテンション計算を可能にします。

私たちのシステムアーキテクチャは、画像とテキストのモダリティ間の相乗効果を最大化し、静止画像からのビデオコンテンツの堅牢で一貫性のある生成を確保するように設計されています。この統合により、生成されたビデオの忠実度が向上するだけでなく、複雑なマルチモーダル入力を解釈し利用するモデルの能力も向上します。全体のアーキテクチャは次のとおりです。
<p align="center">
  <img src="./assets/backbone.png"  height=300>
</p>

## 📜 要件

次の表は、HunyuanVideo-I2Vモデル（バッチサイズ=1）を実行してビデオを生成するための要件を示しています：

|      モデル       | 解像度  | GPUピークメモリ |
|:---------------:|:-------:|:-----------:|
| HunyuanVideo-I2V |  720p   |    60GB     |

* CUDAサポートを備えたNVIDIA GPUが必要です。
  * モデルは単一の80G GPUでテストされています。
  * **最小要件**: 720pの解像度には少なくとも60GBのGPUメモリが必要です。
  * **推奨構成**: より高い生成品質を得るために、80GBのメモリを持つGPUを使用することをお勧めします。
* テストされたオペレーティングシステム：Linux

## 🛠️ 依存関係とインストール

まず、リポジトリをクローンします：
```shell
git clone https://github.com/tencent/HunyuanVideo-I2V
cd HunyuanVideo-I2V
```

### Linux用インストールガイド

手動インストールには、CUDAバージョン12.4または11.8をお勧めします。

Condaのインストール手順は[こちら](https://docs.anaconda.com/free/miniconda/index.html)で確認できます。

```shell
# 1. Conda環境を作成
conda create -n HunyuanVideo-I2V python==3.11.9

# 2. 環境をアクティブにする
conda activate HunyuanVideo-I2V

# 3. Condaを使用してPyTorchおよびその他の依存関係をインストール
# CUDA 12.4の場合
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# 4. pip依存関係をインストール
python -m pip install -r requirements.txt

# 5. 高速化のためにflash attention v2をインストール（CUDA 11.8以上が必要）
python -m pip install ninja
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3
```

特定のGPUタイプで浮動小数点例外（コアダンプ）に遭遇した場合は、次の解決策を試してみてください：

```shell
# CUDA 12.4、CUBLAS>=12.4.5.8、およびCUDNN>=9.00をインストールしていることを確認します（または、単に私たちのCUDA 12 dockerイメージを使用します）。
pip install nvidia-cublas-cu12==12.4.5.8
export LD_LIBRARY_PATH=/opt/conda/lib/python3.8/site-packages/nvidia/cublas/lib/
```

さらに、HunyuanVideo-I2Vは事前構築されたDockerイメージも提供しています。次のコマンドを使用してdockerイメージをプルして実行します。

```shell
# CUDA 12.4の場合（浮動小数点例外を回避するために更新）
docker pull hunyuanvideo/hunyuanvideo-i2v:cuda_12
docker run -itd --gpus all --init --net=host --uts=host --ipc=host --name hunyuanvideo-i2v --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged hunyuanvideo/hunyuanvideo-i2v:cuda_12
```

## 🧱 事前トレーニング済みモデルのダウンロード

事前トレーニング済みモデルの詳細は[こちら](ckpts/README.md)に示されています。

## 🔑 シングルGPU推論

[HunyuanVideo](https://github.com/Tencent/HunyuanVideo)と同様に、HunyuanVideo-I2Vは高解像度のビデオ生成をサポートしており、解像度は最大720P、ビデオの長さは最大129フレーム（5秒）です。

### 画像からビデオへのモデルの使用に関するヒント
- **簡潔なプロンプトを使用する**: モデルの生成を効果的にガイドするために、プロンプトを短く簡潔に保ちます。
- **重要な要素を含める**: よく構造化されたプロンプトには次の要素が含まれているべきです：
  - **主題**: ビデオの主な焦点を指定します。
  - **アクション**: 発生している主要な動きや活動を説明します。
  - **背景（オプション）**: ビデオのシーンを設定します。
  - **カメラアングル（オプション）**: 視点や視点を示します。
- **過度に詳細なプロンプトを避ける**: 長すぎるまたは非常に詳細なプロンプトは、ビデオ出力に不要なトランジションを引き起こす可能性があります。

### コマンドラインの使用

```bash
cd HunyuanVideo-I2V

python3 sample_image2video.py \
    --model HYVideo-T/2 \
    --prompt "A man with short gray hair plays a red electric guitar." \
    --i2v-mode \
    --i2v-image-path ./assets/demo/i2v/imgs/0.png \
    --i2v-resolution 720p \
    --video-length 129 \
    --infer-steps 50 \
    --flow-reverse \
    --flow-shift 17.0 \
    --seed 0 \
    --use-cpu-offload \
    --save-path ./results 
```

### その他の設定

いくつかの便利な設定を以下に示します：

|        引数        |            デフォルト            |                          説明                          |
|:----------------------:|:-----------------------------:|:------------------------------------------------------------:|
|       `--prompt`       |             None              |           ビデオ生成のためのテキストプロンプト。               |
|       `--model`        |      HYVideo-T/2-cfgdistill   | ここではI2V用にHYVideo-T/2を使用し、T2VモードにはHYVideo-T/2-cfgdistillを使用します。 |
|     `--i2v-mode`       |            False              |                I2Vモードを有効にするかどうか。                      |
|  `--i2v-image-path`    | ./assets/demo/i2v/imgs/0.png  |        ビデオ生成のための参照画像。              |
|  `--i2v-resolution`    |            720p               |        生成されるビデオの解像度。                |
|    `--video-length`    |             129               |         生成されるビデオの長さ。                    |
|    `--infer-steps`     |              50               |         サンプリングのステップ数。                     |
|     `--flow-shift`     |             7.0               |     フローマッチングスケジューラのシフトファクター。               |
|   `--flow-reverse`     |            False              | 逆にする場合、t=1からt=0への学習/サンプリング。                |
|        `--seed`        |             None              | ビデオ生成のためのランダムシード。Noneの場合、ランダムシードを初期化します。 |
|  `--use-cpu-offload`   |            False              | モデルのロードにCPUオフロードを使用してメモリを節約します。高解像度ビデオ生成には必要です。 |
|     `--save-path`      |         ./results             |         生成されたビデオを保存するパス。                     |

## 🎉 カスタマイズ可能なI2V LoRA効果のトレーニング

### 要件

次の表は、HunyuanVideo-I2V loraモデル（バッチサイズ=1）をトレーニングしてビデオを生成するための要件を示しています：

|      モデル       | 解像度 | GPUピークメモリ |
|:----------------:|:----------:|:---------------:|
| HunyuanVideo-I2V |    360p    |      79GB       |

* CUDAサポートを備えたNVIDIA GPUが必要です。
  * モデルは単一の80G GPUでテストされています。
  * **最小要件**: 360pの解像度には少なくとも79GBのGPUメモリが必要です。
  * **推奨構成**: より高い生成品質を得るために、80GBのメモリを持つGPUを使用することをお勧めします。
* テストされたオペレーティングシステム：Linux
* 注意: 360pデータでトレーニングし、720pビデオを直接推論することができます

### 環境
```
pip install -r requirements.txt
```

### トレーニングデータの構築
プロンプトの説明: トリガーワードはビデオキャプションに直接書かれます。フレーズや短い文を使用することをお勧めします。

例: AI髪の成長効果（トリガー）: rapid_hair_growth, The hair of the characters in the video is growing rapidly. + 元のプロンプト

トレーニングビデオとプロンプトのペアを用意した後、トレーニングデータの構築については[こちら](hyvideo/hyvae_extract/README.md)を参照してください。

### トレーニング
```
sh scripts/run_train_image2video_lora.sh
```
いくつかのトレーニング特有の設定を以下に示します：

|     引数     |                            デフォルト                            |                         説明                         |
|:----------------:|:-------------------------------------------------------------:|:-----------------------------------------------------------:|
|   `SAVE_BASE`    |                               .                               |         実験結果を保存するためのルートパス。          |
|    `EXP_NAME`    |                           i2v_lora                            |        実験結果を保存するためのパスのサフィックス。         |
| `DATA_JSONS_DIR` | ./assets/demo/i2v_lora/train_dataset/processed_data/json_path | hyvideo/hyvae_extract/start.shによって生成されたデータjsonsディレクトリ。 |
|    `CHIEF_IP`    |                           127.0.0.1                           |            マシンのマスターノードIP。                   |

トレーニング後、`pytorch_lora_kohaya_weights.safetensors`は`{SAVE_BASE}/log_EXP/*_{EXP_NAME}/checkpoints/global_step{*}/pytorch_lora_kohaya_weights.safetensors`に保存され、`--lora-path`に設定して推論を実行できます。

### 推論
```bash
python3 sample_image2video.py \
    --model HYVideo-T/2 \
    --prompt "Two people hugged tightly, In the video, two people are standing apart from each other. They then move closer to each other and begin to hug tightly. The hug is very affectionate, with the two people holding each other tightly and looking into each other's eyes. The interaction is very emotional and heartwarming, with the two people expressing their love and affection for each other." \
    --i2v-mode \
    --i2v-image-path ./assets/demo/i2v_lora/imgs/embrace.png \
    --i2v-resolution 720p \
    --infer-steps 50 \
    --video-length 129 \
    --flow-reverse \
    --flow-shift 5.0 \
    --seed 0 \
    --use-cpu-offload \
    --save-path ./results \
    --use-lora \
    --lora-scale 1.0 \
    --lora-path ./ckpts/hunyuan-video-i2v-720p/lora/embrace_kohaya_weights.safetensors
```
いくつかのlora特有の設定を以下に示します：

|      引数       | デフォルト |         説明          |
|:-------------------:|:-------:|:----------------------------:|
|    `--use-lora`     |  False  |  loraモードを有効にするかどうか。  |
|   `--lora-scale`    |   1.0   | loraモデルの融合スケール。 |
|   `--lora-path`     |   ""    |  loraモデルの重みのパス。 |

## 🔗 BibTeX

[HunyuanVideo](https://arxiv.org/abs/2412.03603)が研究やアプリケーションに役立つ場合は、次のBibTeXを使用して引用してください：

```BibTeX
@misc{kong2024hunyuanvideo,
      title={HunyuanVideo: A Systematic Framework For Large Video Generative Models}, 
      author={Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Junkun Yuan, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yanxin Long, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, and Jie Jiang, along with Caesar Zhong},
      year={2024},
      archivePrefix={arXiv preprint arXiv:2412.03603},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.03603}, 
}
```

## 謝辞

[SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Llama](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [Xtuner](https://github.com/InternLM/xtuner), [diffusers](https://github.com/huggingface/diffusers)および[HuggingFace](https://huggingface.co)リポジトリの貢献者に感謝します。さらに、テキストエンコーダーのサポートに対してTencent Hunyuan Multimodalチームにも感謝します。
