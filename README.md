# マルチモーダルLDA（MLDA）スクラッチ実装

このリポジトリは、**画像特徴とテキストキャプションを統合的に扱う LDA の拡張モデル（Multimodal LDA: mLDA）**のスクラッチ実装を提供します。  
学習には **Collapsed Gibbs Sampling** を採用しています。

---

## 📌 概要

Multimodal LDA は、以下の2つのモダリティ（データの種類）にまたがって潜在トピックを学習します：

- 🖼️ **視覚モダリティ**：画像から抽出された特徴（例：ResNetやBoVW）
- 📝 **言語モダリティ**：画像に付随するキャプション（テキスト）

各画像＋キャプションペアは、**複数のトピックの混合**として表現され、各トピックはそれぞれのモダリティにおいて「視覚語」と「単語」を生成するものと仮定されます。

> 特徴：
> - ✅ Pythonのみで実装（LDAライブラリ未使用）
> - ✅ Collapsed Gibbs Sampling による学習
> - ✅ 画像・テキストの前処理を含む
> - ✅ Perplexityによる評価、t-SNE可視化にも対応

---

## 🛠 必要環境
```bash
pip install -r requirements.txt
```

## 🚀 実行方法
1.	Flickr8k_Dataset/captions.txt と対応する画像を Flickr8k_Dataset/Images/ に配置
2.	以下のコマンドを実行：
```bash
python main.py
```
これにより： <br>
	•	100件の画像＋キャプションペアを読み込み
	•	ResNetにより画像をベクトル化<br>
	•	KMeansでBoVW視覚語に変換<br>
	•	キャプションを分かち書き・ID化<br>
	•	mLDAによる学習<br>
	•	トピックごとの単語出力とt-SNE可視化
## 📁 ディレクトリ構成

```bash

├── Flickr8k_Dataset/
    |-caption.txt# キャプションと画像ファイル
    |-utils.py
    |-__init__.py
├── MLDA/ 
    |-vd_mlda.py #mLDAクラスとサンプリングコード
    |-__init__.py                    
├── preprocessor/
    |-preprocess.py
    |-__init__.py               # 前処理：画像特徴抽出、テキスト分かち書きなど
├── main.py                     # メインスクリプト（学習〜可視化）
├── evaluation/
    |-eval.py #perplexity計算など
    |-__init__.py       
└──  README.md                  
