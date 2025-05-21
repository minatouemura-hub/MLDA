import os  # noqa
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from evaluation import compute_perplexity
from Flickr8k_Dataset import load_100_data
from MLDA import Vis_Doc_LDA
from plot import plot_tsne_map
from preprocessor import doc2vec, image2vec_resnet


def main():
    BASE_DIR = Path(__file__).parent
    df = load_100_data("Flickr8k_Dataset/captions.txt")

    # 画像のベクトル化 + 視聴語の割り当て

    image_vecs = []
    img_paths = []
    img_name_to_vec = {}

    for img_name in df["image"]:
        img_path = BASE_DIR / "Flickr8k_Dataset" / "images" / img_name
        img_paths.append(img_path)
        vec = image2vec_resnet(img_path)  # 1回目のみ推論
        image_vecs.append(vec)
        img_name_to_vec[img_name] = vec  # キャッシュとして保持

    # KMeansクラスタリング
    all_vecs = np.vstack(image_vecs)
    V_v = 100
    kmeans = KMeans(n_clusters=V_v, random_state=0).fit(all_vecs)

    # クラスタ割り当て（視覚語ID）を取得
    img_to_vis_words = {}
    for fname in df["image"]:
        vec = img_name_to_vec[fname]  # 1回目の結果を再利用
        visual_id = kmeans.predict(vec)
        img_to_vis_words[fname] = visual_id.tolist()

    # 3. キャプション → 単語ID変換
    tokenized = [doc2vec(caption) for caption in df["caption"]]

    word2id = {}
    id2word = {}
    counter = 0
    for tokens in tokenized:
        for tok in tokens:
            if tok not in word2id:
                word2id[tok] = counter
                id2word[counter] = tok
                counter += 1
    V_w = len(word2id)

    # 4. docs 構築 [(wv, ww), ...]
    docs = []
    for fname, tokens in zip(df["image"], tokenized):
        wv = img_to_vis_words[fname][0]  # 画像につき1視覚語を使う（複数可に拡張可能）
        doc = [(wv, word2id[word]) for word in tokens if word in word2id]
        if doc:
            docs.append(doc)

    # 5. mLDA モデル学習
    lda = Vis_Doc_LDA(K=10, V_v=V_v, V_w=V_w, iteration=1000)
    lda.fit(docs)

    # 6. トピック表示
    lda.print_topics(id2word=id2word, topn=10)
    plot_tsne_map(lda_model=lda)
    perplexity = compute_perplexity(docs=docs, lda_model=lda)
    print(f"Perplexityは{perplexity:.3f}です")


if __name__ == "__main__":
    main()
