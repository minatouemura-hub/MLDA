import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne_map(lda_model):
    """
    学習済みmLDAモデルの文書トピック分布（θ）をt-SNEで2次元に圧縮し、プロット。

    ラベルや軸は表示しない簡潔なスタイル。
    """
    theta = lda_model.estimate_theta()  # shape: (n_docs, n_topics)

    tsne = TSNE(n_components=2, random_state=0, perplexity=15, n_iter=1000)
    theta_2d = tsne.fit_transform(theta)

    plt.figure(figsize=(8, 6))
    plt.scatter(theta_2d[:, 0], theta_2d[:, 1], alpha=0.7, s=30)
    plt.xticks([])  # x軸非表示
    plt.yticks([])  # y軸非表示

    plt.title("t-SNE of Images in Topic Space", fontsize=14)
    plt.tight_layout()
    plt.savefig("MLDA_theta_plot")
