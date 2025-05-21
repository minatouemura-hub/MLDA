import numpy as np
from tqdm import tqdm


def cgibbs_sampling(
    docs,
    voc,
    vis_voc,
    k: int = 10,
    alpha: float = 0.1,
    beta_v: float = 0.01,
    beta_w: float = 0.01,
    iteration: int = 1000,
    burn_in: int = 200,
):
    """
    collapsed gibbs sampling

    Parameters:
    ----------
    docs:
        対象のマルチモーダルデータ集合[[文書], [画像]]
    viss:
        対象の画像集合(列ベクトル想定)
    vis_voc:
        画像のvocabllary数
    k : int
        トピック数(初期値10)
    alpha :float
        トピック割り当ての分布sitaのDirecllet分布のハイパーパラメータ
    beta : float
        単語分布phiのDirecllet分布のハイパーパラメータ
    iteration: int
        何回サンプリングを行うか
    burn_in :int
        バーンイン期間(必要なら処理を追加<=バーンイン後の平均やパラメータ蓄積を行うなら)
    """
    D = len(docs)
    # n_{d,k}
    doc_counts = np.zeros((D, k), dtype=int)
    # n_{k,v}
    topic_word_counts = np.zeros((k, voc), dtype=int)
    topic_vis_counts = np.zeros((k, vis_voc), dtype=int)

    topic_vis_total = np.zeros(k, dtype=int)
    topic_words_total = np.zeros(k, dtype=int)

    z_dn = []  # トピック割り当て

    # 初期割り当て
    for d, doc in enumerate(docs):
        z_d = []
        for wv, ww in doc:
            z = np.random.randint(k)
            z_d.append(z)

            doc_counts[d, z] += 1
            topic_vis_counts[z, wv] += 1
            topic_word_counts[z, ww] += 1
            topic_vis_total[z] += 1
            topic_words_total[z] += 1
        z_dn.append(z_d)

    for it in tqdm(range(iteration), desc="Sampling"):
        for d, doc in enumerate(docs):
            for n, (wv, ww) in enumerate(doc):
                z_old = z_dn[d][n]  # docのn番目単語がなんのtopicか
                # -iの計算
                doc_counts[d, z_old] -= 1
                topic_vis_counts[z_old, wv] -= 1
                topic_word_counts[z_old, ww] -= 1
                topic_vis_total[z_old] -= 1
                topic_words_total[z_old] -= 1

                # Collapsed Gibbsの事後分布
                p_z = (
                    (doc_counts[d] + alpha)
                    * (topic_vis_counts[:, wv] + beta_v)
                    / (topic_vis_total + beta_v * vis_voc)
                    * (topic_word_counts[:, ww] + beta_w)
                    / (topic_words_total + beta_w * voc)
                )

                if p_z.sum() != 0:
                    p_z /= p_z.sum()
                else:
                    raise ValueError("Sum of p_z got 0")
                z_new = np.random.choice(k, p=p_z)

                # Update counts
                z_dn[d][n] = z_new
                doc_counts[d, z_new] += 1
                topic_vis_counts[z_new, wv] += 1
                topic_word_counts[z_new, ww] += 1
                topic_vis_total[z_new] += 1
                topic_words_total[z_new] += 1
    return z_dn, doc_counts, topic_vis_counts, topic_word_counts
