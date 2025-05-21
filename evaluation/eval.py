import math

import numpy as np


def compute_perplexity(docs, lda_model):
    """
    docs: [(wv, ww), ...] 形式のリストのリスト（=文書）
    lda_model: Vis_Doc_LDA 学習済みインスタンス

    単語（ww）のみを対象としたPerplexityを返す
    """
    theta = lda_model.estimate_theta()  # (D, K)
    phi_w = lda_model.estimate_phi_w()  # (K, Vw)

    total_log_likelihood = 0.0
    total_words = 0

    for d, doc in enumerate(docs):
        for _, ww in doc:
            word_prob = np.dot(theta[d], phi_w[:, ww])  # ∑_k θ_dk * φ_kw
            if word_prob > 0:
                total_log_likelihood += math.log(word_prob)
            else:
                total_log_likelihood += -100  # log(ε) 的な回避
            total_words += 1

    perplexity = np.exp(-total_log_likelihood / total_words)
    return perplexity
