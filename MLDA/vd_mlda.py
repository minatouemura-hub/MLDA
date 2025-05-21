from posterior_estim import cgibbs_sampling


class Vis_Doc_LDA:
    def __init__(
        self,
        K: int,
        V_v: int,
        V_w: int,
        alpha: float = 0.1,
        beta_v: float = 0.01,
        beta_w: float = 0.01,
        iteration: int = 1000,
    ):
        """
        画像と文書の2つのモーダルを持つLDA(collapsed gibbs sampling)

        Parameters:
        ----------
        V_v:
            対象の画像単語数(列ベクトル想定)
        V_w:
            画像のvocabllary数
        K : int
            トピック数(初期値10)
        alpha :float
            トピック割り当ての分布sitaのDirecllet分布のハイパーパラメータ
        beta : float
            単語分布phiのDirecllet分布のハイパーパラメータ
        iteration: int
            何回サンプリングを行うか
        """
        self.K = K
        self.V_v = V_v
        self.V_w = V_w
        self.alpha = alpha
        self.beta_v = beta_v
        self.beta_w = beta_w
        self.iterations = iteration

        self.z_n = None
        self.doc_counts = None
        self.topic_vis_count = None
        self.topic_word_count = None

    def fit(self, docs):
        self.z_n, self.doc_counts, self.topic_vis_count, self.topic_word_count = cgibbs_sampling(
            docs=docs,
            vis_voc=self.V_v,
            voc=self.V_w,
            k=self.K,
            alpha=self.alpha,
            beta_v=self.beta_v,
            beta_w=self.beta_w,
            iteration=self.iterations,
        )

    def estimate_theta(self):
        theta = (self.doc_counts + self.alpha) / (
            self.doc_counts.sum(axis=1, keepdims=True) + self.K * self.alpha
        )
        return theta

    def estimate_phi_v(self):
        phi_v = (self.topic_vis_count + self.beta_v) / (
            self.topic_vis_count.sum(axis=1, keepdims=True) + self.V_v * self.beta_v
        )
        return phi_v

    def estimate_phi_w(self):
        phi_w = (self.topic_word_count + self.beta_w) / (
            self.topic_word_count.sum(axis=1, keepdims=True) + self.V_w * self.beta_w
        )
        return phi_w

    def print_topics(self, id2word=None, topn=10):
        phi_w = self.estimate_phi_w()
        for k, topic_dist in enumerate(phi_w):
            top_words = topic_dist.argsort()[::-1][:topn]
            words = [id2word[w] if id2word else str(w) for w in top_words]
            print(f"Topic {k}: {' '.join(words)}")
