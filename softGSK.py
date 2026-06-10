import numpy as np

class SoftGSK:
    def __init__(self, P1, P2):
        self.P1 = np.array(P1, dtype=float)
        self.P2 = np.array(P2, dtype=float)
        self.s = len(self.P1)
        self.m_minus_s = len(self.P2)
        self.m = self.s + self.m_minus_s
        self.w = None
        self.beta = None
        self.u = None

    def distribution(self, projections, c, reverse=False):
        n_points = len(projections)
        h = np.zeros(n_points)
        if reverse:new_proj = np.argsort(projections)[::-1]
        else: new_proj = np.argsort(projections)
        full = 1.0
        for idx in new_proj:
            if full <= 1e-12: break
            give = min(c, full)
            h[idx] = give
            full -= give
        return h

    def fit(self, c, max_iter=1000, tol=1e-6):
        min_c = max(1.0 / self.s, 1.0 / self.m_minus_s)
        if c < min_c: raise ValueError(f'Параметр C должен быть >= {min_c:.4f}')
        u = np.zeros(self.m)
        u[:self.s] = 1.0 / self.s
        u[self.s:] = 1.0 / self.m_minus_s
        w = np.sum(self.P1 * u[:self.s, np.newaxis], axis=0) - np.sum(self.P2 * u[self.s:, np.newaxis], axis=0)
        for k in range(max_iter):
            wp1 = np.dot(self.P1, w)
            wp2 = np.dot(self.P2, w)
            h1 = self.distribution(wp1, c, reverse=False)
            h2 = self.distribution(wp2, c, reverse=True)
            h = np.concatenate([h1, h2])
            w_h1 = np.sum(self.P1 * h1[:, np.newaxis], axis=0)
            w_h2 = np.sum(self.P2 * h2[:, np.newaxis], axis=0)
            w_h = w_h1 - w_h2
            delta = np.dot(w, w) - np.dot(w, w_h)
            if delta < tol: break
            a = w - w_h
            a_sq = np.dot(a, a)
            if a_sq < 1e-12: break
            lambda_k = min(delta / a_sq, 1.0)
            u = (1 - lambda_k) * u + lambda_k * h
            w = (1 - lambda_k) * w + lambda_k * w_h
        w1 = np.sum(self.P1 * u[:self.s, np.newaxis], axis=0)
        w2 = np.sum(self.P2 * u[self.s:, np.newaxis], axis=0)
        self.w = w
        self.beta = 0.5 * (np.dot(w2, w2) - np.dot(w1, w1))
        self.u = u
        return self.w, self.beta
