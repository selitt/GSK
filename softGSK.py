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

    def distribution(self, projections, mu, reverse = False):
        n = len(projections)
        h = np.zeros(n)
        if reverse: new_proj = np.argsort(projections)[::-1]
        else: new_proj = np.argsort(projections)
        full = 1.0
        for i in new_proj:
            if full == 0: break
            u_i = min(mu, full)
            h[i] = u_i
            full -= u_i
        return h

    def solve(self, mu, max_iter = 1000, tol = 1e-6):
        min_mu = max(1.0 / self.s, 1.0 / self.m_minus_s)
        if mu < min_mu: raise ValueError(f'Штрафной параметр должен быть >= {min_mu:.4f}')
        u = np.zeros(self.m)
        u[:self.s] = 1.0 / self.s
        u[self.s:] = 1.0 / self.m_minus_s
        w = np.sum(self.P1 * u[:self.s, np.newaxis], axis = 0) - np.sum(self.P2 * u[self.s:, np.newaxis], axis = 0)
        for k in range(max_iter):
            wp1 = np.dot(self.P1, w)
            wp2 = np.dot(self.P2, w)
            h1 = self.distribution(wp1, mu, reverse= False)
            h2 = self.distribution(wp2, mu, reverse = True)
            h = np.concatenate([h1, h2])
            w_h1 = np.sum(self.P1 * h1[:, np.newaxis], axis= 0)
            w_h2 = np.sum(self.P2 * h2[:, np.newaxis], axis=0)
            w_h = w_h1 - w_h2
            delta = np.dot(w, w) - np.dot(w, w_h)
            if delta < tol: break
            znam = w - w_h
            znam_sq = np.dot(znam, znam)
            if znam_sq < 1e-12: break
            lamb = min(delta / znam_sq, 1.0)
            u = (1 - lamb) * u + lamb * h
            w = (1 - lamb) * w + lamb * w_h
        w1 = np.sum(self.P1 * u[:self.s, np.newaxis], axis = 0)
        w2 = np.sum(self.P2 * u[self.s:, np.newaxis], axis = 0)
        self.w = w
        self.beta = 0.5 * (np.dot(w2, w2) - np.dot(w1, w1))
        self.u = u
        return self.w, self.beta
