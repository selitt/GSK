import numpy as np

class commonGSK:
    def __init__(self, P1, P2):
        self.P1 = np.array(P1, dtype=float)
        self.P2 = np.array(P2, dtype=float)
        self.s = len(self.P1)
        self.m_minus_s = len(self.P2)
        self.m = self.s + self.m_minus_s
        self.w = None
        self.beta = None
        self.u = None

    def fit(self, max_iter=1000, tol=1e-6):
        u = np.zeros(self.m)
        u[:self.s] = 1.0 / self.s
        u[self.s:] = 1.0 / self.m_minus_s
        w = np.sum(self.P1 * u[:self.s, np.newaxis], axis=0) - np.sum(self.P2 * u[self.s:, np.newaxis], axis=0)
        for k in range(max_iter):
            i_first = np.argmin(np.dot(self.P1, w))
            i_second = np.argmax(np.dot(self.P2, w))
            p_first = self.P1[i_first]
            p_second = self.P2[i_second]
            delta = np.dot(w, w) - np.dot(w, p_first - p_second)
            if delta < tol: break
            a = w - p_first + p_second
            a_sq = np.dot(a, a)
            if a_sq < 1e-12: break
            lambda_k = min(delta / a_sq, 1.0)
            u = (1 - lambda_k) * u
            u[i_first] += lambda_k
            u[self.s + i_second] += lambda_k
            w = (1 - lambda_k) * w + lambda_k * (p_first - p_second)
        w1 = np.sum(self.P1 * u[:self.s, np.newaxis], axis=0)
        w2 = np.sum(self.P2 * u[self.s:, np.newaxis], axis=0)
        beta = 0.5 * (np.dot(w2, w2) - np.dot(w1, w1))
        self.w = w
        self.beta = beta
        self.u = u
        return self.w, self.beta
