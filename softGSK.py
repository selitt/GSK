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

    def _greedy_distribution(self, projections, mu, reverse=False):
        n_points = len(projections)
        h = np.zeros(n_points)
        if reverse:
            sorted_indices = np.argsort(projections)[::-1]
        else:
            sorted_indices = np.argsort(projections)

        remaining_weight = 1.0

        for idx in sorted_indices:
            if remaining_weight <= 1e-12:
                break
            weight_to_give = min(mu, remaining_weight)
            h[idx] = weight_to_give
            remaining_weight -= weight_to_give

        return h

    def fit(self, mu, max_iter=1000, tol=1e-6):
        min_mu = max(1.0 / self.s, 1.0 / self.m_minus_s)
        if mu < min_mu: raise ValueError(f"Параметр mu слишком мал для данных множеств. Должно быть mu >= {min_mu:.4f}")
        u = np.zeros(self.m)
        u[:self.s] = 1.0 / self.s
        u[self.s:] = 1.0 / self.m_minus_s

        w = np.sum(self.P1 * u[:self.s, np.newaxis], axis=0) - \
            np.sum(self.P2 * u[self.s:, np.newaxis], axis=0)

        for k in range(max_iter):
            proj1 = np.dot(self.P1, w)
            proj2 = np.dot(self.P2, w)
            h1 = self._greedy_distribution(proj1, mu, reverse=False)
            h2 = self._greedy_distribution(proj2, mu, reverse=True)
            h = np.concatenate([h1, h2])
            w_h1 = np.sum(self.P1 * h1[:, np.newaxis], axis=0)
            w_h2 = np.sum(self.P2 * h2[:, np.newaxis], axis=0)
            w_h = w_h1 - w_h2
            w_sq = np.dot(w, w)
            delta = w_sq - np.dot(w, w_h)

            if delta < tol:
                print(f"Мягкий алгоритм сошелся на {k}-й итерации. Delta: {delta:.8f}")
                break
            dir_w = w - w_h
            dir_w_sq = np.dot(dir_w, dir_w)

            if dir_w_sq < 1e-12:
                break

            lambda_tilde = delta / dir_w_sq
            lambda_k = min(lambda_tilde, 1.0)
            u = (1 - lambda_k) * u + lambda_k * h
            w = (1 - lambda_k) * w + lambda_k * w_h
        w1_star = np.sum(self.P1 * u[:self.s, np.newaxis], axis=0)
        w2_star = np.sum(self.P2 * u[self.s:, np.newaxis], axis=0)

        w1_sq = np.dot(w1_star, w1_star)
        w2_sq = np.dot(w2_star, w2_star)

        self.w = w
        self.beta = 0.5 * (w2_sq - w1_sq)
        self.u = u

        return self.w, self.beta