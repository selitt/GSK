import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons


class Gradient:
    def __init__(self, P1, P2):
        self.P1 = np.array(P1, dtype=float)
        self.P2 = np.array(P2, dtype=float)
        self.s = len(self.P1)
        self.m_minus_s = len(self.P2)
        self.m = self.s + self.m_minus_s
        self.w = None
        self.beta = None
        self.u = None

    def fit_strict_gsk(self, max_iter=1000, tol=1e-6):
        u = np.zeros(self.m)
        u[:self.s] = 1.0 / self.s
        u[self.s:] = 1.0 / self.m_minus_s

        w = np.sum(self.P1 * u[:self.s, np.newaxis], axis=0) - np.sum(self.P2 * u[self.s:, np.newaxis], axis=0)

        for k in range(max_iter):
            proj1 = np.dot(self.P1, w)
            proj2 = np.dot(self.P2, w)
            i_prime = np.argmin(proj1)
            i_double_prime = np.argmax(proj2)

            p_prime = self.P1[i_prime]
            p_double_prime = self.P2[i_double_prime]

            w_sq = np.dot(w, w)
            delta = w_sq - np.dot(w, p_prime - p_double_prime)

            if delta < tol:
                break

            dir_w = w - p_prime + p_double_prime
            dir_w_sq = np.dot(dir_w, dir_w)

            if dir_w_sq < 1e-12:
                break

            lambda_tilde = delta / dir_w_sq
            lambda_k = min(lambda_tilde, 1.0)

            u = (1 - lambda_k) * u
            u[i_prime] += lambda_k
            u[self.s + i_double_prime] += lambda_k

            w = (1 - lambda_k) * w + lambda_k * (p_prime - p_double_prime)
        w1_star = np.sum(self.P1 * u[:self.s, np.newaxis], axis=0)
        w2_star = np.sum(self.P2 * u[self.s:, np.newaxis], axis=0)

        w1_sq = np.dot(w1_star, w1_star)
        w2_sq = np.dot(w2_star, w2_star)
        beta = 0.5 * (w2_sq - w1_sq)

        self.w = w
        self.beta = beta
        self.u = u

        return self.w, self.beta
