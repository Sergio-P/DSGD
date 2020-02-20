from sklearn.base import ClassifierMixin

from dsgd.core import compute_gradient_dempster_rule
from dsgd.utils import normalize


class DSClassifierGDH(ClassifierMixin):

    def __init__(self, alpha=0.01, gamma=0.01, batch_size=20, min_delta_J=5e-4, learning_rate="constant",
                 max_iter=200, certain_J=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_delta_J = min_delta_J
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.certain_J = certain_J

    def fit(self, X, y):
        self._optimize()

    def _optimize(self):
        last_J = 1.
        J = 0.
        i = 0
        while abs(last_J - J) > self.min_delta_J and i < self.max_iter:
            last_J = J
            J = 0.
            for i in range(self.batch_size):
                # Computing
                px = (ax + bx - ax * bx - ax * by - ay * bx) / (
                ax + bx + ay + by - ax * bx - ay * by - 2 * ax * by - 2 * ay * bx)
                py = (ay + by - ay * by - ax * by - ay * bx) / (
                ax + bx + ay + by - ax * bx - ay * by - 2 * ax * by - 2 * ay * bx)
                y = [px, py]
                y_cup = data[i]
                rx, ry = y_cup
                # Optimization
                dJdax, dJday, dJdbx, dJdby = compute_gradient_dempster_rule(ax, ay, bx, by, rx, ry)
                # Updating
                ax -= self.alpha * dJdax
                ay -= self.alpha * dJday
                bx -= self.alpha * dJdbx
                by -= self.alpha * dJdby
                Jn = .5 * ((y[0] - y_cup[0]) ** 2 + (y[1] - y_cup[1]) ** 2)
                # ae = 1 - ax - ay
                # be = 1 - bx - be
                ae += self.gamma * (Jn - self.certain_J)
                ax, ay, ae = normalize(ax, ay, ae)
                be += self.gamma * (Jn - self.certain_J)
                bx, by, be = normalize(bx, by, be)

                J += Jn / self.batch_size
            # Js.append(J)
            # Debugging
            # print_maf()
            i += 1
            self.alpha *= .95
