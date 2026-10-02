import numpy as np


class LogisticRegression:
    """Multinomial (softmax) logistic regression written from scratch.

    Extras compared with the class notebook:
      * optional Ridge (L2) penalty        -> use_penalty / lambda_
      * metrics implemented by hand        -> accuracy, precision, recall, f1,
                                              macro_* and weighted_* averages
    The bias (intercept) is added inside the class and is NOT penalised.
    """

    def __init__(self, k, alpha=0.01, max_iter=1000, method="batch",
                 batch_size=256, use_penalty=False, lambda_=0.01,
                 tol=1e-7, random_state=42):
        self.k = k                        # number of classes
        self.alpha = alpha                # learning rate
        self.max_iter = max_iter          # number of epochs
        self.method = method              # "batch" | "minibatch" | "sgd"
        self.batch_size = batch_size
        self.use_penalty = use_penalty    # turn the L2 penalty on / off
        self.lambda_ = lambda_            # strength of the L2 penalty
        self.tol = tol                    # stop early when the loss barely moves
        self.random_state = random_state
        self.W = None
        self.n_features_ = None
        self.loss_history = []

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)       # subtract max for stability
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def _add_bias(X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _one_hot(self, y):
        Y = np.zeros((len(y), self.k))
        Y[np.arange(len(y)), y] = 1
        return Y

    def _loss(self, Xb, Y):
        h = self._softmax(Xb @ self.W)
        loss = -np.sum(Y * np.log(h + 1e-15)) / Xb.shape[0]   # mean cross-entropy
        if self.use_penalty:
            loss += self.lambda_ * np.sum(self.W[1:] ** 2)    # + lambda * sum(theta^2)
        return loss

    def _gradient(self, Xb, Y):
        m = Xb.shape[0]
        h = self._softmax(Xb @ self.W)
        grad = Xb.T @ (h - Y) / m
        if self.use_penalty:
            reg = 2 * self.lambda_ * self.W    # derivative of lambda * theta^2
            reg[0] = 0                         # never penalise the bias row
            grad += reg
        return grad

    # ---------------------------------------------------------- training
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        Xb, Y = self._add_bias(X), self._one_hot(y)
        m = Xb.shape[0]
        rng = np.random.default_rng(self.random_state)
        self.n_features_ = X.shape[1]
        self.W = rng.normal(0, 0.01, (Xb.shape[1], self.k))
        self.loss_history = []

        for epoch in range(self.max_iter):
            if self.method == "batch":
                self.W -= self.alpha * self._gradient(Xb, Y)
            else:  # mini-batch (or sgd when batch size is 1)
                bs = 1 if self.method == "sgd" else self.batch_size
                idx = rng.permutation(m)
                for s in range(0, m, bs):
                    b = idx[s:s + bs]
                    self.W -= self.alpha * self._gradient(Xb[b], Y[b])
            self.loss_history.append(self._loss(Xb, Y))
            if epoch > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                break
        return self

    # -------------------------------------------------------- prediction
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}")
        return self._softmax(self._add_bias(X) @ self.W)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    # ----------------------------------------------------------- metrics
    # TP/FP/FN for one class c (one-vs-rest)
    @staticmethod
    def _counts(y_true, y_pred, c):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        return tp, fp, fn

    def accuracy(self, y_true, y_pred):
        return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))  # correct / all

    def precision(self, y_true, y_pred, c):
        tp, fp, _ = self._counts(y_true, y_pred, c)
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    def recall(self, y_true, y_pred, c):
        tp, _, fn = self._counts(y_true, y_pred, c)
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    def f1_score(self, y_true, y_pred, c):
        p, r = self.precision(y_true, y_pred, c), self.recall(y_true, y_pred, c)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def support(self, y_true, c):
        return int(np.sum(np.asarray(y_true) == c))   # true samples of class c

    # per-class helpers used by the averages
    def _scores(self, fn, y_true, y_pred):
        return np.array([fn(y_true, y_pred, c) for c in range(self.k)])

    def _weights(self, y_true):
        s = np.array([self.support(y_true, c) for c in range(self.k)], dtype=float)
        return s / s.sum()

    # macro = plain average over classes
    def macro_precision(self, y_true, y_pred):
        return float(self._scores(self.precision, y_true, y_pred).mean())

    def macro_recall(self, y_true, y_pred):
        return float(self._scores(self.recall, y_true, y_pred).mean())

    def macro_f1(self, y_true, y_pred):
        return float(self._scores(self.f1_score, y_true, y_pred).mean())

    # weighted = average weighted by each class's share of the data
    def weighted_precision(self, y_true, y_pred):
        return float(np.sum(self._weights(y_true) * self._scores(self.precision, y_true, y_pred)))

    def weighted_recall(self, y_true, y_pred):
        return float(np.sum(self._weights(y_true) * self._scores(self.recall, y_true, y_pred)))

    def weighted_f1(self, y_true, y_pred):
        return float(np.sum(self._weights(y_true) * self._scores(self.f1_score, y_true, y_pred)))

    def report(self, y_true, y_pred):
        """Return a table similar to sklearn's classification_report."""
        lines = [f"{'class':>12}{'precision':>11}{'recall':>9}{'f1-score':>10}{'support':>9}"]
        for c in range(self.k):
            lines.append(f"{c:>12}{self.precision(y_true, y_pred, c):>11.2f}"
                         f"{self.recall(y_true, y_pred, c):>9.2f}"
                         f"{self.f1_score(y_true, y_pred, c):>10.2f}{self.support(y_true, c):>9}")
        n = len(y_true)
        lines.append(f"{'accuracy':>12}{'':>11}{'':>9}{self.accuracy(y_true, y_pred):>10.2f}{n:>9}")
        lines.append(f"{'macro avg':>12}{self.macro_precision(y_true, y_pred):>11.2f}"
                     f"{self.macro_recall(y_true, y_pred):>9.2f}{self.macro_f1(y_true, y_pred):>10.2f}{n:>9}")
        lines.append(f"{'weighted avg':>12}{self.weighted_precision(y_true, y_pred):>11.2f}"
                     f"{self.weighted_recall(y_true, y_pred):>9.2f}{self.weighted_f1(y_true, y_pred):>10.2f}{n:>9}")
        return "\n".join(lines)
