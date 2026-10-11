"""Independent NumPy implementation of Duan et al. (WACV 2024), Eqs. 5-10, 19-28.

The backbone is a small MLP. This implements the classification/UQ core,
not the paper's full active domain adaptation training pipeline.
"""
from pathlib import Path
import numpy as np
from scipy.special import digamma, expit, gammaln, polygamma


def _alpha(alpha):
    a = np.asarray(alpha, dtype=float)
    if a.ndim != 2 or a.shape[1] < 2 or not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError('alpha must be a finite, positive (samples, classes) matrix')
    return a


def variance_uncertainty(alpha):
    """Return sample and class uncertainty and all three covariance matrices."""
    a = _alpha(alpha)
    strength = a.sum(axis=1, keepdims=True)
    p = a / strength
    cov = -p[:, :, None] * p[:, None, :]
    idx = np.arange(a.shape[1])
    cov[:, idx, idx] += p
    epistemic = cov / (strength[:, :, None] + 1)
    aleatoric = cov - epistemic
    diagonal = np.diagonal(cov, axis1=1, axis2=2)
    epi_class = diagonal / (strength + 1)
    sd = np.sqrt(diagonal)
    denom = sd[:, :, None] * sd[:, None, :]
    correlation = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    return dict(probabilities=p, strength=strength[:, 0], total_class=diagonal,
                epistemic_class=epi_class, aleatoric_class=diagonal-epi_class,
                total=diagonal.sum(1), epistemic=epi_class.sum(1),
                aleatoric=(diagonal-epi_class).sum(1), covariance=cov,
                epistemic_covariance=epistemic, aleatoric_covariance=aleatoric,
                correlation=correlation)


def entropy_uncertainty(alpha):
    a = _alpha(alpha)
    strength = a.sum(1, keepdims=True)
    p = a / strength
    total = -(p * np.log(p)).sum(1)
    aleatoric = (p * (digamma(strength+1)-digamma(a+1))).sum(1)
    return dict(total=total, aleatoric=aleatoric, epistemic=total-aleatoric)


def evidential_loss(alpha, labels, regularization=0.01):
    """Paper Eqs. 5-7: NLL + KL(Dir(masked alpha) || Dir(1)).

    Returns batch-mean loss and its analytic derivative with respect to alpha.
    The true-class alpha is masked to 1 only in the KL term.
    """
    a = _alpha(alpha)
    labels = np.asarray(labels)
    if labels.shape != (len(a),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('labels must be an integer vector matching the batch')
    if np.any(labels < 0) or np.any(labels >= a.shape[1]) or regularization < 0:
        raise ValueError('invalid labels or regularization')
    y = np.eye(a.shape[1])[labels]
    strength = a.sum(1, keepdims=True)
    nll = np.log(strength[:, 0]) - np.log(a[np.arange(len(a)), labels])
    masked = y + (1-y)*a
    sm = masked.sum(1, keepdims=True)
    k = a.shape[1]
    kl = (gammaln(sm[:, 0])-gammaln(k)-gammaln(masked).sum(1)
          + ((masked-1)*(digamma(masked)-digamma(sm))).sum(1))
    grad_kl = (1-y)*((masked-1)*polygamma(1, masked)-(sm-k)*polygamma(1, sm))
    grad = (1/strength-y/a+regularization*grad_kl)/len(a)
    return float(np.mean(nll+regularization*kl)), grad


class MLP:
    """64 -> 128 ReLU -> 10, trained with manual backpropagation and Adam."""
    def __init__(self, mode='evidential', seed=42, inputs=64, hidden=128, classes=10):
        if mode not in ('evidential', 'softmax'):
            raise ValueError('mode must be evidential or softmax')
        self.mode, self.seed = mode, seed
        rng = np.random.default_rng(seed)
        self.params = dict(w1=rng.normal(0, np.sqrt(2/inputs), (inputs, hidden)),
                           b1=np.zeros(hidden),
                           w2=rng.normal(0, np.sqrt(2/hidden), (hidden, classes)),
                           b2=np.zeros(classes))

    def forward(self, x):
        h = np.maximum(x@self.params['w1']+self.params['b1'], 0)
        z = h@self.params['w2']+self.params['b2']
        if self.mode == 'evidential':
            # Unit uniform prior plus smooth, nonnegative learned evidence.
            out = np.logaddexp(0, z)+1
        else:
            e = np.exp(z-z.max(1, keepdims=True))
            out = e/e.sum(1, keepdims=True)
        return out, h, z

    def probabilities(self, x):
        out = self.forward(x)[0]
        return out/out.sum(1, keepdims=True)

    def fit(self, x, y, x_val, y_val, epochs=250, batch_size=64, lr=0.003,
            regularization=0.01):
        rng = np.random.default_rng(self.seed+1000)
        moments = {k: np.zeros_like(v) for k,v in self.params.items()}
        variances = {k: np.zeros_like(v) for k,v in self.params.items()}
        best_nll, step, self.best_epoch = np.inf, 0, 0
        best = {k:v.copy() for k,v in self.params.items()}
        history = []
        for epoch in range(1, epochs+1):
            order = rng.permutation(len(x))
            losses = []
            for start in range(0, len(x), batch_size):
                ids = order[start:start+batch_size]
                xb, yb = x[ids], y[ids]
                out, h, z = self.forward(xb)
                if self.mode == 'evidential':
                    loss, grad = evidential_loss(out, yb, regularization*min(1, epoch/10))
                    dz = grad*expit(z)
                else:
                    loss = -np.log(out[np.arange(len(yb)), yb].clip(1e-15)).mean()
                    dz = (out-np.eye(out.shape[1])[yb])/len(xb)
                dh = (dz@self.params['w2'].T)*(h > 0)
                grads = dict(w2=h.T@dz, b2=dz.sum(0), w1=xb.T@dh, b1=dh.sum(0))
                step += 1
                for key, g in grads.items():
                    moments[key] = .9*moments[key]+.1*g
                    variances[key] = .999*variances[key]+.001*g*g
                    self.params[key] -= lr*(moments[key]/(1-.9**step))/(np.sqrt(variances[key]/(1-.999**step))+1e-8)
                losses.append(loss*len(xb))
            p = self.probabilities(x_val)
            val_nll = float(-np.log(p[np.arange(len(y_val)), y_val].clip(1e-15)).mean())
            history.append(dict(epoch=epoch, loss=float(sum(losses)/len(x)),
                                val_nll=val_nll, val_accuracy=float((p.argmax(1)==y_val).mean())))
            if val_nll < best_nll:
                best_nll, self.best_epoch = val_nll, epoch
                best = {k:v.copy() for k,v in self.params.items()}
        self.params = best
        return history

    def save(self, path):
        np.savez_compressed(path, mode=self.mode, seed=self.seed, **self.params)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as d:
            model = cls(mode=str(d['mode']), seed=int(d['seed']), inputs=d['w1'].shape[0],
                        hidden=d['w1'].shape[1], classes=d['w2'].shape[1])
            model.params = {k:d[k].copy() for k in ('w1','b1','w2','b2')}
        return model
