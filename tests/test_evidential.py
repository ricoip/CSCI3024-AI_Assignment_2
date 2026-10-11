import numpy as np
import pytest
from src.evidential import MLP, evidential_loss, variance_uncertainty, entropy_uncertainty


def test_loss_gradient_against_central_difference():
    a = np.array([[1.2, 2.1, 4.3], [3., 1.1, 2.]])
    y = np.array([1, 0])
    _, grad = evidential_loss(a, y, .13)
    numeric = np.empty_like(a)
    for index in np.ndindex(a.shape):
        plus, minus = a.copy(), a.copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        numeric[index] = (evidential_loss(plus,y,.13)[0]-evidential_loss(minus,y,.13)[0])/2e-5
    np.testing.assert_allclose(grad, numeric, atol=1e-8, rtol=1e-6)


def test_total_covariance_matches_monte_carlo():
    a = np.array([[2., 5., 3.]])
    u = variance_uncertainty(a)
    p = np.random.default_rng(7).dirichlet(a[0], 150000)
    epi = np.cov(p, rowvar=False, bias=True)
    mean_p = p.mean(0)
    alea = np.diag(mean_p)-p.T@p/len(p)
    np.testing.assert_allclose(u['epistemic_covariance'][0], epi, atol=3e-4)
    np.testing.assert_allclose(u['aleatoric_covariance'][0], alea, atol=8e-4)
    np.testing.assert_allclose(u['covariance'], u['epistemic_covariance']+u['aleatoric_covariance'])
    np.testing.assert_allclose(u['covariance'].sum(2), 0, atol=1e-15)
    assert np.linalg.eigvalsh(u['covariance'][0]).min() > -1e-14
    np.testing.assert_allclose(np.diag(u['correlation'][0]), 1)


def test_same_probabilities_more_evidence_less_epistemic():
    a = np.array([[2., 3., 5.]])
    u, v = variance_uncertainty(a), variance_uncertainty(10*a)
    np.testing.assert_allclose(u['probabilities'], v['probabilities'])
    np.testing.assert_allclose(u['total'], v['total'])
    assert v['epistemic'][0] < u['epistemic'][0]
    e = entropy_uncertainty(a)
    np.testing.assert_allclose(e['total'], e['aleatoric']+e['epistemic'])


@pytest.mark.parametrize('bad', [[[0,1]], [[-1,2]], [[np.nan,1]], [1,2]])
def test_invalid_alpha(bad):
    with pytest.raises(ValueError): variance_uncertainty(bad)


def test_training_and_checkpoint_roundtrip(tmp_path):
    x = np.random.default_rng(8).normal(size=(60, 4))
    y = (x[:,0] > 0).astype(int)
    model = MLP(seed=3, inputs=4, hidden=16, classes=2)
    initial = model.probabilities(x)
    model.fit(x,y,x,y,epochs=40)
    final = model.probabilities(x)
    assert -np.log(final[np.arange(len(y)), y]).mean() < -np.log(initial[np.arange(len(y)), y]).mean()
    file = tmp_path/'model.npz'
    model.save(file)
    np.testing.assert_array_equal(final, MLP.load(file).probabilities(x))


def test_dataset_split_has_no_leakage():
    from sklearn.datasets import load_digits
    from src.experiment import split_indices
    y = load_digits().target
    train, val, test = split_indices(y, 42)
    assert len(train)==1077 and len(val)==360 and len(test)==360
    assert set(train).isdisjoint(val) and set(train).isdisjoint(test) and set(val).isdisjoint(test)
    assert set(np.concatenate([train,val,test]))==set(range(len(y)))
