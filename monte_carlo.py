from jupyter_notebook import * # local helper functions useful in Jupyter

def antithetic_normal(n_periods, n_paths):
    np.random.seed(1234)
    assert n_paths%2 == 0, 'n_paths must be an even number'
    n2 = int(n_paths/2)
    z = np.random.normal(0,1, (n_periods, n2))
    return np.hstack((z,-z))

