#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年09月15日

"""

import numpy as np
from sklearn.linear_model import ridge_regression

class ImprovedPCA(skl_decomposition.PCA):
    """Patch sklearn PCA learner to include randomized PCA for sparse matrices.

    Scikit-learn does not currently support sparse matrices at all, even though
    efficient methods exist for PCA. This class patches the default scikit-learn
    implementation to properly handle sparse matrices.

    Notes
    -----
      - This should be removed once scikit-learn releases a version which
        implements this functionality.

    """
    # pylint: disable=too-many-branches
    def _fit(self, X):
        """Dispatch to the right submethod depending on the chosen solver."""
        X = self._validate_data(
            X,
            dtype=[np.float64, np.float32],
            reset=False,
            accept_sparse=["csr", "csc"],
            copy=self.copy
        )

        # Handle n_components==None
        if self.n_components is None:
            if self.svd_solver != "arpack":
                n_components = min(X.shape)
            else:
                n_components = min(X.shape) - 1
        else:
            n_components = self.n_components

        # Handle svd_solver
        self._fit_svd_solver = self.svd_solver
        if self._fit_svd_solver == "auto":
            # Sparse data can only be handled with the randomized solver
            if sp.issparse(X):
                self._fit_svd_solver = "randomized"
            # Small problem or n_components == 'mle', just call full PCA
            elif max(X.shape) <= 500 or n_components == "mle":
                self._fit_svd_solver = "full"
            elif 1 <= n_components < .8 * min(X.shape):
                self._fit_svd_solver = "randomized"
            # This is also the case of n_components in (0,1)
            else:
                self._fit_svd_solver = "full"

        # Ensure we don't try call arpack or full on a sparse matrix
        if sp.issparse(X) and self._fit_svd_solver != "randomized":
            raise ValueError("only the randomized solver supports sparse matrices")

        # Call different fits for either full or truncated SVD
        if self._fit_svd_solver == "full":
            return self._fit_full(X, n_components)
        elif self._fit_svd_solver in ["arpack", "randomized"]:
            return self._fit_truncated(X, n_components, self._fit_svd_solver)
        else:
            raise ValueError(
                "Unrecognized svd_solver='{0}'".format(self._fit_svd_solver)
            )

    def _fit_truncated(self, X, n_components, svd_solver):
        """Fit the model by computing truncated SVD (by ARPACK or randomized) on X"""
        n_samples, n_features = X.shape

        if isinstance(n_components, six.string_types):
            raise ValueError(
                "n_components=%r cannot be a string with svd_solver='%s'" %
                (n_components, svd_solver)
            )
        if not 1 <= n_components <= min(n_samples, n_features):
            raise ValueError(
                "n_components=%r must be between 1 and min(n_samples, "
                "n_features)=%r with svd_solver='%s'" % (
                    n_components, min(n_samples, n_features), svd_solver
                )
            )
        if not isinstance(n_components, (numbers.Integral, np.integer)):
            raise ValueError(
                "n_components=%r must be of type int when greater than or "
                "equal to 1, was of type=%r" % (n_components, type(n_components))
            )
        if svd_solver == "arpack" and n_components == min(n_samples, n_features):
            raise ValueError(
                "n_components=%r must be strictly less than min(n_samples, "
                "n_features)=%r with svd_solver='%s'" % (
                    n_components, min(n_samples, n_features), svd_solver
                )
            )

        random_state = check_random_state(self.random_state)

        self.mean_ = X.mean(axis=0)
        total_var = ut.var(X, axis=0, ddof=1)

        if svd_solver == "arpack":
            # Center data
            X -= self.mean_
            # random init solution, as ARPACK does it internally
            v0 = random_state.uniform(-1, 1, size=min(X.shape))
            U, S, V = sp.linalg.svds(X, k=n_components, tol=self.tol, v0=v0)
            # svds doesn't abide by scipy.linalg.svd/randomized_svd
            # conventions, so reverse its outputs.
            S = S[::-1]
            # flip eigenvectors' sign to enforce deterministic output
            U, V = svd_flip(U[:, ::-1], V[::-1])

        elif svd_solver == "randomized":
            # sign flipping is done inside
            U, S, V = randomized_pca(
                X,
                n_components=n_components,
                n_iter=self.iterated_power,
                flip_sign=True,
                random_state=random_state,
            )

        self.n_samples_ = n_samples
        self.components_ = V
        self.n_components_ = n_components

        # Get variance explained by singular values
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var.sum()
        self.singular_values_ = S.copy()  # Store the singular values.

        if self.n_components_ < min(n_features, n_samples):
            self.noise_variance_ = (total_var.sum() - self.explained_variance_.sum())
            self.noise_variance_ /= min(n_features, n_samples) - n_components
        else:
            self.noise_variance_ = 0

        return U, S, V

    def transform(self, X):
        check_is_fitted(self, ["mean_", "components_"], all_or_any=all)

        X = self._validate_data(
            X,
            accept_sparse=["csr", "csc"],
            dtype=[np.float64, np.float32],
            reset=False,
            copy=self.copy
        )

        if self.mean_ is not None:
            X = X - self.mean_
        X_transformed = np.dot(X, self.components_.T)
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)
        return X_transformed

def inv_logit_mat(x, min=0.0, max=1.0,):
    # the inverse logit transformation function
    p = np.exp(x) / (1.0+np.exp(x))
    which_large = np.isnan(p) & (~np.isnan(x))
    p[which_large] = 1.0
    return p*(max-min)+min

def sparse_logistic_pca(
        dat, lbd=0.0006, k=2, verbose=False, max_iters=100, crit=1e-5,
        randstart=False, procrustes=True, lasso=True,
):
    """
    A Python implementation of the sparse logistic PCA of the following paper:
        Lee, S., Huang, J. Z., & Hu, J. (2010). Sparse logistic principal components analysis for binary data.
        The annals of applied statistics, 4(3), 1579.

    This implementation is migrated from this R package:
        https://github.com/andland/SparseLogisticPCA

    Args:
        dat: input data, n*d numpy array where n is the numbers of samples and d is the feature dimensionality
        lbd: the lambda value, higher value will lead to more sparse components
        k: the dimension after reduction
        verbose: print log or not
        max_iters: maximum number of iterations
        crit: the minimum difference criteria for stopping training
        randstart: randomly initialize A, B, mu or not
        procrustes: procrustes
        lasso: whether to use LASSO solver

    Returns: a dict containing the results

    """

    ### Initialize q
    q = 2*dat-1
    q[np.isnan(q)] = 0.0
    n,d = dat.shape

    ### Initialize mu, A, B
    if not randstart:
        mu = np.mean(q,axis=0)
        udv_u, udv_d, udv_v = np.linalg.svd(q-np.mean(q,axis=0), full_matrices=False)
        A = udv_u[:,0:k].copy()
        B = np.matmul(udv_v[0:k,:].T, np.diag(udv_d[0:k]))
    else:
        mu = np.random.normal(size=(d,))
        A = np.random.uniform(low=-1.0, high=1.0, size=(n,k,))
        B = np.random.uniform(low=-1.0, high=1.0, size=(d,k,))

    loss_trace = dict()

    ## loop to optimize the loss, see Alogrithm 1 in the paper
    for m in range(max_iters):

        last_mu, last_A, last_B = mu.copy(), A.copy(), B.copy()

        theta = np.outer(np.ones(n), mu) + np.matmul(A, B.T)
        X = theta+4*q*(1-inv_logit_mat(q*theta))
        Xcross = X - np.matmul(A, B.T)
        mu = np.matmul((1.0/n) * Xcross.T, np.ones(n))

        theta = np.outer(np.ones(n), mu) + np.matmul(A, B.T)
        X = theta+4*q*(1-inv_logit_mat(q*theta))
        Xstar = X-np.outer(np.ones(n), mu)

        if procrustes:
            M_u, M_d, M_v = np.linalg.svd(np.matmul(Xstar, B), full_matrices=False)
            A = np.matmul(M_u, M_v)
        else:
            A = Xstar @ B @ np.linalg.inv(B.T @ B)
            A, _ = np.linalg.qr(A)

        theta = np.outer(np.ones(n), mu) + A @ B.T
        X = theta + 4 * q * (1 - inv_logit_mat(q * theta))
        Xstar = X-np.outer(np.ones(n), mu)

        if lasso:
            B_lse = Xstar.T @ A
            B = np.sign(B_lse) * np.maximum(0.0, np.abs(B_lse)-4*n*lbd)
        else:
            C = Xstar.T @ A
            B = (np.abs(B) / (np.abs(B)+4*n*lbd)) * C

        q_dot_theta = q*(np.outer(np.ones(n),mu) + A @ B.T)
        loglike = np.sum(np.log(inv_logit_mat(q_dot_theta))[~np.isnan(dat)])
        penalty = n*lbd*np.sum(abs(B))
        loss_trace[str(m)] = (-loglike+penalty) / np.sum(~np.isnan(dat))

        if verbose:
            print(f"Iter: {m} - Loss: {loss_trace[str(m)]:.4f}, NegLogLike: {-loglike:.4f}, Penalty: {penalty:.4f} ")

        if m>3:
            if loss_trace[str(m-1)] - loss_trace[str(m)] < crit:
                break

    if loss_trace[str(m-1)] < loss_trace[str(m)]:
        mu, A, B, m = last_mu, last_A, last_B, m-1

        q_dot_theta = q*(np.outer(np.ones(n),mu) + A @ B.T)
        loglike = np.sum(np.log(inv_logit_mat(q_dot_theta))[~np.isnan(dat)])

    zeros = sum(np.abs(B))
    BIC = -2.0*loglike+np.log(n)*(d+n*k+np.sum(np.abs(B)>=1e-10))

    res = {
        "mu":mu, "A":A, "B":B, "zeros":zeros,
        "BIC":BIC, "iters":m, "loss_trace":loss_trace, "lambda":lbd,
    }

    return res

class SparseLogisticPCA(object):
    """
    A warper class of sparse logistic PCA, which provides the fit, transform and fit_transform methods
    """
    def __init__(
            self, lbd=0.0001, n_components=2, verbose=False, max_iters=100, crit=1e-5,
            randstart=False, procrustes=True, lasso=True,
            ridge_alpha=0.01,
    ):
        """
        Args:
            lbd: the lambda value, higher value will lead to more sparse components
            n_components: the dimension after reduction, i.e. k in the origin paper
            verbose: print log or not
            max_iters: maximum number of iterations
            crit: the minimum difference criteria for stopping training
            randstart: randomly initialize A, B, mu or not
            procrustes: procrustes
            lasso: whether to use LASSO solver
            ridge_alpha: Amount of ridge shrinkage to apply in order to improve conditioning when
                calling the transform method.
        """
        self.lbd = lbd
        self.n_components = n_components
        self.verbose=verbose
        self.max_iters = max_iters
        self.crit = crit
        self.randstart = randstart
        self.procrustes = procrustes
        self.lasso=lasso
        self.ridge_alpha = ridge_alpha

    def fit(self, dat, verbose=False):
        """

        Args:
            dat: ndarray of shape (n_samples, n_features), the data to be fitted
            verbose: print log or not

        Returns:
            self

        """
        res = sparse_logistic_pca(
            dat, lbd=self.lbd, k=self.n_components, verbose=verbose,
            max_iters=self.max_iters, crit=self.crit,
            randstart=self.randstart, procrustes=self.procrustes,
            lasso=self.lasso,)

        self.mu, self.components_ = res['mu'], res['B'].T
        _, self.d = dat.shape

        components_norm = np.linalg.norm(self.components_, axis=1)[:, np.newaxis]
        components_norm[components_norm == 0] = 1
        self.components_ /= components_norm

        return self

    def transform(self, X):
        """

        Similar to Sparse PCA, the orthogonality of the learned components is not enforced in Sparse Logistic PCA,
            and hence one cannot use a simple linear projection.

        The origin paper does not describe how to transform the new data, and this implementation of transform
            function generally follows that of sklearn.decomposition.SparsePCA:
            https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html

        The handling of missing data (N/A) is not supported in this transform implementation.

        Args:
            X: ndarray of shape (n_samples, n_features), the input data

        Returns:
            ndarray of (n_samples, n_components), the data after dimensionality reduction

        """
        n, d = X.shape
        assert d==self.d,\
            f"Input data should have a shape (n_samples, n_features) and n_features should be {self.d}"

        Xstar = X - np.outer(np.ones(n), self.mu)

        U = ridge_regression(
            self.components_.T, Xstar.T, self.ridge_alpha, solver="cholesky",
        )

        return U

    def fit_transform(self, dat):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Args:
            dat: ndarray of shape (n_samples, n_features)

        Returns:
            ndarray of (n_samples, n_components)

        """
        self.fit(dat)
        return self.transform(dat)

    def fine_tune_lambdas(self, dat, lambdas=np.arange(0, 0.00061, 0.0006 / 10)):
        # fine tune the Lambda values based on BICs following the paper
        BICs, zeros = [], []
        for lbd in lambdas:
            # print(f"Lambda: {lbd:.6f}")
            this_res = sparse_logistic_pca(
            dat, lbd=lbd, k=self.n_components, verbose=False,
            max_iters=self.max_iters, crit=self.crit,
            randstart=self.randstart, procrustes=self.procrustes,
            lasso=self.lasso,)
            BICs.append(this_res['BIC'])
            zeros.append(this_res['zeros'])
        best_ldb = lambdas[np.argmin(BICs)]
        return best_ldb, BICs, zeros

    def set_lambda(self,new_lbd):
        print(f"Setting lambda to: {new_lbd}")
        self.lbd = new_lbd

    def set_ridge_alpha(self, ridge_alpha):
        self.ridge_alpha = ridge_alpha

    def set_n_components(self, n_components):
        self.n_components = n_components

    def get_components(self):
        return self.components_