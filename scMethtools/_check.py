# #!/usr/bin/python3
# # -*- coding: utf-8 -*-
# """
# > Author: zongwt 
# > Created Time: 2023年08月21日

# """
from scipy import sparse,issparse

import numbers
import numpy as np
import pandas as pd
import glob

# TODO: Add docstrings
def make_dense(X):
    """TODO."""
    if issparse(X):
        XA = X.A if X.ndim == 2 else X.A1
    else:
        XA = X.A1 if isinstance(X, np.matrix) else X
    return np.array(XA)


# TODO: Add docstrings
def is_view(adata):
    """TODO."""
    return (
        adata.is_view
        if hasattr(adata, "is_view")
        else adata.isview
        if hasattr(adata, "isview")
        else adata._isview
        if hasattr(adata, "_isview")
        else True
    )

# TODO: Finish docstrings
def strings_to_categoricals(adata):
    """Transform string annotations to categoricals."""
    from pandas import Categorical
    from pandas.api.types import is_bool_dtype, is_integer_dtype, is_string_dtype

    def is_valid_dtype(values):
        return (
            is_string_dtype(values) or is_integer_dtype(values) or is_bool_dtype(values)
        )

    df = adata.obs
    df_keys = [key for key in df.columns if is_valid_dtype(df[key])]
    for key in df_keys:
        c = df[key]
        c = Categorical(c)
        if 1 < len(c.categories) < min(len(c), 100):
            df[key] = c

    df = adata.var
    df_keys = [key for key in df.columns if is_string_dtype(df[key])]
    for key in df_keys:
        c = df[key].astype("U")
        c = Categorical(c)
        if 1 < len(c.categories) < min(len(c), 100):
            df[key] = c


# TODO: Add docstrings
def is_categorical(data, c=None):
    """TODO."""
    from pandas.api.types import is_categorical_dtype as cat

    if c is None:
        return cat(data)  # if data is categorical/array
    if not is_view(data):  # if data is anndata view
        strings_to_categoricals(data)
    return isinstance(c, str) and c in data.obs.keys() and cat(data.obs[c])


# TODO: Add docstrings
def is_int(key):
    """TODO."""
    return isinstance(key, (int, np.integer))

# try:
#     import anndata
# except (ImportError, SyntaxError):
#     # anndata not installed
#     pass

# def _check_data_dir(data_dir, assert_smoothed=False):
#     """
#     quickly peek into data_dir to make sure the user
#     did not specify an empty directory
#     """
#     npz_files = glob(os.path.join(data_dir, "*.npz"))
#     if not npz_files:
#         raise Exception(
#             f"Your specified DATA_DIR '{data_dir}' is invalid since it does not "
#             "contain any chromosome files.\n           Chromosome files "
#             "end in '.npz' and are automatically created by 'scbs prepare'."
#         )
#     if assert_smoothed:
#         smooth_files = glob(os.path.join(data_dir, "smoothed", "*.csv"))
#         if not smooth_files:
#             raise Exception(
#                 f"Your specified DATA_DIR '{data_dir}' is not smoothed yet."
#                 "\n           Please smooth your data with 'scbs smooth'."
#             )


# def check_positive(**params):
#     """Check that parameters are positive as expected.
#     Raises
#     ------
#     ValueError : unacceptable choice of parameters
#     """
#     for p in params:
#         if params[p] <= 0:
#             raise ValueError("Expected {} > 0, got {}".format(p, params[p]))


# def check_int(**params):
#     """Check that parameters are integers as expected.
#     Raises
#     ------
#     ValueError : unacceptable choice of parameters
#     """
#     for p in params:
#         if not isinstance(params[p], numbers.Integral):
#             raise ValueError("Expected {} integer, got {}".format(p, params[p]))


# def check_if_not(x, *checks, **params):
#     """Run checks only if parameters are not equal to a specified value.
#     Parameters
#     ----------
#     x : excepted value
#         Checks not run if parameters equal x
#     checks : function
#         Unnamed arguments, check functions to be run
#     params : object
#         Named arguments, parameters to be checked
#     Raises
#     ------
#     ValueError : unacceptable choice of parameters
#     """
#     for p in params:
#         if params[p] is not x and params[p] != x:
#             [check(p=params[p]) for check in checks]


# def check_in(choices, **params):
#     """Checks parameters are in a list of allowed parameters.
#     Parameters
#     ----------
#     choices : array-like, accepted values
#     params : object
#         Named arguments, parameters to be checked
#     Raises
#     ------
#     ValueError : unacceptable choice of parameters
#     """
#     for p in params:
#         if params[p] not in choices:
#             raise ValueError(
#                 "{} value {} not recognized. Choose from {}".format(
#                     p, params[p], choices
#                 )
#             )


# def check_between(v_min, v_max, **params):
#     """Checks parameters are in a specified range.
#     Parameters
#     ----------
#     v_min : float, minimum allowed value (inclusive)
#     v_max : float, maximum allowed value (inclusive)
#     params : object
#         Named arguments, parameters to be checked
#     Raises
#     ------
#     ValueError : unacceptable choice of parameters
#     """
#     for p in params:
#         if params[p] < v_min or params[p] > v_max:
#             raise ValueError(
#                 "Expected {} between {} and {}, "
#                 "got {}".format(p, v_min, v_max, params[p])
#             )


# def matrix_is_equivalent(X, Y):
#     """Check matrix equivalence with numpy, scipy and pandas."""
#     if X is Y:
#         return True
#     elif X.shape == Y.shape:
#         if sparse.issparse(X) or sparse.issparse(Y):
#             X = scprep.utils.to_array_or_spmatrix(X)
#             Y = scprep.utils.to_array_or_spmatrix(Y)
#         elif isinstance(X, pd.DataFrame) and isinstance(Y, pd.DataFrame):
#             return np.all(X == Y)
#         elif not (sparse.issparse(X) and sparse.issparse(Y)):
#             X = scprep.utils.toarray(X)
#             Y = scprep.utils.toarray(Y)
#             return np.allclose(X, Y)
#         else:
#             return np.allclose((X - Y).data, 0)
#     else:
#         return False


# def convert_to_same_format(data, target_data, columns=None, prevent_sparse=False):
#     """Convert data to same format as target data."""
#     # create new data object
#     if scprep.utils.is_sparse_dataframe(target_data):
#         if prevent_sparse:
#             data = pd.DataFrame(data)
#         else:
#             data = scprep.utils.SparseDataFrame(data)
#         pandas = True
#     elif isinstance(target_data, pd.DataFrame):
#         data = pd.DataFrame(data)
#         pandas = True
#     elif is_anndata(target_data):
#         data = anndata.AnnData(data)
#         pandas = False
#     else:
#         # nothing to do
#         return data
#     # retrieve column names
#     target_columns = target_data.columns if pandas else target_data.var
#     # subset column names
#     try:
#         if columns is not None:
#             if pandas:
#                 target_columns = target_columns[columns]
#             else:
#                 target_columns = target_columns.iloc[columns]
#     except (KeyError, IndexError, ValueError):
#         # keep the original column names
#         if pandas:
#             target_columns = columns
#         else:
#             target_columns = pd.DataFrame(index=columns)
#     # set column names on new data object
#     if pandas:
#         data.columns = target_columns
#         data.index = target_data.index
#     else:
#         data.var = target_columns
#         data.obs = target_data.obs
#     return data



# def is_anndata(data):
#     """Check if an object is an AnnData object."""
#     try:
#         return isinstance(data, anndata.AnnData)
#     except NameError:
#         # anndata not installed
#         return False


# def has_empty_columns(data):
#     """Check if an object has empty columns."""
#     try:
#         return np.any(np.array(data.sum(0)) == 0)
#     except AttributeError:
#         if is_anndata(data):
#             return np.any(np.array(data.X.sum(0)) == 0)
#         else:
#             raise