#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年08月21日

"""
def filter():
    # filter

    # Basic filtering parameters
    mapping_rate_cutoff = 0.5
    mapping_rate_col_name = 'MappingRate'  # Name may change
    final_reads_cutoff = 500000
    final_reads_col_name = 'FinalReads'  # Name may change
    mccc_cutoff = 0.03
    mccc_col_name = 'mCCCFrac'  # Name may change
    mch_cutoff = 0.2
    mch_col_name = 'mCHFrac'  # Name may change
    mcg_cutoff = 0.5
    mcg_col_name = 'mCGFrac'  # Name may change

    # 加载meta数据
    metadata = pd.read_csv(metadata_path, index_col=0)
    total_cells = metadata.shape[0]
    print(f'Metadata of {total_cells} cells')

    # 按照过滤指标过滤
    _cutoff = mapping_rate_cutoff
    _col_name = mapping_rate_col_name

    mapping_rate_judge = metadata[_col_name] > _cutoff
    _passed_cells = mapping_rate_judge.sum()
    print(
        f'{_passed_cells} / {total_cells} cells ({_passed_cells / total_cells * 100:.1f}%) '
        f'passed the {_col_name} cutoff {_cutoff}.')
    # 16985 / 16985 cells (100.0%) passed the MappingRate cutoff 0.5.

    # 结果测试
    try:
        assert (passed_cells / total_cells) > 0.6
    except AssertionError as e:
        e.args += (
            'A large amount of the cells do not pass filter, check your cutoffs or overall dataset quality.',
        )
        raise e

    print('Feel good')

def _filter_cells(data,re_cells=None,min_features=None):
	"""
    Filter cell outliers based on counts and numbers of genes expressed.
    For instance, only keep cells with at least `min_counts` counts or
    `min_tsse` TSS enrichment scores. This is to filter measurement outliers,
    i.e. "unreliable" observations.
    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
    min_counts
        Minimum number of counts required for a cell to pass filtering.
    min_tsse
        Minimum TSS enrichemnt score required for a cell to pass filtering.
    max_counts
        Maximum number of counts required for a cell to pass filtering.
    max_tsse
        Maximum TSS enrichment score expressed required for a cell to pass filtering.
    inplace
        Perform computation inplace or return result.
    Returns
    -------
    np.ndarray | None:
        If `inplace = True`, directly subsets the data matrix. Otherwise return
        a boolean index mask that does filtering, where `True` means that the
        cell is kept, `False` means the cell is removed.
    """
    if min_counts: selected_cells &= data.obs["n_fragment"] >= min_counts
    if max_counts: selected_cells &= data.obs["n_fragment"] <= max_counts
    if min_tsse: selected_cells &= data.obs["tsse"] >= min_tsse
    if max_tsse: selected_cells &= data.obs["tsse"] <= max_tsse

	return cells

def _filter_features(data,white_list=None,black_list=None,min_cov_cells=None, most_variable: int | float | None = 10000):
		"""
    Filter cell outliers based on counts and numbers of genes expressed.
    For instance, only keep cells with at least `min_counts` counts or
    `min_tsse` TSS enrichment scores. This is to filter measurement outliers,
    i.e. "unreliable" observations.
    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
    most_variable_count: number of the most variable features.
    Returns
    -------
    np.ndarray | None:
        If `inplace = True`, directly subsets the data matrix. Otherwise return
        a boolean index mask that does filtering, where `True` means that the
        cell is kept, `False` means the cell is removed.
    """

   	#先按照feature特性筛选
    count = np.zeros(adata.shape[1])

    for batch, _, _ in adata.chunked_X(2000):
        batch.data = np.ones(batch.indices.shape, dtype=np.float64)
        count += np.ravel(batch.sum(axis = 0))

    selected_features = count >= min_cells


	if whitelist is not None:
        selected_features &= internal.intersect_bed(adata.var_names, str(whitelist))
    if blacklist is not None:
        selected_features &= np.logical_not(internal.intersect_bed(adata.var_names, str(blacklist)))

    if most_variable is not None and len(count[selected_features]) > most_variable:
        mean = count[selected_features].mean()
        std = math.sqrt(count[selected_features].var())
        zscores = np.absolute((count - mean) / std)
        cutoff = np.sort(zscores)[most_variable - 1]
        selected_features &= zscores <= cutoff

    if inplace:
        adata.var["selected"] = selected_features
    else:
        return features

def _find_most_accessible_features(
        feature_count,
        filter_lower_quantile,
        filter_upper_quantile,
        total_features,
) -> np.ndarray:
    idx = np.argsort(feature_count)
    for i in range(idx.size):
        if feature_count[idx[i]] > 0:
            break
    idx = idx[i:]
    n = idx.size
    n_lower = int(filter_lower_quantile * n)
    n_upper = int(filter_upper_quantile * n)
    idx = idx[n_lower:n - n_upper]
    return idx[::-1][:total_features]


def select_features(
        adata: AnnData | AnnDataSet | list[AnnData],
        n_features: int = 500000,
        filter_lower_quantile: float = 0.005,
        filter_upper_quantile: float = 0.005,
        whitelist: Path | None = None,
        blacklist: Path | None = None,
        max_iter: int = 1,
        inplace: bool = True,
        n_jobs: int = 8,
) -> np.ndarray | list[np.ndarray] | None:
    """
    Perform feature selection.

    Note
    ----
    This function does not perform the actual subsetting. The feature mask is used by
    various functions to generate submatrices on the fly.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
        `adata` can also be a list of AnnData objects.
        In this case, the function will be applied to each AnnData object in parallel.
    n_features
        Number of features to keep. Note that the final number of features
        may be smaller than this number if there is not enough features that pass
        the filtering criteria.
    filter_lower_quantile
        Lower quantile of the feature count distribution to filter out.
    filter_upper_quantile
        Upper quantile of the feature count distribution to filter out.
    whitelist
        A user provided bed file containing genome-wide whitelist regions.
        None-zero features listed here will be kept regardless of the other
        filtering criteria.
        If a feature is present in both whitelist and blacklist, it will be kept.
    blacklist
        A user provided bed file containing genome-wide blacklist regions.
        Features that are overlapped with these regions will be removed.
    inplace
        Perform computation inplace or return result.
    n_jobs
        Number of parallel jobs to use when `adata` is a list.

    Returns
    -------
    np.ndarray | None:
        If `inplace = False`, return a boolean index mask that does filtering,
        where `True` means that the feature is kept, `False` means the feature is removed.
        Otherwise, store this index mask directly to `.var['selected']`.
    """
    if isinstance(adata, list):
        result = snapatac2._utils.anndata_par(
            adata,
            lambda x: select_features(x, n_features, filter_lower_quantile, filter_upper_quantile, whitelist, blacklist,
                                      max_iter, inplace),
            n_jobs=n_jobs,
        )
        if inplace:
            return None
        else:
            return result

    count = np.zeros(adata.shape[1])
    for batch, _, _ in adata.chunked_X(2000):
        count += np.ravel(batch.sum(axis=0))
    adata.var['count'] = count

    selected_features = _find_most_accessible_features(
        count, filter_lower_quantile, filter_upper_quantile, n_features)

    if blacklist is not None:
        blacklist = np.array(internal.intersect_bed(adata.var_names, str(blacklist)))
        selected_features = selected_features[np.logical_not(blacklist[selected_features])]

    # Iteratively select features
    iter = 1
    while iter < max_iter:
        embedding = snapatac2.tl.spectral(adata, features=selected_features, inplace=False)[1]
        clusters = snapatac2.tl.leiden(snapatac2.pp.knn(embedding, inplace=False))
        rpm = snapatac2.tl.aggregate_X(adata, groupby=clusters).X
        var = np.var(np.log(rpm + 1), axis=0)
        selected_features = np.argsort(var)[::-1][:n_features]

        # Apply blacklist to the result
        if blacklist is not None:
            selected_features = selected_features[np.logical_not(blacklist[selected_features])]
        iter += 1

    result = np.zeros(adata.shape[1], dtype=bool)
    result[selected_features] = True

    # Finally, apply whitelist to the result
    if whitelist is not None:
        whitelist = np.array(internal.intersect_bed(adata.var_names, str(whitelist)))
        whitelist &= count != 0
        result |= whitelist

    logging.info(f"Selected {result.sum()} features.")

    if inplace:
        adata.var["selected"] = result
    else:
        return result