#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年09月22日

"""

import numpy as np

def _find_lower_upper_features(
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
    idx = idx[n_lower:n-n_upper]
    return idx[::-1][:total_features]



