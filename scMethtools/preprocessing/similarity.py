#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年09月06日

"""

import matplotlib.pylab as plt
from chunkdot import cosine_similarity_top_k


a_data = ad.read("D://Test/scMethTools/test_anndata.h5ad")
x = cosine_similarity_top_k(a_data.X, top_k=5) #top n-th neighbors
plt.spy(x,markersize=6)