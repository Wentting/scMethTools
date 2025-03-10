#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
single cell DNA methylation analysis tool 

"""
#from .build_mtx import *
from ..preprocessing.generate_scm import generate_scm,sliding_windows,features_to_scm,load_features,caculate_bins_residual,feature_to_scm,import_cells,save_cells
from .smooth import *
from ..preprocessing.scm import add_meta,load_scm,filter_features,filter_cells,feature_select
from ..dmr.annotation import annotation
from .dimension_reduction import pca,tsne,umap,getNClusters
from .get_df import dmr_df
from .denovo import denovo

__all__ = [
    "generate_scm",
    "sliding_windows"
]


