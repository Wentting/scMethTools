#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年08月21日

"""
from ._version import __version__
__author__ = ', '.join([
    'Wenting Zong',
])

from . import genome as ref
from . import preprocessing as pp
from . import plotting as pl
from . import dmr as dm
from . import get as get
from . import settings as settings
# read functions from AnnData
from anndata import AnnData
from anndata import read
from anndata import read_h5ad, read_csv, read_excel, read_hdf, read_loom, read_mtx, read_text, read_umi_tools
from .plotting._palette import palette,red_palette,blue_palette,green_palette,purple_palette,ditto_palette


import scanpy as sc

name = "scmat"
omics="""
 ####   ####  #    # ##### #    # #####  ####   ####  #       ####  
 #      #    # ##  ##   #   #    #   #   #    # #    # #      #      
  ####  #      # ## #   #   ######   #   #    # #    # #       ####  
      # #      #    #   #   #    #   #   #    # #    # #           # 
 #    # #    # #    #   #   #    #   #   #    # #    # #      #    # 
  ####   ####  #    #   #   #    #   #    ####   ####  ######  ####                                                
"""
print(omics)
print(f'Version: {__version__}, Tutorials: https://ngdc.cncb.ac.cn/methbank/scm')