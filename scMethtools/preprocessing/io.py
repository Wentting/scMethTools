#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年09月06日

"""
import click
import pandas as pd
import numpy as np
import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy import sparse
import datetime as datetime
from typing_extensions import Literal
from typing import Union
import anndata as ad
import subprocess
import pathlib
import collections

def echo(*args, **kwargs):
    click.echo(*args, **kwargs, err=True)
    return
def secho(*args, **kwargs):
    click.secho(*args, **kwargs, err=True)
    return

def caculate_methylation_feature(csr_mat,start,end,n_cells,min_cov=0):
    mean_array = np.zeros(n_cells)
    # 计算每列的平均值
    mean_list = []
    for col_idx in range(csr_mat.shape[1]):
        col_data = csr_mat[start-1:end, col_idx].toarray()
        #col_data = col_data[col_data != 0]# 跳过值为0的数据
        #col_data[col_data == -1] = 0# 在计算的时候把-1当成0计算
        if col_data.size == 0 or col_data.size < min_cov:
            mean_array[col_idx] = np.nan
        else:
            mean_array[col_idx] = np.mean(col_data)
    return mean_array

def process_chromosome(temp_dir,chrom_data):
    chrom, chrom_records = chrom_data
    secho(f"\n merge {chrom} coo_matrix and bin it ", fg="green")
    #for coo_path in sorted(glob(os.path.join(temp_dir, "chr1*.npz"))):
    temp_dir = Path('D://Test/GSE56789/out/')
    coo_mat =  sparse.vstack([sparse.load_npz(path) for path in temp_dir.glob(f'{chrom}*npz')])
    coo_mat.data = np.where(coo_mat.data == 0, -1, coo_mat.data) # change all zero to -1 which means unmethylated for calculating
    csr_mat = coo_mat.tocsr()
    n_cells = csr_mat.shape[1]
    secho(f"\n merge {chrom} coo_matrix and bin it for {n_cells} cell ", fg="green")
    # 遍历每一行
    feature_mtx = {}
    for index, row in chrom_records.iterrows():
        chr = row['chrom']
        start = row['start']
        end = row['end']
        mean = caculate_methylation_feature(csr_mat, start, end, n_cells)
        feature_name = f"{chr}_{start}_{end}"
        feature_mtx[feature_name] = mean
    return feature_mtx