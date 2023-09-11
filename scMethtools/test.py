#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年09月06日

"""
import pandas as pd
import numpy as np
import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy import sparse
import time
from typing_extensions import Literal
from typing import Union
import anndata as ad
import subprocess
import pathlib
import collections


class Genome:
    def __init__(self, chrom_sizes, annotation_filename, fasta=None) -> None:
        self.chrom_sizes = chrom_sizes
        self._annotation_filename = annotation_filename
        self._fasta_filename = fasta

    def fetch_annotations(self):
        return datasets().fetch(self._annotation_filename)

    def fetch_fasta(self):
        return datasets().fetch(self._fasta_filename, processor=Decompress(method="gzip"))


# 定义一个函数，用于处理染色体上的注释
def process_chromosome(temp_dir, chrom_data):
    temp_dir = Path('D://Test/GSE56789/out/')
    chrom, chrom_records = chrom_data
    # 使用 glob 模块查找文件
    matching_files = glob.glob(os.path.join(temp_dir, f'{chrom}*npz'))
    if not matching_files:
        print(f"No matching files found for {chrom}. Returning empty result.")
        return {}  # 返回空的结果，一个空的字典
    else:
        secho(f"\n merge {chrom} coo_matrix and bin it ", fg="green")
        # for coo_path in sorted(glob(os.path.join(temp_dir, "chr1*.npz"))):
        coo_mat = sparse.vstack([sparse.load_npz(path) for path in temp_dir.glob(f'{chrom}*npz')])
        coo_mat.data = np.where(coo_mat.data == 0, -1,
                                coo_mat.data)  # change all zero to -1 which means unmethylated for calculating
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
        print(f'merge finished for {chrom}')
        return feature_mtx


import click


def echo(*args, **kwargs):
    click.echo(*args, **kwargs, err=True)
    return


def secho(*args, **kwargs):
    click.secho(*args, **kwargs, err=True)
    return


def caculate_methylation_feature(csr_mat, start, end, n_cells, min_cov=0):
    mean_array = np.zeros(n_cells)
    # 计算每列的平均值
    mean_list = []
    for col_idx in range(csr_mat.shape[1]):
        col_data = csr_mat[start - 1:end, col_idx].toarray()
        col_data = col_data[col_data != 0]# 跳过值为0的数据
        col_data[col_data == -1] = 0# 在计算的时候把-1当成0计算
        if col_data.size == 0 or col_data.size < min_cov:
            mean_array[col_idx] = np.NaN
        else:
            mean_array[col_idx] = np.mean(col_data)
    return mean_array


def _sliding_windows_with_step_size(window_size, step_size, ref, chrom_size=None, chrom_file=None):
    """

    :param window_size:
    :param step_size:
    :param ref:
        A Genome object, providing gene annotation and chromosome sizes.
        If not set, `gff_file` and `chrom_size` must be provided.
        `genome` has lower priority than `gff_file` and `chrom_size`.
    :param chrom_file:
        File name of the gene annotation file in BED or GFF or GTF format.
        This is required if `ref` is not set.
        Setting `chrom_file` will override the annotations from the `genome` parameter.
    :param chrom_size:
        A dictionary containing chromosome sizes, for example,
        `{"chr1": 2393, "chr2": 2344, ...}`.
        This is required if `genome` is not set.
        Setting `chrom_size` will override the chrom_size from the `genome` parameter.
    :return:
    """
    chrom_size_dict = {}

    if step_size is None:
        step_size = window_size
    if chrom_file is None:
        if ref is not None:
            chrom_size_dict = ref.chrom_sizes
    else:
        chrom_size_dict = _load_chrom_size_file(chrom_file)  # user defined reference, especially for other species

    records = []
    for chrom, chrom_length in chrom_size_dict.items():
        bin_start = np.array(list(range(0, chrom_length, step_size)))
        bin_end = bin_start + window_size
        bin_end[np.where(bin_end > chrom_length)] = chrom_length
        chrom_df = pd.DataFrame(dict(start=bin_start, end=bin_end))
        chrom_df['chrom'] = chrom
        records.append(chrom_df)
    total_df = pd.concat(records)[['chrom', 'start', 'end']].reset_index(drop=True)
    return total_df


def main():
    GRCh38 = Genome(
        {
            "chr1": 248956422,
            "chr2": 242193529,
            "chr3": 198295559,
            "chr4": 190214555,
            "chr5": 181538259,
            "chr6": 170805979,
            "chr7": 159345973,
            "chr8": 145138636,
            "chr9": 138394717,
            "chr10": 133797422,
            "chr11": 135086622,
            "chr12": 133275309,
            "chr13": 114364328,
            "chr14": 107043718,
            "chr15": 101991189,
            "chr16": 90338345,
            "chr17": 83257441,
            "chr18": 80373285,
            "chr19": 58617616,
            "chr20": 64444167,
            "chr21": 46709983,
            "chr22": 50818468,
            "chrX": 156040895,
            "chrY": 57227415
        },
        "gencode_v41_GRCh38.gff3.gz",
        "gencode_v41_GRCh38.fa.gz",test_chr1.txt
    )
    hg38 = GRCh38
    cpu = 5
    feature = _sliding_windows_with_step_size(100000, 100000, ref=hg38)
    # all chromosome or defined chromosome
    chrom_data = [(chrom, feature[feature['chrom'] == chrom]) for chrom in feature['chrom'].unique()]
    pool = mp.Pool(processes=min(cpu, len(chrom_data)))
    # 将染色体数据分组传递给处理函数
    results = []
    print("开始执行主程序")
    start_time = time.time()
    for i in range(len(chrom_data)):
        chrom = chrom_data[i]
        results.append(pool.apply_async(process_chromosome, args=('D://Test/GSE56789/out/', chrom)))
    # with Pool(processes=2) as pool:
    #     # 使用 map 方法并行处理 chrom_data 中的每个元素，固定参数为 fixed_arg
    #     partial_process_chromosome = partial(process_chromosome, 'D://Test/GSE56789/out/')
    #     results = pool.map(partial_process_chromosome, chrom_data)

    # 关闭线程池
    pool.close()
    print("线程池已关闭")
    pool.join()
    print("线程池已加入")
    # for r in results:
    #     print("enter")
    #     print(r)
    #     if r.get():
    #         print(r.get())
    # # 将处理结果合并成一个 DataFrame
    result_df = pd.concat([pd.DataFrame.from_dict(r.get()) for r in results], ignore_index=True)
    print("all finish")
    # # 将处理结果合并成一个 DataFrame
    # result_df = pd.concat([pd.DataFrame.from_dict(r.get()) for r in results], ignore_index=True)
    # print(result_df)

    print("注释合并进程结束耗时%s" % (time.time() - start_time))

    # 步骤 1: 将 DataFrame 转换为 CSR 矩阵
    # 步骤 1: 填充 DataFrame 中的 NaN 值
    df_filled = result_df.fillna(0)  # 这里将 NaN 填充为你希望的缺失值填充值（例如 0）
    csr_matrix = sparse.csr_matrix(df_filled.values)

    # 保存行名和列名
    row_names = result_df.index.tolist()
    col_names = result_df.columns.tolist()

    # 步骤 2: 存储 CSR 矩阵到 AnnData 对象
    #`var` must have number of columns of `X` (2490), but has 30894 rows.

    adata = ad.AnnData(X=csr_matrix,obs=,var=feature)

    # 步骤 3: 将 DataFrame 保存为文本文件
    result_df.to_csv('test_chr1.txt', sep='\t', index=False)

    # 保存 AnnData 对象到 HDF5 文件（如果需要）
    adata.write('test_anndata.h5ad')

    # 1. 从另一个文件中读取包含所有数据的DataFrame
    # 假设您的数据存储在一个名为 'data.csv' 的CSV文件中，且该文件包含多列数据，包括 'id' 列作为标识符
    data_df = pd.read_csv('data.csv')

    # 2. 使用ID列表和DataFrame创建映射字典
    id_to_data = {}
    for sample_id in row_names:
        data_row = data_df[data_df['id'] == sample_id]
        if not data_row.empty:
            id_to_data[sample_id] = data_row

    # 3. 创建一个AnnData对象并将数据添加到obs中
    # 假设您已经有一个名为 'adata' 的AnnData对象
    for sample_id, data_row in id_to_data.items():
        adata.obs[f'ID_{sample_id}'] = data_row.iloc[0]  # 将整行数据添加到obs中，以ID为列名


if __name__ == "__main__":
    main()