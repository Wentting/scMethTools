#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年08月21日

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
import datetime as datetime
from typing_extensions import Literal
from typing import Union
import anndata as ad
import subprocess
import pathlib
from .._genome import Genome
import collections

__all__ = ['import_bed_file','convert_coo_file_to_matrix']



def find_files_with_suffix(path, suffix):
    matching_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(suffix):
                matching_files.append(os.path.join(root, file))
    return matching_files


def filter_cells(
        data: ad.AnnData,
        min_counts: int = 1000,
        min_tsse: float = 5.0,
        max_counts: int = 30000,
        max_tsse: float = 10.0,
        inplace: bool = True,
) -> Union[np.ndarray, None]:
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
    selected_cells = True
    if min_counts: selected_cells &= data.obs["n_fragment"] >= min_counts
    if max_counts: selected_cells &= data.obs["n_fragment"] <= max_counts
    if min_tsse: selected_cells &= data.obs["tsse"] >= min_tsse
    if max_tsse: selected_cells &= data.obs["tsse"] <= max_tsse

    if inplace:
        if data.isbacked:
            data.subset(selected_cells)
        else:
            data._inplace_subset_obs(selected_cells)
    else:
        return selected_cells


def stat(
        cov_df: str,
        cell_name: str,
        out_dir: Path,
):
    sta_df = cov_df.groupby('mc_class').agg({"methylated": "sum", "mc_class": "count", "total": "sum"})
    sta_df['overall_mc_level'] = sta_df['methylated'] / sta_df['total']
    # 使用 stack() 函数将 DataFrame 转换为单列的 Series 对象
    df_transposed = sta_df.stack().to_frame().T
    # 将列名修改为需要的格式
    df_transposed.columns = [f'{col}_{row}' for row in sta_df.index for col in sta_df.columns]
    df_transposed['sample'] = cell_name
    stat_file = os.path.join(out_dir, "basic_summary.csv")
    if os.path.exists(stat_file):
        df_transposed.to_csv(stat_file, index=False, header=False, sep='\t', mode='a')
    else:
        df_transposed.to_csv(stat_file, index=False, sep='\t', mode='a')
    return stat_file


def _iter_chunks(data_dir, chrom):
    print(os.path.join(data_dir, f"chr{chrom}_chunk*.coo"))
    chunk_paths = glob.glob(os.path.join(data_dir, "{}_chunk*.coo".format(chrom)))
    for chunk_path in sorted(chunk_paths):
        chunk = pd.read_csv(chunk_path, delimiter=",", header=None).values  # array
        # os.remove(chunk_file) #删除中间文件
        # sort_chunk = np.sort(chunk) #numpy.ndarray  ############################################这里要排序，报错了这里
        yield chunk


def _read_coo_file(coo_file):
    rows, cols, data = coo_file[:, 0], coo_file[:, 1], coo_file[:, 2]
    # print(rows,cols,data)
    return rows, cols, data

def _convert_cov_to_coo(
        output_dir: Path,
        cov_file: str,
        cell_id: int,
        pipeline: str = 'Bismark',
        out_file: str = None,
        chunksize: int = 1000000,
        round_sites: bool = True,
) -> dict[str, int]:
    """
    convert single methylation cov/bed file into coo_matrix with fixed batch in temp dir
    :param out_dir:
    :param cov_file:
    :param chunksize:
    :param cell_id:
    :param out_file:
    :param round_sites:
    :return:
    """
    print('step1 : begin build coo matrix ----- \n')
    print('pipeline is ' +pipeline+'\n')


    if out_file is None:
        name = os.path.basename(cov_file)
        sample_name = os.path.splitext(name)[0]
        print('processed cell ', sample_name, '...')

    cov_df = pd.read_csv(cov_file, sep='\t', header=None,
                         names=['chr', 'strand', 'pos', 'mc_class', 'mc_context', 'mc_level', 'methylated',
                                'total'])  # usecols=['chr','pos','mc_level','total']
    cov_df = cov_df.sort_values(['chr', 'pos'])
    chrom_sizes = {}
    current_chrom = None
    current_chunk = None
    current_file = None
    # record qc into stats file
    stat_file = stat(cov_df, sample_name, output_dir)

    # 测试用，不测试的时候删除
    valid_values = ["chr1", "chr2"]

    for i, row in cov_df.iterrows():
        chrom, pos, mc_level = row['chr'], row['pos'], row['mc_level']
        chunk = int(pos // chunksize)
        if chrom in valid_values:  # 测试用，不测试删除这个判断
            if chrom != current_chrom or chunk != current_chunk:
                current_chrom = chrom
                current_chunk = chunk
                if current_file:
                    current_file.close()  # 每次结束一个染色体的一个chunk区域后要关闭文件
                filename = '{}_chunk{:07d}.coo'.format(current_chrom, current_chunk)
                current_file = open(os.path.join(output_dir, filename), 'a')
                if chrom not in chrom_sizes:
                    chrom_sizes[chrom] = 0
            current_file.write('{}, {} ,{}\n'.format(pos, cell_id, mc_level))
            if pos > chrom_sizes[current_chrom]:
                chrom_sizes[current_chrom] = pos  # 记录的是测到的最大碱基的位置不是真正的基因组位置
    if current_file:
        current_file.close()
    return chrom_sizes


def caculate_methylation_feature(csr_mat,start,end,n_cells,min_cov=3):
    mean_array = np.zeros(n_cells)
    # 计算每列的平均值
    mean_list = []
    for col_idx in range(csr_mat.shape[1]):
        col_data = csr_mat[start-1:end, col_idx].toarray()
        col_data = col_data[col_data != 0]# 跳过值为0的数据
        col_data[col_data == -1] = 0# 在计算的时候把-1当成0计算
        if col_data.size == 0:
            mean_array[col_idx] = np.nan
        if col_data.size < min_cov:
            mean_array[col_idx] = np.nan #如果一个feature覆盖的细胞数量少于多少就删除
        else:
            mean_array[col_idx] = np.mean(col_data)
    return mean_array


def convert_coo_file_to_matrix(data_dir, chrom):
    print("convet file into coo_matrix")
    coo_matrix_path = os.path.join(data_dir, "{}_coo_matrix.npz".format(chrom))
    chunk_row = []
    chunk_col = []
    chunk_data = []
    for chunk_file in _iter_chunks(data_dir, chrom):  # 按顺序处理染色体的chunk
        file_rows, file_cols, file_data = _read_coo_file(chunk_file)
        chunk_row.extend(file_rows.astype(int))
        chunk_col.extend(file_cols.astype(int))
        chunk_data.extend(file_data)
    print(chunk_row)
    coo_matrix_result = coo_matrix((chunk_data, (chunk_row, chunk_col)))
    sparse.save_npz(coo_matrix_path, coo_matrix_result)
    return coo_matrix_path, coo_matrix_result

def _load_chrom_size_file(chrom_file,remove_chr_list=None):
    with open(chrom_file) as f:
        chrom_dict = collections.OrderedDict()
        for line in f:
            # *_ for other format like fadix file
            chrom, length, *_ = line.strip('\n').split('\t')
            if chrom in remove_chr_list:
                continue
            chrom_dict[chrom] = int(length)
    return chrom_dict
    pass


def _sliding_windows_with_step_size(window_size,step_size,ref,chrom_size = None,chrom_file = None):
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
        chrom_size_dict = _load_chrom_size_file(chrom_file) # user defined reference, especially for other species

    records = []
    for chrom, chrom_length in chrom_size_dict.items():
        bin_start = np.array(list(range(0, chrom_length, step_size)))
        bin_end = bin_start + window_size
        bin_end[np.where(bin_end > chrom_length)] = chrom_length
        chrom_df = pd.DataFrame(dict(bin_start=bin_start, bin_end=bin_end))
        chrom_df['chrom'] = chrom
        records.append(chrom_df)
    total_df = pd.concat(records)[['chrom', 'start', 'end']].reset_index(drop=True)
    return total_df


def convert_coo_to_csr_with_annotation(coo_mat,chrom,window_size=100000,step_size=5000,feature_file=None):
    coo_mat = sparse.load_npz(coo_mat)
    csr_mat = coo_mat.tocsr()
    n_cells = csr_mat.shape[1]
    # # 找到所有非空行的索引
    # non_empty_rows = np.unique(csr_mat.nonzero()[0])
    # # 从CSR矩阵中提取这些行的数据，创建一个新的矩阵
    # dense = csr_mat[non_empty_rows].todense()
    # chunk = pd.read_csv('D://Test/GSE56789/temp/chrY_chunk0000000.coo', delimiter=",", header=None).values
    # sorting_idx = np.lexsort((chunk[:, 1], chunk[:, 0]))
    # chunk[sorting_idx, 0]
    # data = (chunk[sorting_idx, 2] + 1).astype(int)
    # row = sorting_idx.astype(int)
    # col = chunk[sorting_idx, 1].astype(int)
    # coo_new = sparse.coo_matrix((data, (row, col)))
    # csr_new = coo_new.tocsr()
    feature_mtx = {}
    if feature_file is not None:
        for bed_entries in _load_feature(feature_file, chrom):
            chr, start, end, others = bed_entries
            mean = caculate_methylation_feature(csr_mat, start, end, n_cells)
            feature_name = '_'.join([chr, str(start), str(end)])
            feature_mtx[feature_name] = mean
    elif window_size !=0:
        feature_entries = _sliding_windows_with_step_size(window_size,step_size)
        # 遍历每一行
        for index, row in feature_entries.iterrows():
            chr = row['chrom']
            start = row['start']
            end = row['end']
            mean = caculate_methylation_feature(csr_mat, start, end, n_cells)
            feature_name = f"{chr}_{start}_{end}"
            feature_mtx[feature_name] = mean
    return feature_mtx


def _load_feature(feature_file,chrom,file_format = None):
    if file_format is None:
        file_format = feature_file[-3:]
    if file_format not in ['bed','txt','gtf','gff']:
        print (f"not support {file_format}  format right now, please check your file again")
    # need specify gtf gff loading method
    else:
        features_chrom = {}
        with open(feature_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                value = line.strip().split("\t")
                print(value)
                if value[0] in chrom:
                    yield value[0],int(value[1]),int(value[2]),value[3:] #返回染色体，起始终止位置和文件的其他列即feature的名称等



def _region_bin_annotation_chrom_coo_to_csr(csr_mtx,feature,chrom):
    """

    for coo_matrix, measure methylation level for given bins or genomic features

    :param mtx:
    :param feature:
    :param chrom:
    :return:
    """
    # 指定行范围
    start = 1
    end = 2

    mean_array = np.zeros(3)

    mean_list = []
    for col_idx in range(csr_mtx.shape[1]):
        col_data = csr_mtx[start - 1:end, col_idx].toarray()
        print(col_data)
        col_data = col_data[col_data != 0]
        col_data[col_data == -1] = 0  # 跳过值为-1的数据
        if col_data.size == 0:
            mean_array[col_idx] = np.nan
        else:
            mean_array[col_idx] = np.mean(col_data)

    print(mean_array)
    meth_levels_bins.append(np.nan)




    windows = np.arange(start, end, stepsize)
    smoothed_var = np.empty(windows.shape, dtype=np.float64)
    for i in prange(windows.shape[0]):
        pos = windows[i]
        mean_shrunk_resid = _calc_mean_shrunken_residuals(
            data_chrom,
            indices_chrom,
            indptr_chrom,
            pos - half_bw,
            pos + half_bw,
            smoothed_vals,
            n_cells,
            chrom_len,
        )
        smoothed_var[i] = np.nanvar(mean_shrunk_resid)
    return windows, smoothed_var




def _load_csr_into_anndata(csr_mtx,out_dir) -> Path:
    if os.path.exists(out_dir):
        return out_dir

    var_df = _generate_chrom_bin_bed_dataframe(chrom_size_path, bin_size)
    total_matrix = sparse.vstack([sparse.load_npz(path) for path in matrix_paths])

    adata = ad.AnnData(X=total_matrix,
                    obs=pd.DataFrame([], index=obs_names),
                    var=var_df[['chrom']],
                    uns=dict(bin_size=bin_size,
                             chrom_size_path=chrom_size_path,
                             mc_type=mc_type,
                             count_type=count_type,
                             step_size=step_size,
                             strandness=strandness))
    adata.write(output_path,
                compression=compression,
                compression_opts=compression_opts)
    return output_path

def import_bed_file(
        data_dir: Path,
        output_dir: Path,
        file: str = None,
        suffix: str = "bed",
        cpu: int = 10,
        pipeline : str = 'Bismark',
        chunksize: int = 100000,
        low_memory: bool = True,
        sorted_by_name: bool = True,
):
    """
    import single cov/bed methylation file into one csr sparse matrix for each chromsome,
    csr matrix will be saved at tmp_file and it is single-nucleotide
    :param data_dir:
    :param file:
        File name of the output h5ad file used to store the result. If provided,
        result will be saved to a backed AnnData, otherwise an in-memory AnnData
        is used.
    :param chrom:
    :param low_memory:
    :param sorted_by_name:
    :param blacklist:
        File name or a list of barcodes. If it is a file name, each line
        must contain a valid barcode. When provided, only barcodes in the whitelist
        will be retained.
    :param tempdir:
    :return:
    -------
    AnnData | ad.AnnData
        An annotated data matrix of shape `n_obs` x `n_vars`. Rows correspond to
        cells and columns to regions. If `file=None`, an in-memory AnnData will be
        returned, otherwise a backed AnnData is returned.
    """
    # make output_dir if not exits
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    adata = ad.AnnData() if file is None else ad.AnnData(filename=file)
    start_time = datetime.datetime.now()
    print(suffix)
    samples = find_files_with_suffix(data_dir, suffix)
    print(samples)
    n_cells = len(samples)
    chunksize = 10000
    chrom_size = {}
    processes = min(cpu, n_cells)
    print(processes)
    print('Import starting --- ---  ---')
    with Pool(processes=min(cpu, n_cells)) as pool:
        print('process ', n_cells, 'samples with', (min(cpu, n_cells)), 'cpu paraller')
        coo_results = []
        for cell_id, cov_file in enumerate(samples):
            # save cell id and name table
            # with open(output_dir + "file_list.csv", 'w') as f:
            #     f.write(f"{cell_n}\t{cov_file}\n")
            print(cov_file)
            coo_results.append(
                pool.apply_async(
                    _convert_cov_to_coo,
                    args=(
                        output_dir,
                        cov_file,
                        cell_id,
                        pipeline,
                    ),
                )
            )
        print("**** Basic statistics summary has been saved at " + output_dir)
        # conver to coo_matrix .npz format
        pool = mp.Pool(processes=cpu)
        coo_matrix_results=[]
        for res in coo_results:
            chrom_sizes = res.get()
            for chrom, size in chrom_sizes.items():
                if chrom not in chrom_size:
                    chrom_size[chrom] = 0
                chrom_size[chrom] = max(chrom_sizes[chrom], size)
                coo_matrix_results.append(
                    pool.apply_async(convert_coo_file_to_matrix, args=(output_dir,chr)))
            pool.close()
            pool.join()

        # aggregate all the cells
        print('Aggregate cells into adata')
        temp_dir = output_dir
        total_idx = []
        total_idy = []
        total_data = []
        for path in temp_dir.glob('*hdf'):
            idy = allc_path_idy[path.name[:-4]]
            pv = pd.read_hdf(path)
            if pv.size == 0:
                # no sig result
                continue
            total_idx.append(pv.index.values)
            total_idy.append([idy] * pv.size)
            total_data.append(pv.values)
            # the cell by region matrix
            (np.concatenate(total_data),
             (np.concatenate(total_idy), np.concatenate(total_idx))),
        region_bed = _read_region_bed(bed_path)
        _data = csr_matrix(
            shape=(len(allc_path_idy), region_bed.shape[0]))

        # save the data as anndata
        adata = ad.AnnData(_data,
                                obs=pd.DataFrame([], index=allc_paths.index),
                                var=region_bed)
        adata.X = adata.X.astype('float16')
        if str(output_prefix)[-5:] in {'.mcad', '.h5ad'}:
            output_h5ad_path = output_prefix
        else:
            output_h5ad_path = f'{output_prefix}.mcad'
        adata.write_h5ad(pathlib.Path(output_h5ad_path))

        cleanup = True
        # remove temp
        if cleanup:
            subprocess.run(['rm', '-rf', str(temp_dir)])
        return






