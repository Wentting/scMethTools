#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import List
from ..io import *
import pandas as pd
import numpy as np
import anndata as ad
from scipy import sparse
import datetime as datetime
import anndata as ad
import gzip
from .smooth import _smoothing_chrom,_smoothing_chrom_adaptive
from collections import namedtuple
import multiprocessing as mp
from multiprocessing import Pool
import muon as mu
import concurrent.futures
import scMethtools.logging as logg
import warnings
from anndata import ImplicitModificationWarning
from concurrent.futures import ProcessPoolExecutor
from functools import partial, lru_cache


# ignore division by 0 and division by NaN error
np.seterr(divide="ignore", invalid="ignore")

#在创建 AnnData 对象时仍然收到警告，可以选择忽略这些警告，前提是你确认它们不会影响你的数据处理流程
warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
def echo(*args, **kwargs):
    click.echo(*args, **kwargs, err=True)
    return


def secho(*args, **kwargs):
    click.secho(*args, **kwargs, err=True)
    return


def import_cells(input_dir: Path,
                 output_dir: Path,
                 context: str = "CG",
                 suffix: str = "bed",
                 cpu: int = 10,
                 pipeline: str = 'bsseeker2',
                 smooth: bool = True,
                 exclude_chrom: List[str] = None,
                 keep_tmp: bool =True):
    """
    import single-cell methylation file and save as sparse matrix
    if smooth is True, smooth the methylation matrix and caculate relative methylation level

    Args:
        input_dir (str): path to the directory containing the methylation files
        output_dir (str): _description_
        suffix (str, optional):  Defaults to "bed".
        cpu (int, optional): _description_. Defaults to 10.
        pipeline (str, optional): _description_. Defaults to 'bisseeker2'.
        smooth (bool, optional): whether conduct smooth steps. Defaults to True.
        exclude_chrom (List[str], optional): _description_. Defaults to None.
        keep_tmp (bool, optional): whether keep tmp coo files or not. Defaults to True.

    Returns:
        _type_: _description_
    """
    make_dir(output_dir) #parent outdir
    cells = find_files_with_suffix(input_dir, suffix)
    n_cells = len(cells)
    # thread default 10
    cpu = min(cpu, n_cells)
    logg.info( f"...import {n_cells} cells with {cpu} cpus")
    # adjust column order with different pipeline
    column_order = reorder_columns_by_index(pipeline)
    #tmp_path is coo file which can be deleted when keep_tmp is False
    stat_df, tmp_path = _import_cells_worker(cells, output_dir, context, cpu,*column_order)
    #npz dir
    #/xtdisk/methbank_baoym/zongwt/single/data/GSE97179/scbs/
    save_cells(tmp_path, output_dir, cpu=cpu, smooth=smooth, exclude_chrom=exclude_chrom, keep_tmp=keep_tmp)
    logg.info("...import cells done")  
    return stat_df, tmp_path
    

class CoverageFormat(
    namedtuple('CoverageFormat', ['chrom', 'pos', 'meth', 'umeth', 'context', 'coverage', 'sep', 'header'])):
    """Describes the columns in the coverage file.
    chrom, pos, meth, umeth, context, coverage, sep, header

    Params:
    chrom (int): 染色体列索引
    pos (int): 位置列索引
    meth (int): 甲基化列索引
    umeth (int): 未甲基化列索引
    context (int): 上下文列索引
    coverage (bool): 是否使用覆盖度计数
    sep (str): 分隔符
    header (bool): 是否有表头
   
    """
    def remove_chr_prefix(self):
        """Remove "chr" or "CHR" etc. from chrom."""
        return self._replace(chrom=self.chrom.lower().lstrip("chr"))

def _custom_format(format_string):
    """
    Create from user specified string. Adapted from scbs function
    
     Args:
        format_string: e.g chrom:pos:meth:coverage/unmeth:context:sep:header
    """
    try:
        parts = format_string.lower().split(":")
        if len(parts) != 7:
            raise Exception("Invalid number of ':'-separated values in custom input format")
         # 使用列表推导式简化索引转换
        indices = [int(p) - 1 for p in parts[:3]]
        chrom, pos, meth = indices
         # 使用正则表达式解析coverage/unmeth
        import re
        match = re.match(r'(\d+)([cuCU])', parts[3])
        if not match:
            raise Exception(
                "The 4th column of a custom input format must contain an integer and "
                "either 'c' for coverage or 'u' for unmethylated counts (e.g. '4c'), "
                f"but you provided '{format_string[3]}'."
            )
        umeth, info = match.groups()
        umeth = int(umeth) - 1 
        coverage = info == 'c'
        #umeth = int(format_string[3][0:-1]) - 1
        context = int(parts[4])-1
        sep = "\t" if parts[5].lower() in ("\\t", "tab", "t") else parts[5]
        header = bool(int(parts[6]))
        return chrom, pos, meth, umeth, context, coverage, sep, header
    except (ValueError, IndexError) as e:
        raise ValueError(f"Format parsing error : {str(e)}")


def reorder_columns_by_index(pipeline):
    """_summary_

    Args:
        pipeline (_str_): software pipeline used to generate the methylation coverage file
        software name or custom order string can be accepted
        software name: bismark, bsseeker2, methylpy
        custom order string: "1:2:3:4c:5:\t:0" (chrom:position:methylated_C:coverage(c)/unmethylated_C(u):context:sep:header) note: 1-based index                 
    Returns:
        _tuple_: the order of the columns in the coverage file
    """
    pipeline = pipeline.lower()
    # bismark format: chr1 10004 + 0 0 CHH CCC
    # bisseeker format: chr1 C 10060 CHH CT 1.00 1 1
    # methylpy format:
    # order: chr pos(1-based) meth unmeth context coverage(all-C) sep header
    format_orders = {
        'bismark': CoverageFormat(0, 1, 3, 4, 5, False, "\t", False),
        'bsseeker2': CoverageFormat(0, 2, 6, 7, 3, True, "\t", False),
        'bsseeker': CoverageFormat(0, 2, 6, 7, 3, True, "\t", False),
        'methylpy': CoverageFormat(0, 2, 6, 7, 3, True, "\t", False)
    }
    # 根据指定的格式调整列顺序
    if pipeline in format_orders:
        logg.info("## BED column format: " + pipeline)
        new_order = format_orders[pipeline]
        return new_order
    elif ":" in pipeline:
        logg.info("## BED column format:  Custom")
        new_order = _custom_format(pipeline)
        return new_order
    else:
        raise ValueError(f"Invalid format type or custom order {pipeline}.")


def read_bed_CG(bed_file, cell_id, chrom_col, pos_col, meth_col, umeth_col, context_col, cov, sep, header):
    """
    Reads BED files and generates methylation matrices.
    :param bed_file:
    :param cell_id:
    :param : column_order
    :return:
    """
    #chrom_col, pos_col, meth_col, umeth_col, context_col, cov, sep, header = reorder_columns_by_index(pipeline)
    reduced_cyt = {}
    cell_name = os.path.splitext(os.path.basename(bed_file))[0]
    stat_dict = {'cell_id': cell_id, 'cell_name': cell_name , 'total': 0, 'n_meth' : 0, 'n_total': 0}
    if bed_file.endswith('.gz'):
         with gzip.open(bed_file, 'rb') as sample:
            if header:
                # Skip the first line
                next(sample)
            for line in sample:
                if line.startswith('#'):
                    continue
                line = line.split(sep)
                chrom, pos, meth, status = line[chrom_col], int(line[pos_col]), int(line[meth_col]), line[context_col]
                if status in ['CGG', 'CGC', 'CGA', 'CGT', 'CGN' , 'CG']:
                    if cov:
                        cov = int(line[umeth_col])
                    else:
                        cov = int(line[umeth_col]) + meth
                    if chrom not in reduced_cyt:
                        reduced_cyt[chrom] = []
                    reduced_cyt[chrom].append((cell_id, pos, meth/cov, cov)) #meth/cov是甲基化的水平
                    stat_dict['total'] += 1
                    stat_dict['n_meth'] += meth
                    stat_dict['n_total'] += cov
    else:
        with open(bed_file) as sample:
            if header:
                # Skip the first line
                next(sample)
            for line in sample:
                if line.startswith('#'):
                    continue
                line = line.split(sep)
                chrom, pos, meth, status = line[chrom_col], int(line[pos_col]), int(line[meth_col]), line[context_col]
                if status in ['CGG', 'CGC', 'CGA', 'CGT', 'CGN' , 'CG']:
                    if cov:
                        cov = int(line[umeth_col])
                    else:
                        cov = int(line[umeth_col]) + meth
                    if chrom not in reduced_cyt:
                        reduced_cyt[chrom] = []
                    reduced_cyt[chrom].append((cell_id, pos, meth/cov, cov)) #meth/cov是甲基化的水平
                    stat_dict['total'] += 1
                    stat_dict['n_meth'] += meth
                    stat_dict['n_total'] += cov
    return reduced_cyt, stat_dict

def caculate_bins_mean(methylation_dict,annotation_dict):
    meth_levels_bins = []
    for chromosome, regions in annotation_dict.items():
        if chromosome in methylation_dict:
            for region_start, region_end in regions:
                methylation_values = [level for _, pos, level, cov in methylation_dict[chromosome] if region_start <= pos <= region_end and cov > 0] #可以加cov判断
                #加判断
                if methylation_values:
                    average_methylation = sum(methylation_values) / len(methylation_values)
                    meth_levels_bins.append(average_methylation)
                    #print(f"Chromosome: {chromosome}, Region: ({region_start}, {region_end}), Average Methylation: {average_methylation}, number: {len(methylation_values)}")
                else: #no cov
                    meth_levels_bins.append(np.nan)
        else:
            raise Exception(
                        f"{chromosome} is not in your data! "
                        "Please check again"
                    )
    return meth_levels_bins

           
        
def caculate_bins_residual(data_path,annotation_dict,half_bw=500):
    meth_shrunken_bins = []
    mean_bins = []
    region_mtx = []
    for chromosome, regions in annotation_dict.items():
            methylation_mat_path = os.path.join(data_path, f"{chromosome}_coo.npz")
            try:
                data_chrom = sparse.load_npz(methylation_mat_path)
            except FileNotFoundError:
                secho("Warning: ", fg="red", nl=False)
                echo(
                    f"Couldn't load methylation data for chromosome {chromosome} "
                )
                data_chrom = None
            chrom_len, n_cells = data_chrom.shape
            smooth_path = os.path.join(data_path,"smooth", f"{chromosome}_coo.csv")
            smooth_df = pd.read_csv(smooth_path, delimiter=",", header=None,names=['key', 'value'])
            smooth_dict = dict(zip(smooth_df['key'], smooth_df['value']))
            for region in regions:#annotation regions 
                if len(region) == 2:
                    region_start, region_end = region
                else:
                    region_start, region_end, *additional_info = region 
                result = _calc_mean_shrunken_residuals(
                    data_chrom,
                    region_start,
                    region_end,
                    smooth_dict,
                    n_cells,
                    chrom_len
                )
                meth_shrunken_bins.append(result[0])
                mean_bins.append(result[1])
                region_mtx.append(_calculate_region_statistics(chromosome,region_start,region_end,result[0],result[1]))
    return meth_shrunken_bins,mean_bins,region_mtx

def _caculate_bins_residual_chrom(npz_path,chromosome,regions,smooth_dict):
    meth_shrunken_bins = []
    mean_bins = []
    region_mtx = []  
    try:
        data_chrom = sparse.load_npz(os.path.join(npz_path, f"{chromosome}.npz"))
    except FileNotFoundError:
        secho("Warning: ", fg="red", nl=False)
        echo(
            f"Couldn't load methylation data for chromosome {chromosome} at {npz_path} "
        )
        data_chrom = None
    chrom_len, n_cells = data_chrom.shape
    for region in regions:#annotation regions 
        if len(region) == 2:
            region_start, region_end = region
        else:
            region_start, region_end, *additional_info = region
        #print("annotation-region",region_start,'-',region_end)
        #返回值不是元组，而是两个独立的列表。因此，在调用 _calc_mean_shrunken_residuals 函数时，只能使用一个变量来接收返回值
        #mean_shrunk_resid, mean_level = _calc_mean_shrunken_residuals
        result = _calc_mean_shrunken_residuals(
            data_chrom,
            region_start,
            region_end,
            smooth_dict,
            n_cells,
            chrom_len
        )
    
        meth_shrunken_bins.append(result[0])
        mean_bins.append(result[1])
        #统计var
        region_mtx.append(_calculate_region_statistics(chromosome,region_start,region_end,result[0],result[1]))
    return meth_shrunken_bins,mean_bins,region_mtx


def _calculate_region_statistics(chromosome,start,end,sr,mean):
    covered_cell = np.count_nonzero(~np.isnan(mean))
    if np.count_nonzero(~np.isnan(mean)) > 1:
        mean_var = np.nanvar(mean)
    else:
        mean_var = np.nan
    # 计算 sr 的方差，如果样本数量太小(都是na)，设置为 np.nan,不然会报错自由度不够
    if sr is None:
        return {
            'chromosome': chromosome,
            'start': start,
            'end': end,
            'covered_cell': covered_cell,
            'var': mean_var,
        }
    else:
        #sr_var = np.where(np.count_nonzero(~np.isnan(sr)) > 1, np.nanvar(sr), np.nan)
        if np.count_nonzero(~np.isnan(sr)) > 1:
            sr_var = np.nanvar(sr)
        else:
            sr_var = np.nan
        return {
            'chromosome': chromosome,
            'start': start,
            'end': end,
            'covered_cell': covered_cell,
            'var': mean_var,
            'sr_var': sr_var
        }
    
def sliding_windows(window_size, ref=None,step_size=None, chrom_file=None):
    """

    :param window_size:
    :param step_size:
    :param ref:
        A Genome object, providing gene annotation and chromosome sizes. ref should be one of `hg38,hg19,mm10,mm9,GRCh37,GRCh38,GRCm39`
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
        chrom_size_dict = load_chrom_size_file(chrom_file)  # user defined reference, especially for other species
    features_dict = {}
    for chrom, chrom_length in chrom_size_dict.items():
        bin_start = np.array(list(range(1, chrom_length, step_size)))
        bin_end = bin_start + window_size - 1
        bin_end[np.where(bin_end > chrom_length)] = chrom_length
        chrom_ranges = [(start, end) for start, end in zip(bin_start, bin_end)]
        features_dict[chrom]= chrom_ranges  
    return features_dict

def load_features(feature_file,format=None):
    #TO DO: load features from gtf file or gff file
    features_dict = {}
    if format==None:
        input_file_format = feature_file[-3:]
    if input_file_format=="bed":
        features_dict = read_annotation_bed(feature_file)
    # elif input_file_format=='gtf':
    #     features_dict = load_features_gtf(feature_file,feature_type="gene")
    # elif input_file_format=='gff':
    #     features_dict = load_features_gff(feature_file,feature_type="gene")
    # elif input_file_format=='csv':
    #     features_dict = load_features_csv(feature_file)
    else:
        raise ValueError("Unsupported file format")
    return features_dict    


def _calc_mean_shrunken_residuals(
    data_chrom,
    region_start,
    region_end,
    smoothed_vals,
    n_cells,
    chrom_len,
    shrinkage_factor=1,
):
    shrunken_resid = np.full(n_cells, np.nan, dtype=np.float32)
    mean_level = np.full(n_cells, np.nan, dtype=np.float32)
    start = max(region_start - 1, 0)
    end = min(region_end - 1, chrom_len)
    if start >= chrom_len or start >= end:
        return shrunken_resid, mean_level

    # 获取选定区域的行
    selected_rows = data_chrom[start:end + 1, :]
    if selected_rows.nnz == 0:
        return shrunken_resid, mean_level

    # 直接在稀疏矩阵上计算 cell_sums，跳过值为 -1 的元素
    cell_sums = np.zeros(n_cells, dtype=np.float32)
    n_obs = np.zeros(n_cells, dtype=np.int32)
    for start, end in zip(selected_rows.indptr[:-1], selected_rows.indptr[1:]):
        for index in range(start, end):
            col = selected_rows.indices[index]
            value = selected_rows.data[index]
            if value != -1:
                cell_sums[col] += value
            if value != 0: 
                n_obs[col] += 1
    # 计算 smooth_sum
    smooth_sum = np.zeros(n_cells, dtype=np.float32)
    for i in range(n_cells):
        if n_obs[i] > 0:
            non_zero_indices  = selected_rows[:, i].nonzero()[0] + region_start -1 # must -1
            smooth_sum[i] = sum(smoothed_vals.get(j, 0) for j in non_zero_indices)
    
    valid_obs_mask = n_obs > 0
    # 计算 shrunken_resid 和 mean_level
    shrunken_resid[valid_obs_mask] = (cell_sums[valid_obs_mask] - smooth_sum[valid_obs_mask]) / \
                                     (n_obs[valid_obs_mask] + shrinkage_factor)
    mean_level[valid_obs_mask] = np.round(cell_sums[valid_obs_mask] / n_obs[valid_obs_mask], 3)
    return shrunken_resid, mean_level

def _save_coo(reduced_cyt,context,data_path):
    for chromosome, values in reduced_cyt.items():
        coo_name = os.path.join(
                    data_path, f"{chromosome}_{context}.coo"
                )
        # 将列表转换为 JSON 格式的字符串
        lines = ['\t'.join(map(str, value)) for value in values]
        # 将 JSON 字符串写入文件
        with open(coo_name, 'a') as file:
            # 如果文件不为空，先写入一个换行符
            #if file.tell() != 0:
            file.write('\n')
            file.write('\n'.join(lines))
    del reduced_cyt
    
def save_chrom_dict_to_coo(chrom_dict, output_dir, context):
    for chrom, values in chrom_dict.items():
        arr = np.array(values)
        if arr.shape[0] == 0:
            continue
        coo = sparse.coo_matrix((arr[:, 2], (arr[:, 0], arr[:, 1])))
        sparse.save_npz(os.path.join(output_dir, f"{chrom}_{context}.npz"), coo)    

def read_pandas_in_chunks(cell_id,bed_file,data_path, chrom_col, pos_col, meth_col, umeth_col, context_col, cov, sep, header, chunk_size=100000):
    stat_dict = {'cell_id': cell_id, 'cell_name': os.path.basename(bed_file).split('.')[0], 'sites': 0, 'meth': 0, 'n_total': 0}
    reduced_cyt = {}

    use_cols = [chrom_col, pos_col, meth_col, umeth_col, context_col]
    names = ['chrom', 'pos', 'context', 'meth', 'umeth']
    dtype = {chrom_col: str, context_col: str}

    for chunk in pd.read_csv(
        bed_file,
        sep=sep,
        header=0 if header else None,
        compression='infer',
        usecols=use_cols,
        names=names if not header else None,
        dtype=dtype,
        chunksize=chunk_size,
        low_memory=False
    ):
        #这些判断会导致比episcanpy更慢，因为他完全没有做任何的判断
        chunk['chrom'] = 'chr' + chunk['chrom'].str.lstrip('chr')
        chunk = chunk[chunk['context'].isin(['CGG', 'CGC', 'CGA', 'CGT', 'CGN', 'CG', 'CpG'])]
        if chunk.empty:
            continue
        coverage = chunk['umeth'] + (0 if cov else chunk['meth'])
        meth_ratio = chunk['meth'] / coverage
        meth_value = np.where(meth_ratio >= 0.9, 1, np.where(meth_ratio <= 0.1, -1, 0))
        chunk['meth_value'] = meth_value
        for chrom, group in chunk.groupby('chrom'):
            reduced_cyt.setdefault(chrom, []).extend(zip(group['pos'],  group['meth'], group['umeth']))

        stat_dict['sites'] += len(chunk)
        stat_dict['meth'] += np.sum(meth_value == 1)
        stat_dict['n_total'] += coverage.sum()
    stat_dict['global_meth_level'] = stat_dict['meth'] / stat_dict['sites'] if stat_dict['sites'] else 0
    save_chrom_dict_to_coo(reduced_cyt, data_path, 'CG')#这样就变快了
    return stat_dict

def _read_CG_bed_save_coo(cell_id, bed_file, data_path, chrom_col, pos_col, meth_col, umeth_col, context_col, cov, sep, header):
    """
     cell_id : int
        Unique identifier for the cell.
    bed_file : str
        Path to the BED file to read (can be gzipped).
    data_path : str
        Path where to save the COO file.
    chrom_col : int, default=0
        Column index for chromosome information.
    pos_col : int, default=1
        Column index for position information.
    meth_col : int, default=4
        Column index for methylated read counts.
    umeth_col : int, default=5
        Column index for unmethylated read counts.
    context_col : int, default=6
        Column index for sequence context.
    cov : bool, default=False
        If True, coverage includes methylated reads.
    sep : str, default='\t'
        Delimiter used in the BED file.
    header : bool, default=False
        Whether the BED file has a header line.
    chunk_size : int, default=1000000
        Number of lines to process at once when using pandas.
    use_pandas : bool, default=True
        Whether to use pandas for reading (faster for large files).

    """
    reduced_cyt = {}
    cell_name = os.path.splitext(os.path.basename(bed_file))[0]
    stat_dict = {'cell_id': cell_id, 'cell_name': cell_name, 'sites': 0, 'meth': 0, 'n_total': 0, 'global_meth_level': 0}
    is_gz = bed_file.endswith('.gz')
    open_func = gzip.open if is_gz  else open
    with open_func(bed_file, 'rb' if is_gz else 'r') as sample:
         # 如果是 gz，转换为文本流
        if is_gz:
            sample = (line.decode('utf-8') for line in sample)
        if header:
            next(sample)
        for line in sample:
            if line.startswith('#'):
                continue
            line = line.split(sep)
            chrom = "chr" + line[chrom_col].lstrip("chr")  # 直接去除前缀，提高效率
            pos, meth, status = int(line[pos_col]), int(line[meth_col]), line[context_col]
            if status in {'CGG', 'CGC', 'CGA', 'CGT', 'CGN', 'CG', 'CpG'}:
                #cov为True，说明umeth列提供的是总覆盖度，否则umeth列提供的是未甲基化的覆盖度，所以需要相加
                coverage = int(line[umeth_col]) + (0 if cov else meth) #如果cov为True，则coverage = umeth + 0，否则coverage = meth
                meth_ratio = meth / coverage
                meth_value = 1 if meth_ratio >= 0.9 else -1 if meth_ratio <= 0.1 else 0
                reduced_cyt.setdefault(chrom, []).append((pos, cell_id, meth_value))
                stat_dict['sites'] += 1
                if meth_value == 1:
                    stat_dict['meth'] += 1
                stat_dict['n_total'] += coverage
    stat_dict['global_meth_level'] = stat_dict['meth'] / stat_dict['sites'] if stat_dict['sites'] else 0
    _save_coo(reduced_cyt, 'CG', data_path)
    del reduced_cyt
    return stat_dict

def _read_CHG_bed_save_coo(cell_id, bed_file, data_path, chrom_col, pos_col, meth_col, umeth_col, context_col, cov, sep, header):
    """
    Reads BED files and generates methylation matrices.

    """
    reduced_cyt = {}
    cell_name = os.path.splitext(os.path.basename(bed_file))[0]
    stat_dict = {'cell_id': cell_id, 'cell_name': cell_name, 'sites': 0, 'meth': 0, 'n_total': 0, 'global_meth_level': 0}

    open_func = gzip.open if bed_file.endswith('.gz') else open
    with open_func(bed_file, 'rb' if bed_file.endswith('.gz') else 'r') as sample:
        if header:
            next(sample)
        for line in sample:
            if bed_file.endswith('.gz'):
                line = line.decode('utf-8')

            if line.startswith('#'):
                continue

            line = line.split(sep)
            chrom = "chr" + line[chrom_col] if not line[chrom_col].startswith("chr") else line[chrom_col]
            pos, meth, status = int(line[pos_col]), int(line[meth_col]), line[context_col]
            if status not in {'CGG', 'CGC', 'CGA', 'CGT', 'CGN', 'CG', 'CpG'}:
                coverage = int(line[umeth_col]) + (0 if cov else meth)
                meth_value = 1 if meth / coverage >= 0.9 else -1 if meth / coverage <= 0.1 else 0
                reduced_cyt.setdefault(chrom, []).append((pos, cell_id, meth_value))
                stat_dict['sites'] += 1
                if meth_value == 1:
                    stat_dict['meth'] += 1
                stat_dict['n_total'] += coverage

    stat_dict['global_meth_level'] = stat_dict['meth'] / stat_dict['sites'] if stat_dict['sites'] else 0
    _save_coo(reduced_cyt,'nonCG', data_path)
    del reduced_cyt
    return stat_dict



def _process_partial(cell,context,**args):
    """ 
    read single cell bed file and save as coo file
    """
    cell_id = cell[0]
    bed_file = cell[1]
    
    if context == 'CG':
        return _read_CG_bed_save_coo(cell_id, bed_file, **args)
    else:
        return _read_CHG_bed_save_coo(cell_id, bed_file, **args)  


def _import_cells_worker(cells, out_dir, context, cpu,chrom_col, pos_col, meth_col, umeth_col, context_col, cov, sep, header):
    
    """
    import cells in parallel and save as coo file
    Params:
        cells: list of cells
        out_dir: output directory
        context: CG or nonCG
        cpu: number of threads
    Returns:
        _type_: _description_
    """
    stat_result = []
    # making temp directory for coo file
    data_path = os.path.join(out_dir, "tmp")   
    os.makedirs(data_path, exist_ok=True)
    logg.info(f"# Temp coo data writing to {data_path}")
    # TO DO: 现在不能重复导入同一个细胞的数据，不然会在coo文件中反复修改，导致值发生变化
    with ProcessPoolExecutor(max_workers=cpu) as executor:
        process_partial_param = partial(_process_partial, data_path=data_path, context=context, chrom_col=chrom_col, pos_col=pos_col,
                                  meth_col=meth_col, umeth_col=umeth_col, context_col=context_col, cov=cov, sep=sep, header=header)
        # 使用tqdm显示进度
        from tqdm import tqdm
        stat_result = list(tqdm(
            executor.map(process_partial_param, enumerate(cells)),
            total=len(cells)
        ))
    stat_df = pd.DataFrame(stat_result)
    stat_path = os.path.join(out_dir, "basic_stats.csv")
    stat_df.to_csv(stat_path, index=False)
    logg.info(f"## Basic summary writing to {stat_path} ...")
    return stat_df, data_path
    

def generate_scm(
        data_dir: str,
        output_dir: str,
        features: str,
        out_file: str = "scm",
        suffix: str = "bed",
        cpu: int = 10,
        pipeline: str = 'bsseeker2',
        relative: bool=True,
        smooth: bool=True,
        copy: bool=False
):
    """ For generate single cell methylation data object with anndata format
        adata.X(csr_matrix) with sample is row and feature is column
        adta.obs(stat_file)
        adta.var(feature_stat)

    Args:
        data_dir (str): directory for input mehtylation files
        output_dir (str): output_dir
        out_file (str, optional): output file name suffix. Defaults to None and the filename will set to .
        suffix (str, optional): input file suffix. Defaults to "bed".
        cpu (int, optional):  Defaults to 10.
        pipeline (str, optional):  Defaults to 'bsseeker2'.
        chunksize (int, optional):  Defaults to 100000.
    """
    # make output_dir if not exits
    make_dir(output_dir)
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    echo(f'... scm object generating at {start_time} ')
    #TO DO： delet some chromosomes in defined list
    stat_df, tmp_path = import_cells(data_dir,output_dir,suffix,cpu,pipeline)
    #TO DO: denovo find features
    adata = feature_to_scm(features,tmp_path,output_dir,out_file,meta=stat_df,cpu=cpu,relative=relative,smooth=smooth,copy=copy)
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    echo(f'... scm object finished and saved in {output_dir} at {end_time}')
    if copy:
        return adata
    
def features_to_scm(features,feature_names,output_dir,out_file,cpu=10,npz_path=None,meta_df=None,smooth=False,relative=True,copy=True):
    """
    generate anndata object with features methylation matrix

    Args:
        features (_list_): features name list generated by scm  
        feature_names (_list_): output features name list 
        meta_df (_type_): meta file 
        npz_path (_type_): _description_
        output_dir (_type_): _description_
        out_file (_type_): _description_
        cpu (int, optional): _description_. Defaults to 10.
        smooth (bool, optional): _description_. Defaults to True.
        relative (bool, optional): _description_. Defaults to True.
        copy (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    # 如果 features 是字符串，则转换为列表
    if isinstance(features, str):
        features = [features]
    # 如果 feature_names 是字符串，则转换为列表
    if isinstance(feature_names, str):
        feature_names = [feature_names]
    # 检查两个列表的长度是否相等
    if len(features) != len(feature_names):
        raise ValueError("The length of features and feature_names must be equal.")    
    make_dir(output_dir)
    meta_file_path = os.path.join(output_dir, f"basic_stats.csv")
    
    if os.path.exists(meta_file_path):
        try:
            meta_df = pd.read_csv(meta_file_path, sep=',')
        except Exception as e:
            logg.warn(f"No meta file found at {meta_file_path}, scm object will not contain meta information")
   
    modality_data = {}
    for feature_index, feature in enumerate(features):
        fn = feature_names[feature_index]
        logg.info(f"...generating matrix for feature {fn} with {cpu} cpus")
        #feature_to_scm(feature,output_dir,out_file,cpu,relative,smooth,copy=False,meta=None):
        adata = feature_to_scm(feature=feature,npz_path=npz_path,output_dir=output_dir,out_file=fn,cpu=cpu,relative=relative,smooth=smooth,copy=True,meta=meta_df)
        logg.info(f"...finish generating matrix for feature {fn}")
        mn = f"mod_{fn}"
        modality_data[mn]=adata
    mu_data = mu.MuData(modality_data)
    mu_data.meta = meta_df
    mu_data.uns['description'] = f"Description of the dataset:\n"
    mu_data.uns['description'] += f"Number of Modalities: {len(mu_data.mod)}\n"
    mu_data.uns['description'] += f"Modalities: {', '.join(mu_data.mod.keys())}\n"
    mu_data.uns['features'] = feature_names
    mu_data.write(os.path.join(
                        output_dir, f"{out_file}.h5mu"))
    logg.info(f"...scm object generating finish and save at {output_dir}/{out_file}.h5mu")
    if copy:
        return mu_data
    
    
def _matrix_npz(tmp_path,output_dir,smooth=True,exclude_chrom=None,keep_tmp=True):
    """
    单核

    Args:
        tmp_path (_str_): coo file tempory directory
        output_dir (_str_): output directory
        smooth (bool, optional): whether conduct smooth and calcuate relative methylation level. Defaults to True.
        exclude_chrom (_tuple_, optional):  Defaults to None.
        keep_tmp (bool, optional): keep coo file in temp directory or not. Defaults to True.
    """
    make_dir(output_dir)
    file_list = os.listdir(tmp_path)
    if exclude_chrom is None:
        exclude_chrom = []  # 如果 exclude_chrom 是 None，则将其设为一个空列表
    for file in file_list:
        if file.endswith(".coo"):
            chrom = os.path.basename(file).split('_')[0]
            if chrom in exclude_chrom:
                continue
            logg.info(f"... saving sparse csr matrix for {chrom}")
            coo_file = os.path.join(tmp_path,file)
            try:
                with open(coo_file, 'r') as file:
                    lines = file.readlines()
                    # 用于存储有效的行数据
                    valid_lines = []
                    for line in lines:
                        try:
                            row, col, val = map(int, line.split())
                            valid_lines.append((row, col, val))
                        except ValueError:
                            # 如果解析整数失败，跳过这行数据
                            pass
                # 将有效数据创建为稀疏矩阵
                if valid_lines:
                    rows, cols, data = zip(*valid_lines)
                    coo_matrix_loaded = sparse.coo_matrix((data, (rows, cols)))
                    csr_matrix_result = sparse.csr_matrix(coo_matrix_loaded)
                else:
                    print("No valid data to create the sparse matrix.")
                sparse.save_npz(os.path.join(output_dir, f"{chrom}.npz"),csr_matrix_result)
            except Exception as e:
                print(f"Error: CSR matrix saving has error occurred: {e}")
            if smooth:
                _smoothing_chrom(chrom,output_dir)
            #del smoothed_chrom
            if keep_tmp == False:
                os.remove(coo_file)

def matrix_npz_worker(file, tmp_path, npz_path, output_dir, smooth):
    #一个file是一个染色体
    chrom = os.path.basename(file).split('_')[0]
    coo_file = os.path.join(tmp_path, file)
    try:
        # 使用生成器读取和处理文件，减少内存使用
        with open(coo_file, 'r') as f:
            valid_lines = ((int(row), int(col), int(val)) for line in f if line.count('\t') == 2 for row, col, val in [line.split('\t')])
            rows, cols, data = zip(*valid_lines) if valid_lines else ([], [], [])
        if not data:
            return
        csr_matrix_result = sparse.csr_matrix((data, (rows, cols)))
        sparse.save_npz(os.path.join(npz_path, f"{chrom}.npz"), csr_matrix_result)
        if smooth:
            _smoothing_chrom(chrom, npz_path, output_dir)
    except Exception as e:
        logg.warnings(f"Error in processing {file}: {e}")
        

def save_cells(tmp_path, output_dir, cpu=10, smooth=True, exclude_chrom=None, keep_tmp=True):
    """
    read coo file and save csr matrix in npz format

    Args:
        tmp_path (_type_): _description_
        output_dir (_type_): _description_
        cpu (int, optional): _description_. Defaults to 10.
        smooth (bool, optional): _description_. Defaults to True.
        exclude_chrom (_type_, optional): _description_. Defaults to None.
        keep_tmp (bool, optional): _description_. Defaults to True.
    """
    make_dir(output_dir)
    file_list = [f for f in os.listdir(tmp_path) if f.endswith(".coo")]

    if exclude_chrom is None:
        exclude_chrom = []
        
    npz_path = os.path.join(output_dir, "data")
    make_dir(npz_path)
    logg.info(f"...saving sparse matrix at {npz_path}")
        
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu) as executor:
        futures = [executor.submit(matrix_npz_worker, file, tmp_path, npz_path, output_dir, smooth)
                   for file in file_list if os.path.basename(file).split('_')[0] not in exclude_chrom]
        concurrent.futures.wait(futures)
    if not keep_tmp:
        for file in file_list:
            os.remove(os.path.join(tmp_path, file))
    
#generate methylation matrix step after import and feature reading
def feature_to_scm(feature,output_dir,out_file,npz_path=None, cpu=1,relative=True,smooth=False,copy=False,meta=None):
    #feature_to_scm(feature=features[feature_index],output_dir=npz_path,out_file=fn,cpu=cpu,relative=relative,smooth=smooth,copy=True,meta=meta_df)
    """
    generate one anndata object with one feature methylation matrix

    Args:
        feature (_str_): a feature object
        output_dir (_path_) path to save the output file
        out_file (_str_): name of the output file
        npz_path (_path_, optional): path to npz file after import cells. Defaults to None.
        cpu (int, optional): _description_. Defaults to 1.
        relative (bool, optional): whether to conduct relative caculating. Defaults to True.
        smooth (bool, optional): whether to calculate smooth value for all positions. Defaults to False.
        copy (bool, optional): whether return object. Defaults to False.
        meta (_path_, optional): path to meta file. Defaults to None.

    Returns:
        _type_: _description_
        
    Example:
        w100k = scm.pp.feature_to_scm(feature=windows,output_dir="./out",npz_path=None,out_file="toy_100k",cpu=10,smooth=False,relative=True,copy=True)
    """
    make_dir(output_dir)
    npz_file_path = npz_path or os.path.join(output_dir, "data")
    cpu = min(cpu, len(feature))
    
    pool  = mp.Pool(processes=cpu)
    result = []
    #if feature is not None:
    for chrom,regions in feature.items():
        #ensure npz file exits
        if os.path.exists(npz_file_path):
            result.append(
                pool.apply_async(_feature_to_scm_parallel, args=(regions, chrom, output_dir, npz_file_path, relative, smooth))
            )
    pool.close()
    pool.join()
    # Combine the results into a final list using the zip function.
    final_result = [sum(combined, []) for combined in zip(*[res.get() for res in result])]
    logg.info(f"## anndata saved at {output_dir}")
    #result list: residual(optional),mean,var
    if len(final_result) == 2:
        m,var = final_result
        mean_csr = sparse.csr_matrix(m,dtype='float32')
        var_mtx = pd.DataFrame(var)
        var_mtx['index'] = var_mtx['chromosome'] + ':' + var_mtx['start'].astype(str) + '-' + var_mtx['end'].astype(str)
        var_mtx.set_index('index', inplace=True)
        if meta is not None:
            adata = ad.AnnData(mean_csr.T,obs=meta,var=var_mtx)
        else:
            adata = ad.AnnData(mean_csr.T,var=var_mtx)
        set_workdir(adata,workdir=output_dir)
        adata.write(os.path.join(output_dir, f"{out_file}.h5ad"))
        if copy:
            return adata                
    else:
        r,m,var = final_result
        mean_csr = sparse.csr_matrix(m,dtype='float32')
        residual_csr = sparse.csr_matrix(r,dtype='float32')
        var_mtx = pd.DataFrame(var)
        var_mtx['index'] = var_mtx['chromosome'] + ':' + var_mtx['start'].astype(str) + '-' + var_mtx['end'].astype(str)
        var_mtx.set_index('index', inplace=True)
        if meta is not None:
            adata = ad.AnnData(mean_csr.T,obs=meta,var=var_mtx)
        else:
            adata = ad.AnnData(mean_csr.T,var=var_mtx)
        adata.layers['relative'] = residual_csr.T
        set_workdir(adata,workdir=output_dir)
        adata.write(os.path.join(output_dir, f"{out_file}.h5ad"))
        if copy:
            return adata
  
        
def _feature_to_scm_parallel(regions,chrom,output_dir,npz_path,relative,smooth):
    """
    calculate methylation level and residual for each chromosome

    Args:
        regions (_type_): _description_
        chrom (_type_): _description_
        output_dir (_type_): _description_
        relative (_type_): _description_
        smooth (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        #echo(f"...saving single-cytosine sparse matrix for {chrom}")
        # csr_matrix_chrom = _save_npz(tmp_path,output_dir,chrom)
        #_save_npz(tmp_path,output_dir,chrom)     
        if relative: 
            if smooth:
                # smooth_dict = _smoothing_chrom(csr_matrix_chrom,chrom,output_dir) 
                smooth_dict = _smoothing_chrom(chrom,npz_path,output_dir) #parent outdir
            else:
                csv_file_path = os.path.join(os.path.join(output_dir,"smooth"), f"{chrom}.csv")
                df = pd.read_csv(csv_file_path, header=None, names=["pos", "smooth_val"])
                # Convert DataFrame to a dictionary
                smooth_dict = dict(zip(df["pos"], df["smooth_val"]))   
            logg.info(f"...caculate {chrom} relative methylation level")    
            r,m,var = _caculate_bins_residual_chrom(npz_path,chrom,regions,smooth_dict)
            return r,m,var
        else:
            logg.info(f"...caculate {chrom} methylation level")  
            m,var = _caculate_bins_mean_chrom(npz_path,chrom,regions)
    except Exception as e:
        print(f"An error occurred: {e}")

            
def _caculate_bins_mean_chrom(npz_path,chrom,regions):
    mean_bins = []
    region_mtx = []   
    methylation_mat_path = os.path.join(npz_path, f"{chrom}.npz")
    try:
        data_chrom = sparse.load_npz(methylation_mat_path)
    except FileNotFoundError:
        secho("Warning: ", fg="red", nl=False)
        echo(
            f"Couldn't load methylation data for chromosome {chrom} "
        )
        data_chrom = None
    chrom_len, n_cells = data_chrom.shape
    for region_start, region_end in regions: #annotation regions 
        #print("annotation-region",region_start,'-',region_end)
        #返回值不是元组，而是两个独立的列表。因此，在调用 _calc_mean_shrunken_residuals 函数时，只能使用一个变量来接收返回值
        #mean_shrunk_resid, mean_level = _calc_mean_shrunken_residuals
        result = _calculate_mean_level(
            data_chrom,
            region_start,
            region_end,
            n_cells,
            chrom_len,
        )
        mean_bins.append(result)
        #统计var
        region_mtx.append(_calculate_region_statistics(chrom,region_start,region_end,sr=None,mean=result))
    return mean_bins,region_mtx

def _calculate_mean_level(data_chrom,region_start, region_end,n_cells, chrom_len,):
    mean_level = np.full(n_cells, np.nan)
    start = max(region_start - 1, 0)
    end = min(region_end - 1, chrom_len)
    if start >= chrom_len or start >= end:
        return mean_level
    selected_rows = data_chrom[start:end + 1, :].toarray() #已测试numpy方法更快
    if selected_rows.size == 0:
        return mean_level
    cell_sums = np.sum(np.where(selected_rows == 1, selected_rows, 0), axis=0)
    n_obs = np.sum(selected_rows != 0, axis=0)
    nonzero_mask = n_obs > 0
    mean_level[nonzero_mask] = np.round(cell_sums[nonzero_mask] / n_obs[nonzero_mask], 3)

    return mean_level
 