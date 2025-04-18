#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import List
from scMethtools.io import *
import pandas as pd
import numpy as np
import anndata as ad
from scipy import sparse
import datetime as datetime
import anndata as ad
from scMethtools.preprocessing.adaptive_smooth import _smoothing_chrom_fast
import concurrent.futures
import scMethtools.logging as logg
import warnings
from anndata import ImplicitModificationWarning
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from scMethtools.preprocessing._format import reorder_columns_by_index
import multiprocessing as mp
from scMethtools.preprocessing._methylation_level import _feature_methylation_chrom
import muon as mu


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

def save_chrom_dict_to_coo(chrom_dict, output_dir, context):
    for chromosome, values in chrom_dict.items():
        coo_name = os.path.join(
                    output_dir, f"{chromosome}_{context}.coo"
                )
        # 将列表转换为 JSON 格式的字符串
        lines = ['\t'.join(map(str, value)) for value in values]
        # 将 JSON 字符串写入文件
        with open(coo_name, 'a') as file:
            # 如果文件不为空，先写入一个换行符
            #if file.tell() != 0:
            file.write('\n')
            file.write('\n'.join(lines))
    del chrom_dict

def read_pandas_in_chunks_CG(cell_id,bed_file,out_dir, chrom_col, pos_col, meth_col, umeth_col, context_col, cov, sep, header, chunk_size=100000):
    stat_dict = {'cell_id': cell_id, 'cell_name': os.path.basename(bed_file).split('.')[0], 'sites': 0, 'meth': 0, 'n_total': 0}
    reduced_cyt = {}
    use_cols = [chrom_col, pos_col, meth_col, umeth_col, context_col]
    names = ['chrom','pos', 'context', 'meth', 'umeth']

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
        #print(chunk.head())
        #这些判断会导致比episcanpy更慢，因为episcanpy完全没有做任何的判断
        chunk['chrom'] = 'chr' + chunk['chrom'].str.lstrip('chr')
        chunk = chunk[chunk['context'].str.startswith('CG')]
        if chunk.empty:
            continue
        coverage = chunk['umeth'] + (0 if cov else chunk['meth'])
        meth_ratio = chunk['meth'] / coverage
        meth_value = np.where(meth_ratio >= 0.9, 1, np.where(meth_ratio <= 0.1, -1, 0))
        chunk['meth_value'] = meth_value
        for chrom, group in chunk.groupby('chrom'):
            reduced_cyt.setdefault(chrom, []).extend(zip(group['pos'],  [cell_id] * len(group), group['meth_value']))

        stat_dict['sites'] += len(chunk)
        stat_dict['meth'] += np.sum(meth_value == 1)
        stat_dict['n_total'] += coverage.sum()
        #save_chrom_dict_to_coo(reduced_cyt, output_dir, cell_id) 把save写到这里就慢多了
    stat_dict['global_meth_level'] = stat_dict['meth'] / stat_dict['sites'] if stat_dict['sites'] else 0
    save_chrom_dict_to_coo(reduced_cyt, out_dir, 'CG')#这样就变快了
    #这个coo文件到底是为了干什么
    #应该是为了算smooth平滑值，要不然没有办法算所有文件的平滑值
    return stat_dict


def read_pandas_in_chunks_nonCG(cell_id, bed_file, out_dir, chrom_col, pos_col, meth_col, umeth_col, context_col, cov, sep, header, chunk_size=100000):
    stat_dict = {'cell_id': cell_id, 'cell_name': os.path.basename(bed_file).split('.')[0], 'sites': 0, 'meth': 0, 'n_total': 0}
    reduced_cyt = {}
    use_cols = [chrom_col, pos_col, meth_col, umeth_col, context_col]
    names = ['chrom', 'pos', 'meth', 'umeth','context']
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
        # print(chunk.head())
        chunk['chrom'] = 'chr' + chunk['chrom'].str.lstrip('chr')
        chunk = chunk[~chunk['context'].isin(['CGG', 'CGC', 'CGA', 'CGT', 'CGN', 'CG', 'CpG'])]
        if chunk.empty:
            continue
        coverage = chunk['umeth'] + (0 if cov else chunk['meth'])
        meth_ratio = chunk['meth'] / coverage
        meth_value = np.where(meth_ratio >= 0.9, 1, np.where(meth_ratio <= 0.1, -1, 0))
        chunk['meth_value'] = meth_value
        for chrom, group in chunk.groupby('chrom'):
            reduced_cyt.setdefault(chrom, []).extend(zip(group['pos'],  [cell_id] * len(group), group['meth_value']))

        stat_dict['sites'] += len(chunk)
        stat_dict['meth'] += np.sum(meth_value == 1)
        stat_dict['n_total'] += coverage.sum()
    stat_dict['global_meth_level'] = stat_dict['meth'] / stat_dict['sites'] if stat_dict['sites'] else 0
    save_chrom_dict_to_coo(reduced_cyt, out_dir, 'nonCG')#这样就变快了
    return stat_dict

def _process_partial(cell,context,data_path,**args):
    """ 
    read single cell bed file and save as coo file
    """
    logg.info(f"...Reading cells in {context} context")  
    cell_id = cell[0]
    bed_file = cell[1]   
    if context == 'CG':
        return read_pandas_in_chunks_CG(cell_id, bed_file, data_path,**args)
    else:
        return read_pandas_in_chunks_nonCG(cell_id, bed_file,data_path,**args)

def _import_worker(cells,context,out_dir, cpu, chrom_col, pos_col, meth_col, umeth_col, context_col, cov, sep, header):
    
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
    #print(f"reading ... {len(cells)}")
    # TO DO: 现在不能重复导入同一个细胞的数据，不然会在coo文件中反复修改，导致值发生变化
    with ProcessPoolExecutor(max_workers=cpu) as executor:
        from tqdm import tqdm
        # 定义部分参数
        process_partial_param = partial(
            _process_partial,
            data_path=data_path, 
            context=context, 
            chrom_col=chrom_col, 
            pos_col=pos_col,
            meth_col=meth_col, 
            umeth_col=umeth_col, 
            context_col=context_col, 
            cov=cov, 
            sep=sep, 
            header=header
        )

         # 使用tqdm显示进度
        from tqdm import tqdm
        stat_result = list(
            executor.map(process_partial_param, enumerate(cells),
            
        ))
    # 将结果转化为DataFrame
    stat_df = pd.DataFrame(stat_result)
    stat_df.to_csv(os.path.join(out_dir, "basic_stats.csv"), index=False)
    logg.info(f"## Basic summary writing to {data_path} ...")
    return stat_df, data_path

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
    stat_df, tmp_path = _import_worker(cells, context, output_dir, cpu, *column_order)
    #npz dir
    save_cells(tmp_path, output_dir, cpu=cpu, smooth=smooth, exclude_chrom=exclude_chrom, keep_tmp=keep_tmp)
    logg.info("...import cells done")  
    return stat_df, tmp_path
                   
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
        logg.info(f"...saving sparse matrix at {npz_path}")
        if smooth:
            _smoothing_chrom_fast(chrom, npz_path, output_dir)
    except Exception as e:
        logg.error(f"Error in processing {file}: {e}")
def save_cells(tmp_path, output_dir, cpu=10, smooth=False, exclude_chrom=None, keep_tmp=True):
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
        
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu) as executor:
        futures = [executor.submit(matrix_npz_worker, file, tmp_path, npz_path, output_dir, smooth)
                   for file in file_list if os.path.basename(file).split('_')[0] not in exclude_chrom]
        concurrent.futures.wait(futures)
    if not keep_tmp:
        for file in file_list:
            os.remove(os.path.join(tmp_path, file))

#generate methylation matrix step after import and feature reading
def feature_to_scm(feature,output_dir,out_file,npz_path=None, cpu=10,relative=True,smooth=False,copy=False,meta=None):
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
                pool.apply_async(_feature_methylation_chrom, args=(regions, chrom, output_dir, npz_file_path, relative, smooth))
            )
    pool.close()
    pool.join()
    # Combine the results into a final list using the zip function.
    final_result = [sum(combined, []) for combined in zip(*[res.get() for res in result])]
    #result list: residual(optional),mean,var
    adata = construct_anndata(final_result, meta=meta, output_dir=output_dir, out_file=out_file, copy=copy)
    logg.info(f"## anndata saved at {output_dir}")
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

        
def construct_anndata(final_result, meta=None, output_dir="./", out_file="output", copy=False, set_X="mean"):
        # 解析 final_result
    if len(final_result) == 2:
        mean_matrix, var = final_result
        residual_matrix = None
    else:
        residual_matrix, mean_matrix, var = final_result

    # 构建 mean 和 residual 的稀疏矩阵
    mean_csr = sparse.csr_matrix(mean_matrix, dtype='float32')
    residual_csr = sparse.csr_matrix(residual_matrix, dtype='float32') if residual_matrix is not None else None

    # 构建 var（features）
    var_df = pd.DataFrame(var)
    var_df['index'] = var_df['chromosome'] + ':' + var_df['start'].astype(str) + '-' + var_df['end'].astype(str)
    var_df.set_index('index', inplace=True)

    # 构建 AnnData，注意 .X 默认使用 mean 或 residual
    X = residual_csr.T if (set_X == 'residual' and residual_csr is not None) else mean_csr.T
    adata = ad.AnnData(X=X, obs=meta, var=var_df) if meta is not None else ad.AnnData(X=X, var=var_df)

    # 加入其他 layer
    adata.layers['mean'] = mean_csr.T
    if residual_csr is not None:
        adata.layers['relative'] = residual_csr.T
    # 设置 .raw 为 mean（可选）
    adata.raw = adata.copy()

    set_workdir(adata, workdir=output_dir)
    if not out_file.endswith('.h5ad'):
        out_file += '.h5ad'

    adata.write(os.path.join(output_dir, out_file))

    # 返回
    if copy:
        return adata
    
            
            
            
            
