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

def read_pandas_in_chunks_CG(cell_id, bed_file, out_dir, chrom_col, pos_col, meth_col, 
                             umeth_col=None, context_col=None, cov=False, sep='\t', header=False, chunk_size=100000):
    """
    Read methylation file in chunks and process CG sites.
    Compatible with both full coverage and simplified 3-column format.
    """
    stat_dict = {'cell_id': cell_id, 'cell_name': os.path.basename(bed_file).split('.')[0], 
                 'sites': 0, 'meth': 0, 'n_total': 0}
    reduced_cyt = {}

    # columns to read
    use_cols = [chrom_col, pos_col, meth_col]
    names = ['chrom', 'pos', 'meth']
    if umeth_col is not None:
        use_cols.append(umeth_col)
        names.append('umeth')
    if context_col is not None:
        use_cols.append(context_col)
        names.append('context')

    dtype = {chrom_col: str}
    if context_col is not None:
        dtype[context_col] = str

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
        # standardize chrom
        chunk['chrom'] = 'chr' + chunk['chrom'].str.lstrip('chr')

        # filter CG context if available
        if context_col is not None:
            chunk = chunk[chunk['context'].str.startswith('CG')]
        if chunk.empty:
            continue

        # compute meth_ratio
        if umeth_col is None:  # simplified 3-column format
            meth_ratio = chunk['meth']
        else:  # full coverage format
            coverage = chunk['umeth'] + (0 if not cov else chunk['meth'])
            meth_ratio = chunk['meth'] / coverage

        # assign meth_value
        meth_value = np.where(meth_ratio >= 0.9, 1, np.where(meth_ratio <= 0.1, -1, 0))
        chunk['meth_value'] = meth_value

        for chrom, group in chunk.groupby('chrom'):
            reduced_cyt.setdefault(chrom, []).extend(zip(group['pos'], [cell_id]*len(group), group['meth_value']))

        stat_dict['sites'] += len(chunk)
        stat_dict['meth'] += np.sum(meth_value == 1)
        if umeth_col is None:
            stat_dict['n_total'] += len(chunk)  # simplified: each site counts as 1
        else:
            stat_dict['n_total'] += coverage.sum()

    stat_dict['global_meth_level'] = stat_dict['meth'] / stat_dict['sites'] if stat_dict['sites'] else 0
    save_chrom_dict_to_coo(reduced_cyt, out_dir, 'CG')
    return stat_dict



def read_pandas_in_chunks_nonCG(cell_id, bed_file, out_dir, chrom_col, pos_col, meth_col,
                                umeth_col=None, context_col=None, cov=False, sep='\t', header=False,
                                chunk_size=100000):
    """
    Read methylation file in chunks and process non-CG sites.
    Compatible with both full coverage and simplified 3-column format.
    """
    stat_dict = {'cell_id': cell_id, 'cell_name': os.path.basename(bed_file).split('.')[0],
                 'sites': 0, 'meth': 0, 'n_total': 0}
    reduced_cyt = {}

    # columns to read
    use_cols = [chrom_col, pos_col, meth_col]
    names = ['chrom', 'pos', 'meth']
    if umeth_col is not None:
        use_cols.append(umeth_col)
        names.append('umeth')
    if context_col is not None:
        use_cols.append(context_col)
        names.append('context')

    dtype = {chrom_col: str}
    if context_col is not None:
        dtype[context_col] = str

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
        # standardize chrom
        chunk['chrom'] = 'chr' + chunk['chrom'].str.lstrip('chr')

        # filter non-CG sites if context available
        if context_col is not None:
            chunk = chunk[~chunk['context'].isin(['CGG', 'CGC', 'CGA', 'CGT', 'CGN', 'CG', 'CpG'])]
        if chunk.empty:
            continue

        # compute meth_ratio
        if umeth_col is None:  # simplified 3-column format
            meth_ratio = chunk['meth']
        else:  # full coverage format
            coverage = chunk['umeth'] + (0 if not cov else chunk['meth'])
            meth_ratio = chunk['meth'] / coverage

        # assign meth_value
        meth_value = np.where(meth_ratio >= 0.9, 1, np.where(meth_ratio <= 0.1, -1, 0))
        chunk['meth_value'] = meth_value

        for chrom, group in chunk.groupby('chrom'):
            reduced_cyt.setdefault(chrom, []).extend(zip(group['pos'], [cell_id] * len(group), group['meth_value']))

        stat_dict['sites'] += len(chunk)
        stat_dict['meth'] += np.sum(meth_value == 1)
        if umeth_col is None:
            stat_dict['n_total'] += len(chunk)  # simplified: each site counts as 1
        else:
            stat_dict['n_total'] += coverage.sum()

    stat_dict['global_meth_level'] = stat_dict['meth'] / stat_dict['sites'] if stat_dict['sites'] else 0
    save_chrom_dict_to_coo(reduced_cyt, out_dir, 'nonCG')
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
        # 1. 先估算文件大小，决定处理方式
        file_size = os.path.getsize(coo_file)
        large_file = file_size > 1e9  # 如果文件超过1GB，采用分块处理
        
        if large_file:
            print(f"{chrom} with {file_size}, using _process_large")
            # 对大文件采用分块处理
            return _process_large_coo_file(coo_file, chrom, npz_path, output_dir, smooth)
        else:
            print(f"{chrom} with {file_size}, using _process_small")
            # 小文件直接处理
            return _process_small_coo_file(coo_file, chrom, npz_path, output_dir, smooth)
            
    except Exception as e:
        import traceback
        logg.error(f"Error in processing {file}: {e}")
        logg.error(traceback.format_exc())
        # 返回错误状态，方便外部检查
        return (chrom, False, str(e))

def _process_small_coo_file(coo_file, chrom, npz_path, output_dir, smooth):
    """处理较小的COO文件，可以一次性读入内存"""
    try:
        # 使用numpy直接读取文件更高效
        import numpy as np
        from scipy import sparse
        
        # 读取并解析数据
        data_array = np.loadtxt(coo_file, delimiter='\t', dtype=np.int32)
        if data_array.size == 0 or len(data_array.shape) < 2:
            logg.warning(f"No valid data in {coo_file}")
            return (chrom, True, "No data")
            
        # 构建稀疏矩阵
        rows, cols, data = data_array[:, 0], data_array[:, 1], data_array[:, 2]
        csr_matrix = sparse.csr_matrix((data, (rows, cols)))
        
        # 保存矩阵
        npz_file = os.path.join(npz_path, f"{chrom}.npz")
        sparse.save_npz(npz_file, csr_matrix)
        logg.info(f"Saved matrix for {chrom}")
        
        # 清理内存
        del data_array, rows, cols, data, csr_matrix
        
        # 执行平滑处理
        if smooth:
            _smoothing_chrom_fast(chrom, npz_path, output_dir)
        
        return (chrom, True, "Success")
    except Exception as e:
        logg.error(f"Error in _process_small_coo_file for {chrom}: {e}")
        return (chrom, False, str(e))

def _process_large_coo_file(coo_file, chrom, npz_path, output_dir, smooth):
    """分块处理大型COO文件"""
    try:
        import numpy as np
        from scipy import sparse
        import gc
        
        # 先计算矩阵维度
        max_row = 0
        max_col = 0
        with open(coo_file, 'r') as f:
            for i, line in enumerate(f):
                if i % 1000000 == 0:  # 每百万行记录一次进度
                    logg.info(f"Scanning dimensions for {chrom}: {i} lines processed")
                if line.count('\t') == 2:
                    try:
                        row, col, _ = map(int, line.strip().split('\t'))
                        max_row = max(max_row, row + 1)
                        max_col = max(max_col, col + 1)
                    except ValueError:
                        continue
        
        logg.info(f"Matrix dimensions for {chrom}: {max_row} x {max_col}")
        
        # 分块读取文件
        chunk_size = 5000000  # 每块行数
        temp_files = []
        
        with open(coo_file, 'r') as f:
            chunk_counter = 0
            while True:
                # 读取一个数据块
                rows = []
                cols = []
                data = []
                
                for _ in range(chunk_size):
                    line = f.readline()
                    if not line:
                        break
                    if line.count('\t') == 2:
                        try:
                            row, col, val = map(int, line.strip().split('\t'))
                            rows.append(row)
                            cols.append(col)
                            data.append(val)
                        except ValueError:
                            continue
                
                if not rows:  # 文件读完了
                    break
                    
                # 为当前块创建稀疏矩阵
                chunk_matrix = sparse.csr_matrix(
                    (data, (rows, cols)), 
                    shape=(max_row, max_col)
                )
                
                # 保存临时文件
                temp_file = os.path.join(npz_path, f"{chrom}_chunk_{chunk_counter}.npz")
                sparse.save_npz(temp_file, chunk_matrix)
                temp_files.append(temp_file)
                
                logg.info(f"Saved chunk {chunk_counter} for {chrom}")
                chunk_counter += 1
                
                # 清理内存
                del rows, cols, data, chunk_matrix
                gc.collect()
        
        # 合并所有临时文件
        if temp_files:
            # 对于非常大的矩阵，分批合并
            logg.info(f"Merging {len(temp_files)} chunks for {chrom}")
            final_matrix = None
            
            for i, temp_file in enumerate(temp_files):
                if i % 5 == 0:  # 记录进度
                    logg.info(f"Merging chunk {i}/{len(temp_files)} for {chrom}")
                    
                chunk = sparse.load_npz(temp_file)
                
                if final_matrix is None:
                    final_matrix = chunk
                else:
                    # 合并稀疏矩阵
                    final_matrix = final_matrix + chunk
                
                # 删除临时文件并清理内存
                os.remove(temp_file)
                del chunk
                gc.collect()
            
            # 保存最终矩阵
            if final_matrix is not None:
                npz_file = os.path.join(npz_path, f"{chrom}.npz")
                sparse.save_npz(npz_file, final_matrix)
                logg.info(f"Saved final matrix for {chrom}")
                
                # 清理内存
                del final_matrix
                gc.collect()
            
            # 执行平滑处理
            if smooth:
                _smoothing_chrom_fast(chrom, npz_path, output_dir)
            
            return (chrom, True, f"Success with {chunk_counter} chunks")
        else:
            logg.warning(f"No valid data for {chrom}")
            return (chrom, True, "No data")
            
    except Exception as e:
        import traceback
        logg.error(f"Error in _process_large_coo_file for {chrom}: {e}")
        logg.error(traceback.format_exc())
        return (chrom, False, str(e))
        
def save_cells(tmp_path, output_dir, cpu=10, smooth=True, exclude_chrom=None, keep_tmp=True):
    """
    read coo file and save csr matrix in npz format
    """

    

    file_list = [f for f in os.listdir(tmp_path) if f.endswith(".coo")]

    if exclude_chrom is None:
        exclude_chrom = []
        
    # 筛选需要处理的文件
    to_process = []
    for file in file_list:
        chrom = os.path.basename(file).split('_')[0]
        if chrom not in exclude_chrom:
            to_process.append(file)    
         
    npz_path = os.path.join(output_dir, "data")
    if not os.path.exists(npz_path):
        os.makedirs(npz_path)

    
    logg.info(f"...saving sparse matrix at {npz_path}")
    
    #根据染色体文件大小排序，先处理小文件
    to_process.sort(key=lambda f: os.path.getsize(os.path.join(tmp_path, f)))
    # 限制同时运行的进程数量 - 对于大文件，单次只允许1-2个进程
    concurrent_limit = max(1, min(5, cpu // 3))  # 最多使用CPU数的1/3，最少1个，最多3个
    # 限制并发进程数，避免内存溢出
    # 估算每个进程可能需要的内存
    logg.info(f"Using {concurrent_limit} concurrent processes based on available memory")
    
    
    # 记录结果
    results = {}
    
    # 批量处理文件
    with ProcessPoolExecutor(max_workers=concurrent_limit) as executor:
        # 先处理小文件
        futures = {}
        for file in to_process:
            chrom = os.path.basename(file).split('_')[0]
            # 检查结果文件是否已存在
            if os.path.exists(os.path.join(npz_path, f"{chrom}.npz")):
                logg.info(f"Skipping {chrom}, output already exists")
                results[chrom] = True
                continue
                
            futures[executor.submit(matrix_npz_worker, file, tmp_path, npz_path, output_dir, smooth)] = chrom
        
        # 处理结果
        for future in concurrent.futures.as_completed(futures):
            chrom = futures[future]
            try:
                success = future.result()
                results[chrom] = success
                if success:
                    logg.info(f"Successfully processed {chrom}")
                else:
                    logg.error(f"Failed to process {chrom}")
            except Exception as e:
                logg.error(f"Exception processing {chrom}: {e}")
                results[chrom] = False
    
    # 报告结果
    processed = sum(1 for s in results.values() if s)
    failed = sum(1 for s in results.values() if not s)
    logg.info(f"Processing complete: {processed} succeeded, {failed} failed")
    
    # 列出失败的染色体
    if failed > 0:
        failed_chroms = [c for c, s in results.items() if not s]
        logg.error(f"Failed chromosomes: {', '.join(failed_chroms)}")
    
    # 清理临时文件
    if not keep_tmp:
        for file in file_list:
            try:
                os.remove(os.path.join(tmp_path, file))
            except Exception as e:
                logg.warning(f"Failed to remove temp file {file}: {e}")

#generate methylation matrix step after import and feature reading
def feature_to_scm(feature,output_dir,out_file,npz_path=None, cpu=10,relative=True,smooth=False,copy=False,meta=None):
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
    
     # safely collect results, handle failed tasks
    successful_results = []
    failed_chroms = []
    for i, res in enumerate(result):
        chrom_name = list(feature.keys())[i]  # get corresponding chromosome name
        try:
            # try to get result, set timeout to avoid infinite waiting
            chrom_result = res.get(timeout=None)
            if chrom_result is None:
                logg.warn(f"Chromosome {chrom} returned None, skipping.")
                continue
            successful_results.append(chrom_result)
            logg.info(f"Successfully processed chromosome: {chrom_name}")
        except Exception as e:
            # record failed chromosome and error information
            failed_chroms.append(chrom_name)
            logg.warn(f"Failed to process chromosome {chrom_name}: {str(e)}")
            continue
    
    # check if there are successful results
    if not successful_results:
        error_msg = f"All chromosomes failed to process. Failed chromosomes: {failed_chroms}"
        logg.error(error_msg)
        raise RuntimeError(error_msg)
    
    # if there are some failed, record warning information
    if failed_chroms:
        logg.warn(f"The following chromosomes failed and were excluded from analysis: {failed_chroms}")
        logg.info(f"Successfully processed {len(successful_results)} out of {len(result)} chromosomes")
    
    # Combine the successful results into a final list using the zip function.
    # Combine the successful results into a final list
    try:
        meth_list, mean_list, var_list = [], [], []

        for r in successful_results:
            # unpack safely
            if len(r) == 3:
                meth, mean, var = r
            elif len(r) == 2:
                meth = None  # 占位，没有 residual
                mean, var = r
            else:
                raise ValueError(f"Unexpected result length: {len(r)}")

            # extend lists safely
            if meth is not None:
                meth_list.extend(meth)
            mean_list.extend(mean)
            var_list.append(var)  # var 一定存在

        # 合并 var
        var_df = pd.concat(var_list)

        final_result = [meth_list, mean_list, var_df]

        # construct anndata
        adata = construct_anndata(final_result, meta=meta, output_dir=output_dir, out_file=out_file, copy=copy)
        logg.info(f"## anndata saved at {output_dir}")
        return adata

    except Exception as e:
        logg.error(f"Failed to construct anndata: {str(e)}")
        raise




    
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
        mean_matrix, var_df = final_result
        residual_matrix = None
    else:
        residual_matrix, mean_matrix, var_df = final_result

    mean_csr = sparse.csr_matrix(mean_matrix, dtype='float32')
    residual_csr = sparse.csr_matrix(residual_matrix, dtype='float32') if residual_matrix is not None else None

    X = residual_csr.T if (set_X == 'residual' and residual_csr is not None) else mean_csr.T
    adata = ad.AnnData(X=X, obs=meta, var=var_df) if meta is not None else ad.AnnData(X=X, var=var_df)

    # adata.layers['mean'] = mean_csr.T
    if residual_csr is not None:
        adata.layers['relative'] = residual_csr.T
    #adata.raw = adata.copy()

    set_workdir(adata, workdir=output_dir)
    if not out_file.endswith('.h5ad'):
        out_file += '.h5ad'
    adata.write(os.path.join(output_dir, out_file))

    if copy:
        return adata
    
    del mean_csr, residual_csr
    return None
    
            
            
            
            
