#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年08月21日

"""
import os, fnmatch, sys
import numpy as np
import pandas as pd
import click
import collections
import gzip
import zipfile
from pybedtools import BedTool


def echo(*args, **kwargs):
    click.echo(*args, **kwargs, err=True)
    return


def secho(*args, **kwargs):
    click.secho(*args, **kwargs, err=True)
    return

def title(*args, **kwargs):
    click.echo(click.style(*args, **kwargs),err=False)


def make_dir(dirname):
    """Create directory `dirname` if non-existing.

    Parameters
    ----------
    dirname: str
        Path of directory to be created.

    Returns
    -------
    bool
        `True`, if directory did not exist and was created.
    """
    if os.path.exists(dirname):
        return False
    else:
        os.makedirs(dirname)
        return True
    
def set_workdir(adata,workdir=None):
    """set working dir
    workdir: `Path`, optional (default: None)
        Working directory. If it's not specified, a folder named 'scm_result' will be created under the current directory
    """
    if(workdir==None):
        workdir = os.path.join(os.getcwd(), 'scm_result')
        echo("Using default working directory.")
    if(not os.path.exists(workdir)):
        os.makedirs(workdir)
    adata.uns['workdir'] = workdir
    echo(f"Saving results in: {workdir}")
    
# def find_files_with_suffix(path, suffix):
#     matching_files = []
#     #for root, dirs, files in os.walk(path):
#     for root, files in os.listdir(path):
#         for file in files:
#             if file.endswith(suffix):
#                 matching_files.append(os.path.join(root, file))
#     return matching_files
def find_files_with_suffix(path, suffix):
    matching_files = []
    # 使用os.listdir遍历指定目录下的文件和子目录
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        # 只处理文件，忽略子目录
        if os.path.isfile(item_path) and item.endswith(suffix):
            matching_files.append(item_path)
    return matching_files

def find_files_with_pattern(path, pattern):
    list_files = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        list_files += [os.path.join(dirpath, file) for file in filenames]

    # Print the files
    list_dataset = []
    for elem in list_files:
        if fnmatch.fnmatch(elem, pattern):
            list_dataset.append(elem)
    return list_dataset

def iter_lines(file):
    """ Helper for iterating only nonempty lines without line breaks"""
    for line in file:
        line = line.rstrip('\r\n')
        if line:
            yield line

def format_chromosome(chro):
    """Format chromosome name.

    Makes name upper case, e.g. 'mt' -> 'MT' and removes 'chr',
    e.g. 'chr1' -> '1'.
    """
    return chro.str.upper().str.replace('^CHR', '')

def is_binary(values):
    """Check if values in array `values` are binary, i.e. zero or one."""
    return ~np.any((values > 0) & (values < 1))

def read_meta(
        path: str,
        meta_name: str,
        sample_prefix: str = None,
) -> str:
    prefix = "sample" if sample_prefix is None else sample_prefix
    sample_list = pd.read_csv(path + '/' + meta_name, sep=',').loc[:, prefix].unique().tolist()
    return sample_list

def read_annotation_bed(annotation_file, chrom_col=None, start_col=None, end_col=None, name_col=None, keep_other_columns=True):
    """
    Read a BED file and optionally specify columns for chromosome, start, end, and name.
    
    Args:
        annotation_file (str): Path to the BED file
        chrom_col (int, optional): Index of chromosome column, default 0
        start_col (int, optional): Index of start position column, default 1
        end_col (int, optional): Index of end position column, default 2
        name_col (int, optional): Index of annotation name column, default None (generate "chr:start-end")
        keep_other_columns (bool, optional): Whether to keep other columns (not used currently)
        
    Returns:
        dict: {chromosome: set((start, end, name))}
    """
    annotation_dict = {}
    
    # Default column indices
    chrom_col = 0 if chrom_col is None else chrom_col
    start_col = 1 if start_col is None else start_col
    end_col = 2 if end_col is None else end_col

    with open(annotation_file, 'r') as features:
        for line in features:
            if not line.strip():
                continue
            ar = line.strip().split()
            
            try:
                chromosome = ar[chrom_col]
                start = int(ar[start_col])
                end = int(ar[end_col])
            except (IndexError, ValueError):
                raise ValueError(f"Cannot parse line: {line.strip()}")
            
            # Name column
            if name_col is not None and len(ar) > name_col:
                name = ar[name_col]
            else:
                name = f"{chromosome}:{start}-{end}"
            
            if chromosome not in annotation_dict:
                annotation_dict[chromosome] = set()
            
            annotation_dict[chromosome].add((start, end, name))
    
    return annotation_dict


def read_bed(filename, sort=False, usecols=[0, 1, 2], *args, **kwargs):
    """
    Load data from .bed file and return as a dictionary of chromosome data.

    :param filename: str
        The path to the .bed file.
    :param sort: bool, optional
        Whether to sort the data by chromo, start, and end columns.
    :param usecols: list of int, optional
        Columns to read from the .bed file.
    :param args: Additional arguments passed to pd.read_table.
    :param kwargs: Additional keyword arguments passed to pd.read_table.
    :return: dict
        Dictionary with chromosome as key and a set of tuples (start, end, other_column) as value.
    """
    # Read the .bed file with the specified columns and data types
    d = pd.read_table(filename, sep='\t', header=None, usecols=usecols, dtype={0: str}, *args, **kwargs)
    d.columns = range(d.shape[1])
    d.rename(columns={0: 'chromo', 1: 'start', 2: 'end'}, inplace=True)

    # If there are more columns (beyond the first 3), include them in the result
    if len(usecols) > 3:
        other_columns = d.columns[3:]
        for col in other_columns:
            d[col] = d[col].astype(str)

    # Sort the dataframe if required
    if sort:
        d.sort_values(['chromo', 'start', 'end'], inplace=True)

    # Create a dictionary where the key is the chromo and value is a set of tuples (start, end, other_column)
    result = {}
    for _, row in d.iterrows():
        chromo = row['chromo']
        start, end = row['start'], row['end']
        other_values = tuple(row[3:])  # Get other columns if any
        if chromo not in result:
            result[chromo] = set()
        result[chromo].add((start, end) + other_values)

    return result


def load_chrom_size_file(chrom_file,remove_chr_list=None):
    with open(chrom_file) as f:
        chrom_dict = collections.OrderedDict()
        for line in f:
            # *_ for other format like fadix file
            chrom, length, *_ = line.strip('\n').split('\t')
            if remove_chr_list is not None and chrom in remove_chr_list:
                continue               
            chrom_dict[chrom] = int(length)
    return chrom_dict

def parse_gtf(gtf, gene_type='protein_coding'):
    """_summary_

    Args:
        gtf : absolute path to gtf file

    Returns:
        BedTool object
    """
    print("... Loading gene references")
    genes = BedTool(gtf)
    # 从基因参考集中筛选出蛋白编码基因
    coding = []
    for x in genes:
        if 'chr' not in x[0]:
        # 如果第一列中不包含'chr'，则添加'chr'到第一列
            x[0] = 'chr' + x[0]
        if 'gene_type' in x[-1]:
            if np.logical_and(x['gene_type'] == gene_type, x[2] == 'gene'):
                coding.append(x)
        if 'gene_biotype' in x[-1]:
            if np.logical_and(x['gene_biotype'] == gene_type, x[2] == 'gene'):
                coding.append(x)
    print("... Done")
    return coding

def read_gff3(filename, sort=False, usecols=[0, 3, 4], *args, **kwargs):
    """
    load data from .gff3 file which is tab delimited txt file
    :param filename: str
        file to load
    :param sort:
    :param usecols:
    :param args:
    :param kwargs:
    :return:
    """
    d = pd.read_table(filename, header=None, usecols=usecols, *args, **kwargs)
    d.columns = range(d.shape[1])
    d.rename(columns={0: 'chromo', 3: 'start', 4: 'end'}, inplace=True)
    if sort:
        d.sort(['chromo', 'start', 'end'], inplace=True)
    return d


def read_gtf(file_path,exclude_chromosomes=None, feature_type="gene", gene_type='protein_coding',tss_up=0, tss_down=0):
    """gtf file can be download from ensembl  
    'seqname','source','feature','start','end','score','strand','attribute','other'
    Args:
        file_path (_type_): _description_
        feature_type (str): None or string ('transcript', 'exon', 'gene')
        promoter_distance(int) : tss_up=0, tss_down=0
    Returns:
        _type_: _description_
    """
    annotation_dict = {}
    # 获取文件扩展名
    _, ext = os.path.splitext(file_path)

    # 打开文件的逻辑写在一行中
    with (zipfile.ZipFile(file_path, 'r').open(zip_info) if ext == '.zip' and (zip_info := zipfile.ZipFile(file_path, 'r').infolist()[0]).filename.endswith('.txt') else gzip.open(file_path, 'rt') if ext == '.gz' else open(file_path, 'r')) as file:
    #with open(file_path, 'r') as file:
        i=0
        for line in file:
            if line.startswith('#'):
                continue  
            fields = line.strip().split('\t')
            feature_type = fields[2]
            i +=1
            if i==1:
                print(fields)
            if feature_type == 'gene':
                attributes = dict(item.strip().split(' ') for item in fields[8].split(';') if item.strip())
                gene_type = attributes.get('gene_biotype', attributes.get('gene_type', '')).replace('"', '')
                gene_name = attributes.get('gene_name', '').replace('"', '')
                gene_id = attributes.get('gene_id', '').replace('"', '')
                chrom = 'chr' + fields[0]
                start = int(fields[3])
                end = int(fields[4])
                strand = fields[6]
                if i==1:
                    print(gene_type,gene_name)
                if gene_type == 'protein_coding':
                    #print(gene_type)
                    # 计算启动子的起始位置和终止位置
                    if tss_up != 0:
                        if strand == '+':
                            promoter_start = max(1, start - tss_up)
                            promoter_end = start + tss_down
                        elif strand == '-':
                            promoter_start = end - tss_down
                            promoter_end = end + tss_up
                        else:
                            raise ValueError(f"Unknown strand: {strand}")

                        annotation_dict.setdefault(chrom, []).append((promoter_start, promoter_end, gene_name, gene_id,strand))
                    else:
                        annotation_dict.setdefault(chrom, []).append((start, end, gene_name, gene_id,strand))
     
    # 如果需要排除的染色体列表为空，直接返回原始字典
    if not exclude_chromosomes:
        return annotation_dict
    
    # 使用字典推导式，排除掉不需要的染色体
    filtered_dict = {chrom: regions for chrom, regions in annotation_dict.items() if chrom not in exclude_chromosomes}
    return filtered_dict
