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

def read_annotation_bed(annotation_file,keep_other_columns=True):
    annotation_dict = {}
    with open(annotation_file, 'r') as features:
        for line in features:
            ar = line.strip().split()
            chromosome, start, end = ar[0], int(ar[1]), int(ar[2])
            if chromosome not in annotation_dict:
                annotation_dict[chromosome] = set()
            annotation_dict[chromosome].add((start, end))
    return annotation_dict

def read_bed(filename, sort=False, usecols=[0, 1, 2], *args, **kwargs):
    """
    load data from .bed file which is tab delimited txt file
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
    d.rename(columns={0: 'chromo', 1: 'start', 2: 'end'}, inplace=True)
    if sort:
        d.sort(['chromo', 'start', 'end'], inplace=True)
    return d

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


