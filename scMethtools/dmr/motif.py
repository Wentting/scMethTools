import tempfile
import os
import pyfaidx
import tqdm
import numpy as np
import pandas as pd
from glob import glob
import subprocess
from scipy import sparse
import re
import scanpy as sc
from anndata import AnnData
from scipy.stats import fisher_exact

PWM_DIR = '/p300s/baoym_group/zongwt/zongwt/motifs' #需要修改为自己的PWM文件路径
PWM_suffix = 'jaspar'


def get_motif_hits(peak_sequences_file, num_peaks, pvalue_threshold = 0.00005):

    print('Scanning peaks for motif hits with p >= {} ...'.format(str(pvalue_threshold)))

    motifs_directory = PWM_DIR
    matrix_list = [
        os.path.basename(x) for x in
        glob(os.path.join(motifs_directory, '*.{}'.format(PWM_suffix)))
    ]

    command = ['moods-dna.py', 
        '-m', *matrix_list, 
        '-s', peak_sequences_file, 
        '-p', str(pvalue_threshold), 
        '--batch']

    print.info('Building motif background models ...')
    process = subprocess.Popen(
        ' '.join(command), 
        stdout=subprocess.PIPE, 
        shell=True, 
        stderr=subprocess.PIPE, 
        cwd = motifs_directory
    )

    motif_matrices = matrix_list 
    motif_idx_map = dict(zip(motif_matrices, np.arange(len(motif_matrices))))

    motif_indices, peak_indices, scores = [],[],[]
    i=0
    while process.stdout.readable():
        line = process.stdout.readline()

        if not line:
            break
        else:
            if i == 0:
                print.info('Starting scan ...')
            i+=1

            peak_num, motif, hit_pos, strand, score, site, snp = line.decode().strip().split(',')
            
            motif_indices.append(motif_idx_map[motif])
            peak_indices.append(peak_num)
            scores.append(float(score))

            if i%1000000 == 0:
                print.info('Found {} motif hits ...'.format(str(i)))

    if not process.poll() == 0:
        raise Exception('Error while scanning for motifs: ' + process.stderr.read().decode())

    print.info('Formatting hits matrix ...')
    return sparse.coo_matrix((scores, (peak_indices, motif_indices)), 
        shape = (num_peaks, len(motif_matrices))).tocsr().T.tocsr()
    


def motif_scan(adata, regions, genome, pvalue_threshold = 0.00005):
    """_summary_

    Args:
        regions (_type_): regions to scan for motifs, should be a numpy array with columns chrom, start, end
        genome (_type_): path to genome fasta file
        pvalue_threshold (float, optional): Adjusted p-value threshold for calling a motif hit within a region. Defaults to 0.00005.

    Returns:
    ----------------
    
    """
    #regions = featch_dmr(adata,group=None)
    temp_fasta = tempfile.NamedTemporaryFile(delete = False)
    temp_fasta_name = temp_fasta.name
    temp_fasta.close()

    try:

        get_region_sequences(regions, genome, temp_fasta_name)

        hits_matrix = get_motif_hits(temp_fasta_name, len(regions), pvalue_threshold = pvalue_threshold)

        ids, factors = list(zip(*list_motif_ids()))
        parsed_factor_names = list(map(_parse_motif_name, factors))
        
        adata.uns["motif_analysis"] = {
            "hits_matrix": hits_matrix,  # regions x motifs hits matrix
            "motif_names": parsed_factor_names,
            "motif_ids": ids,
            "motif_factors": factors
        }
        return ids, factors, parsed_factor_names, hits_matrix

    finally:
        os.remove(temp_fasta_name)
        
def _parse_motif_name(motif_name):
    return [x.upper() for x in re.split('[/::()-]', motif_name)][0]        
        
def get_motif_glob_str():
    return os.path.join(PWM_DIR, '*.{}'.format(PWM_suffix))

def list_motif_matrices():

    if not os.path.isdir(PWM_DIR):
        return []

    return list(glob(get_motif_glob_str()))
  
def list_motif_ids():
    return [os.path.basename(x).replace('.jaspar', '').split('_') for x in list_motif_matrices()]  

def get_region_sequences(peaks, genome_fasta, output_file):
    """
    Get sequences of regions
    """
    print('Get sequences of regions ...')
    fa = pyfaidx.Fasta(genome_fasta)

    with open(output_file, 'w') as f:
        for i, (chrom,start,end) in tqdm.tqdm(enumerate(peaks)):
            
            try:
                peak_sequence = fa[chrom][int(start) : int(end)].seq
            except KeyError:
                peak_sequence = 'N'*(int(end) - int(start))

            print('>{idx}\n{sequence}'.format(
                    idx = str(i),
                    sequence = peak_sequence.upper()
                ), file = f, end=  '\n')
    
    return None

def fetch_region(adata, chrom='chromosome', start='start', end='end'):
    """
    Fetch DMR region from genome and return a numpy array.
    """
    try:
        return adata.var[[chrom, start, end]].values  # 直接返回 ndarray
    except KeyError:
        raise Exception(f'Some of the columns {chrom}, {start}, {end} are not in .var')

def featch_dmr(adata,group=None,key_added='rank_genes_groups',chrom='chromosome', start='start', end='end',**kwargs):
    """
    Fetch DMR region from genome and return a numpy array.

    Args:
        adata (_type_): _description_
        group (_type_, optional): _description_. Defaults to None.
        key_added (str, optional): _description_. Defaults to 'rank_genes_groups'.
        chrom (str, optional): _description_. Defaults to 'chromosome'.
        start (str, optional): _description_. Defaults to 'start'.
        end (str, optional): _description_. Defaults to 'end'.

    Example:
    regions = featch_dmr(adata,group='0')
    """
    regions = sc.get.rank_genes_groups_df(adata, group=group, key=key_added,**kwargs)
    try:
        return adata.var.loc[regions['names'],[chrom, start, end]].values  # 直接返回 ndarray
    except KeyError:
        raise Exception(f'Some of the columns {chrom}, {start}, {end} are not in .var')
    
# #计算每一种细胞类型的差异DMR中富集的TF

def motif_enrichment(adata, key_added='rank_genes_groups', group=None):
    """
    calculate the enrichment of TF in DMRs of each cell type

    Args:
        hits_matrix (_type_): matrix of motif hits, must run motif_scan first
        adata (_type_): _description_
        key_add (str, optional): _description_. Defaults to 'rank_genes_groups'.
        group (_type_, optional): _description_. Defaults to None.
    
    ----------------
    Example:
    >>> motif_enrichment(adata, key_added='rank_genes_groups', group=['0','1'])
    >>> motif_enrichment(adata, key_added='rank_genes_groups')
    """
    if "motif_analysis" not in adata.uns:
        raise ValueError("Please run motif_scan first to generate hits_matrix.")
    # Check if key_add exists in adata.uns
    if key_added not in adata.uns:
        raise ValueError(f"'{key_added}' is not found in adata.uns. Differential analysis may not have been performed.")
    
    hits_matrix =  adata.uns["motif_analysis"]["hits_matrix"]
    # 从 adata.uns 中获取 motif ID 和 TF 名称
    motif_ids = adata.uns["motif_analysis"]["motif_ids"]
    tf_names = adata.uns["motif_analysis"]["motif_factors"]
    if group is None:
        group_list = list(adata.uns[key_added]['names'].dtype.names)
    else:
        # 确保 groups 是列表格式
        if isinstance(group, str):
            group_list = [group]  # 如果是字符串，转换成列表
    # Store results in dictionaries
    enrichment_results = {}
    for g in group_list:
        print(f'Calculating enrichment for group {g} ...')
        #计算group的dmr列表的富集情况
        dmr_names = adata.uns[key_added]['names'][g].tolist()
        # 有一种可能，存放的不是dmr,而是所有的region，他没有进行过滤
        # 获取在 `adata.var.index` 中的索引
        gene_idx = adata.var.index.get_indexer(dmr_names)
        # 过滤掉未匹配的基因（索引值为 -1）
        gene_idx = gene_idx[gene_idx >= 0]

        pvals, test_statistics = [], []
        for i in tqdm(range(hits_matrix.shape[0]), desc='Finding enrichments'):
        
            tf_hits = hits_matrix[i,:].indices #获取第 i 个 TF 在所有regions（（包括 DMR 和非 DMR）上的 motif 结合情况
            overlap = len(np.intersect1d(tf_hits, gene_idx)) # 计算该 TF motif 与 DMRs 的重叠数
            #构建2*2列联表
            module_only = len(gene_idx) - overlap
            tf_only = len(tf_hits) - overlap
            neither = hits_matrix.shape[1] - (overlap + module_only + tf_only)
            contingency_matrix = np.array([[overlap, module_only], [tf_only, neither]])
            #执行Fisher检验
            stat,pval = fisher_exact(contingency_matrix, alternative='greater')
            pvals.append(pval)
            #test_statistics.append(stat) 
            
        # Store results for the current group
        #enrichment_results[g] = {'pval': pvals, 'stat': test_statistics}
        # Convert results to DataFrame
    
    # Store results for the current group
    enrichment_results[g] = {'pval': pvals, 'stat': test_statistics}
    # Convert results to DataFrame
    # df_pvals = pd.DataFrame({g: enrichment_results[g]['pval'] for g in group_list}, index=tf_names)
    # df_stats = pd.DataFrame({g: enrichment_results[g]['stat'] for g in group_list}, index=tf_names)
    df_pvals = pd.DataFrame({g: enrichment_results[g]['pval'] for g in group_list}, index=tf_names)
    # Combine p-values and statistics into a single DataFrame
    #df_enrichment = pd.concat([df_pvals.add_suffix('_pval'), df_stats.add_suffix('_stat')], axis=1)
    # Save to adata.uns
    if 'motif_enrichment' not in adata.uns:
        adata.uns['motif_enrichment'] = {}  # 初始化为空字典或其他数据结构
    adata.uns['motif_enrichment'][key_added]= df_pvals
    return df_pvals


    
    