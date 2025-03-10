#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年08月23日

"""

import os
import numpy as np
import logging
import pandas as pd
from pybedtools import BedTool
import pyranges as pr

class _GenomicRegions:

    def __init__(self, gtf_file, feature_order):
        if isinstance(gtf_file, str):
            gr = pr.read_gtf(gtf_file)
        elif isinstance(gtf_file, pr.PyRanges):
            # 如果 gtf_file 是 PyRanges 对象，直接赋值
            gr = gtf_file
        else:
            raise ValueError('invalid gtf_file argument')
       
        # 检查和添加前缀
        self.gr = self._add_chromosome_prefix(gr, prefix='chr')
        #self.gr = gr[gr.gene_biotype == 'protein_coding']
        
        self.feature_order = feature_order
        
    def _add_chromosome_prefix(self, gr, prefix='chr'):
         #标准化，ensemle的注释文件跟gencode不一样。
        feature_mapping = {
            '5UTR': 'five_prime_utr',
            '3UTR': 'three_prime_utr'
        }
        df = gr.df
        if not df['Chromosome'].str.startswith(prefix).all():
            df['Chromosome'] = df['Chromosome'].apply(lambda x: f"{prefix}{x}" if not x.startswith(prefix) else x)
        df['Feature'] = df['Feature'].replace(feature_mapping)
        return pr.PyRanges(df)
    

    def features(self, features: set = None) -> pr.PyRanges:
        # 提取指定的特征（如启动子、内含子等），并进行注释
        features = features or set(self.feature_order)

        gr = self.gr[self.gr.Feature.isin(features)]
        df = gr.df

        if 'intron' in features:
            # 如果包含内含子特征，则生成内含子注释
            gr_introns = self.gr.features.introns()
            df = pd.concat([df, gr_introns.df])
        if 'promoter' in features:
            #如果包含promoter，需要自己根据gene特征进行生成
            additional_regions = generate_promoter_and_downstream(self.gr)
            df =  pd.concat([df, additional_regions.df], ignore_index=True)
        # 核心列
        _core_columns = ['Chromosome', 'Start', 'End', 'Strand']
        df = df[[*_core_columns, 'Feature', 'gene_id', 'gene_name']]

        #df['annotated_site'] = self.annotated_site(df)

        return df

    def annotate(self, gr,features=None):
        """_summary_

        Args:
            gr (_pyranges object_): _description_
            features (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        df_gtf = self.features(features)
        # dmr 通常是没有链特征的
        # df_gtf = df_gtf[df_gtf['Strand'].isin(gr.Strand.cat.categories)]
        gr_gtf = pr.PyRanges(df_gtf, int64=True)
            
        gr_ann = gr.join(
            gr_gtf, strandedness=False, how='left') \
            .drop(['Start_b', 'End_b', 'Strand'])
        df = gr_ann.df.drop_duplicates()
        
        # 进行注释，使用 PyRanges 的 join 方法
        df['Feature'] = df['Feature'].replace('-1', 'Distal intergenic')
        df['gene_id'] = df['gene_id'].replace('-1', '')
        df['gene_name'] = df['gene_name'].replace('-1', '')

        #_tqdm_pandas_gr()

        # if gene_id defined overlap 如果一个区域匹配到多个基因的不同区域要进行去重
        _core_columns = ['Chromosome', 'Start', 'End']
        # 检查 'group' 列是否存在，如果存在则加入核心列
        if 'group' in df.columns:
            _core_columns.insert(0, 'group')
            
        df = df.groupby(_core_columns, observed=True) \
               .apply(self._agg_annotation_gene) \
               .reset_index(drop=True)
        
        #注释intergenic区域
        #df = self.intergenic_genes(df)
        return df

    def intergenic_genes(self, df):
        '''
        Assign `gene_id` and `gene_name` to intergenic clusters.
        '''
        genes = pd.Series(range(sum(df['Feature'] == 'intergenic'))) \
                  .astype('str').add_prefix('intergenic_').index

        df.loc[df['Feature'] == 'intergenic', 'gene_id'] = genes.tolist()
        df.loc[df['Feature'] == 'intergenic', 'gene_name'] = genes.tolist()

        return df

    def _agg_annotation_gene(self, df):
        """
        主要目的是在注释过程中，根据 feature_order 中的特征优先级，选择适当的注释特征。
        如果某个基因具有多个特征（如多个外显子或 UTR.该方法将根据特定规则聚合这些特征。
        """
        feature_order = self.feature_order

        if df.shape[0] == 1:
            return df

        for feature in feature_order:
            _df = df[df['Feature'] == feature]

            if _df.shape[0] == 0:
                continue

            if _df.shape[0] == 1:
                return _df
            # 将 gene_id 和 gene_name 合并到一个新的列中, 当区域匹配到多个基因的相同转录元件区域时候，不去重而是进行合并
            _df = _df.copy()
            _df['gene_id'] = ','.join(_df['gene_id'].astype(str).unique())
            _df['gene_name'] = ','.join(_df['gene_name'].astype(str).unique())

            return _df.drop_duplicates(subset='gene_id')

class DmrGenomicRegions(_GenomicRegions):
    '''
    Annotate differentially methylation regions based on the genomics features.

    Args:
      gtf_file: Annotation file overlap against.

    Examples:
      Annotation of dmr features in pyranges format:
      
      >>> regions = DmrGenomicRegions('hg38.gtf')
      >>> dmr_gr = df_to_pyranges(dmr_df)
      >>> DmrGenomic.annotate(dmr_gr)
     
    '''

    def __init__(self, gtf_file):
        feature_order = ['promoter(1-5kb)','promoter(<=1kb)','five_prime_utr','three_prime_utr', 'exon',
                         'intron', 'downstream','gene','promoter']
        #之后注释会按照这个顺序进行去重，因此顺序很重要
        super().__init__(gtf_file, feature_order)
        
def generate_promoter_and_downstream(gr):
    # 提取基因的起始位点（TSS）
    
    tss = gr[(gr.Feature == 'transcript') | (gr.Feature == 'gene')].df
    tss['Start'] = tss['Start'] - 1  # 转换为0-based
    tss['End'] = tss['Start'] + 1

    # 根据链方向生成不同的启动子和下游区域
    def create_promoter(row, start_offset, end_offset):
        if row.Strand == '+':
            start = row.Start - end_offset
            end = row.Start - start_offset
        else:
            start = row.End + start_offset
            end = row.End + end_offset
        return pd.Series([row.Chromosome, start, end, row.Strand, row.gene_id, row.gene_name])

    # 创建不同距离的启动子区域
    promoter_1_5kb = tss.apply(lambda row: create_promoter(row, 1000, 5000), axis=1)
    promoter_lt_1kb = tss.apply(lambda row: create_promoter(row, 0, 1000), axis=1)
    
    # 创建下游区域
    def create_downstream(row, start_offset, end_offset):
        if row.Strand == '+':
            start = row.End + start_offset
            end = row.End + end_offset
        else:
            start = row.Start - end_offset
            end = row.Start - start_offset
        return pd.Series([row.Chromosome, start, end, row.Strand, row.gene_id, row.gene_name])
    
    downstream = tss.apply(lambda row: create_downstream(row, 0, 1000), axis=1)

    # 合并所有区域并添加特征标签
    regions = pd.concat([promoter_1_5kb, promoter_lt_1kb, downstream], ignore_index=True)
    regions.columns = ['Chromosome', 'Start', 'End', 'Strand', 'gene_id', 'gene_name']
    regions['Feature'] = ['promoter(1-5kb)'] * len(promoter_1_5kb) + \
                         ['promoter(<=1kb)'] * len(promoter_lt_1kb) + \
                         ['downstream'] * len(downstream)

    return pr.PyRanges(regions)

# dataframe to pyranges object
def df_to_pyranges(
    df, 
    start_col="start",
    end_col="end",
    chr_col="chromosome",
    start_slop=0,
    end_slop=0,
):
    """
    Convert a pandas DataFrame to a PyRanges object.
    do annotation for the dataframe

    Args:
        df (_type_): _description_
        start_col (str, optional): _description_. Defaults to "start".
        end_col (str, optional): _description_. Defaults to "end".
        chr_col (str, optional): _description_. Defaults to "chromosome".
        start_slop (int, optional): _description_. Defaults to 0.
        end_slop (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    if chr_col not in df.columns:
        # 拆分 'names' 列成 'Chromosome', 'Start', 'End'
        df[['Chromosome', 'Start', 'End']] = df['names'].str.split('[:-]', expand=True)
        # 将 'Start' 和 'End' 转换为整数类型
        df['Start'] = df['Start'].astype(int)- start_slop
        df['End'] = df['End'].astype(int) + end_slop
    else:
        df["Chromosome"] = df[chr_col]
        df["Start"] = df[start_col] - start_slop
        df["End"] = df[end_col] + end_slop
    return pr.PyRanges(df)


def parse_gtf(gtf,gene_type="protein_coding"):
    print("... Loading gene references")
    genes = BedTool(gtf)
    # 从基因参考集中筛选出蛋白编码基因
    coding = []
    print(f"... {gene_type} gene will be annotated.")
    for x in genes:
        if 'chr' not in x[0]:
        # 如果第一列中不包含'chr'，则添加'chr'到第一列
            x[0] = 'chr' + x[0]
        if gene_type == 'all':
            if x[2] == 'gene':
                coding.append(x)
        else:
            if 'gene_type' in x[-1]:
                if np.logical_and(x['gene_type'] == gene_type, x[2] == 'gene'):
                    coding.append(x)
            if 'gene_biotype' in x[-1]:
                if np.logical_and(x['gene_biotype'] == gene_type, x[2] == 'gene'):
                    coding.append(x)
    coding = BedTool(coding)
    print("... Done")
    return coding



def annotation(adata,ref,gene_type='protein_coding'):
    """Annotate bins with gene information. Now not work on windows system because of the pybedtools package.
    Args:
        adata : scm object
        ref : path to the gene reference file in gtf format.
        gene_type (str, optional): Defaults to 'protein_coding'.
    """
    coding = ref
    print("... Loading regions info")
    # 将 var中存储的feature的 bin 信息转换为 BedTool 对象
    bins = np.stack([adata.var['chromosome'], adata.var['start'], adata.var['end'],range(len(adata.var['chromosome']))]).T.tolist()
    bins = BedTool(bins)
    # 对于多个基因位于相同距离的情况，仅为距离最近的基因进行标注
    print("... Overlapping genes with regions")
    annot = bins.sort().closest(coding.sort(), d=True, t='first')
    
    # 创建一个字典用于存储注释信息
    an_dict = {'Accession': [], 'Gene': [], 'Distance': [], 'Loc':[]}
    
    for x in annot:
        # 将注释信息拆分为键值对并存储到字典中
        feats = x[12].split(';')[:-1]    
        feats = [x.split(' "') for x in feats]
        # [['gene_id', 'ENSG00000223972.5"'], [' gene_type', 'transcribed_unprocessed_pseudogene"'], [' gene_name', 'DDX11L1"'], [' level 2'], [' hgnc_id', 'HGNC:37102"'], [' havana_gene', 'OTTHUMG00000000961.2"']]
        #feats = {k: v for k, v in feats}
        feats = {pair[0].strip(): pair[1].rstrip('"') for pair in feats if len(pair) == 2}
        an_dict['Accession'].append(feats['gene_id'].strip('"'))
        # 判断gene_name是否为None
        gene_name = feats.get('gene_name')
        if gene_name is not None:
            an_dict['Gene'].append(gene_name.strip('"'))
        else:
            an_dict['Gene'].append('')
        an_dict['Distance'].append(int(x[-1]))
        an_dict['Loc'].append(int(x[3]))
    # 对注释信息进行排序并转换为 NumPy 数组
    an_dict = {x: np.array(an_dict[x]) for x in an_dict}
    an_dict = {x: an_dict[x][an_dict['Loc'].argsort()] for x in an_dict}
    # 将注释信息添加到 loom 文件的 row attributes 中
    for x in ['Accession', 'Gene', 'Distance']:
        adata.var[x] = an_dict[x]   
    print("...Overlapping finish")
    return None

def annotation_df(df,ref):
    """Annotate bins with gene information. Now not work on windows system because of the pybedtools package.
    Args:
        df : dmr datafram
        ref : gtf file in BedTool object
    """
    coding = ref
    print("... Loading regions info")
    # 拆分 names 列为 chromosome, start 和 end
    df[['chromosome', 'start', 'end']] = df['names'].str.split(':|-', expand=True)
    # 转换 start 和 end 列为整数类型
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    bins = np.stack([df['chromosome'], df['start'], df['end'], range(len(df))]).T.tolist()
    bins = BedTool(bins)
    # 对于多个基因位于相同距离的情况，仅为距离最近的基因进行标注
    print("... Overlapping genes with regions")
    annot = bins.sort().closest(coding.sort(), d=True, t='first')
    # 创建一个字典用于存储注释信息
    an_dict = {'Accession': [], 'Gene': [], 'Distance': [], 'Loc':[]}
    for x in annot:
        # 将注释信息拆分为键值对并存储到字典中
        feats = x[12].split(';')[:-1]
        feats = [x.split(' ') for x in feats]
        feats = {k: v for k, v in feats}
        an_dict['Accession'].append(feats['gene_id'].strip('"'))
        # 判断gene_name是否为None
        gene_name = feats.get('gene_name')
        if gene_name is not None:
            an_dict['Gene'].append(gene_name.strip('"'))
        else:
            an_dict['Gene'].append('')
        an_dict['Distance'].append(int(x[-1]))
        an_dict['Loc'].append(int(x[3]))
    # 对注释信息进行排序并转换为 NumPy 数组
    an_dict = {x: np.array(an_dict[x]) for x in an_dict}
    an_dict = {x: an_dict[x][an_dict['Loc'].argsort()] for x in an_dict}
    # 将注释信息添加到 loom 文件的 row attributes 中
    for x in ['Accession', 'Gene', 'Distance']:
        df[x] = an_dict[x]   
    print("...Overlapping finish")
    return df
        
        
def read_annotation_bed(annotation_file,keep_other_columns=True):
    annotation_dict = {}
    with open(annotation_file) as features:
        for line in features:
            ar = line.strip().split()
            chromosome, start, end = ar[0], int(ar[1]), int(ar[2])
            if chromosome not in annotation_dict:
                annotation_dict[chromosome] = []
            annotation_dict[chromosome].append((start, end))
    return annotation_dict

def _load_chrom_size_file(file_path):
    chrom_size_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            chrom, chrom_length = line.strip().split()
            chrom_size_dict[chrom] = int(chrom_length)
    return chrom_size_dict
                     
def sliding_windows_with_step_size(window_size, step_size, ref, chrom_size=None, chrom_file=None):
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
    annotation_dict = {}
    for chrom, chrom_length in chrom_size_dict.items():
        bin_start = np.array(list(range(0, chrom_length, step_size)))
        bin_end = bin_start + window_size
        bin_end[np.where(bin_end > chrom_length)] = chrom_length
        for start, end in zip(bin_start, bin_end):
            annotation_dict.setdefault(chrom, []).append((start, end))
    return annotation_dict

def read_gtf(file_path,feature_type="gene", tss_up=0, tss_down=0):
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
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  
            fields = line.strip().split('\t')
            feature_type = fields[2]
            
            if feature_type == 'gene':
                attributes = dict(item.strip().split(' ') for item in fields[8].split(';') if item.strip())
                gene_type = attributes.get('gene_biotype', '')
                gene_name = attributes.get('gene_name', '')
                chrom = fields[0]
                start = int(fields[3])
                end = int(fields[4])
                strand = fields[6]

                if gene_type == 'protein_coding':
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

                        annotation_dict.setdefault(chrom, []).append((promoter_start, promoter_end, gene_name, strand))
                    else:
                        annotation_dict.setdefault(chrom, []).append((start, end, gene_name))
    return annotation_dict


def read_gff(file_path,feature_type="gene", tss_up=0, tss_down=0):
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
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  
            fields = line.strip().split('\t')
            feature_type = fields[2]
            
            if feature_type == 'gene':
                attributes = dict(item.strip().split(' ') for item in fields[8].split(';') if item.strip())
                gene_type = attributes.get('gene_biotype', '')
                gene_name = attributes.get('gene_name', '')
                chrom = fields[0]
                start = int(fields[3])
                end = int(fields[4])
                strand = fields[6]

                if gene_type == 'protein_coding':
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

                        annotation_dict.setdefault(chrom, []).append((promoter_start, promoter_end, gene_name, strand))
                    else:
                        annotation_dict.setdefault(chrom, []).append((start, end, gene_name))
    return annotation_dict

def map_ensembl_to_gene_name(de_results, gene_names_file):
    """
    Map Ensembl IDs to gene names and remove genes with no mapping.

    Parameters
    ----------
    de_results : pd.DataFrame
        DataFrame containing the differential expression analysis results.
    gene_names_file : str
        Path to the file containing the mapping between Ensembl IDs and gene names.

    Returns
    -------
    pd.DataFrame
        DataFrame with Ensembl IDs mapped to gene names and genes with no mapping removed.
    """
    # Read the gene names file
    gene_names = pd.read_csv(gene_names_file, sep='\t', header=None, names=['ensembl_id', 'gene_name'])

    # Create a dictionary for mapping Ensembl IDs to gene names
    id_to_name = dict(zip(gene_names['ensembl_id'], gene_names['gene_name']))

    # Map Ensembl IDs to gene names in the results DataFrame
    de_results['gene_name'] = de_results['gene'].map(id_to_name)

    # Remove genes with no mapping
    de_results = de_results.dropna(subset=['gene_name'])

    return de_results