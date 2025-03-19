import requests
import json
from collections.abc import Iterable
import os
import scMethtools

#这个富集有两种情况，一种有网的时候，可以直接使用enrichr的接口这个时候不需要下载gmt文件
#如果不能联网，就需要下载gmt文件，然后指定库文件（似乎必须以gmt为后缀）
#为了方便直接进行富集分析，在软件中配了几个常用的库文件，可以直接使用
#如果需要使用其他库文件，可以下载对应的gmt文件，然后指定库文件

#for online enrichment analysis using Enrichr API
ENRICHR_URL = 'http://maayanlab.cloud/Enrichr/'
POST_ENDPOINT = 'addList'
GET_ENDPOINT = 'enrich?userListId={list_id}&backgroundType={ontology}'
HEADERS = 'rank,term,pvalue,zscore,combined_score,genes,adj_pvalue'.split(',')
LEGACY_ONTOLOGIES = 'WikiPathways_2019_Human,WikiPathways_2019_Mouse,KEGG_2019_Human,KEGG_2019_Mouse,GO_Molecular_Function_2018,GO_Cellular_Component_2018,GO_Biological_Process_2018,BioPlanet_2019'.split(',')

scm_dir = os.path.dirname(scMethtools.__file__)
ENRICH_DIR = os.path.join(scm_dir, 'package_data/motifs/')

def enrich(gene_list, gene_sets = 'WikiPathways_2019_Human',outdir=None, cutoff=0.05):
    """
    Enrichment analysis.
    """
    import gseapy as gp
     # 判断 gene_sets 是否是一个文件路径（即用户提供了自定义数据库）
    if os.path.exists(gene_sets):
        if not gene_sets.endswith('.gmt'):
            raise ValueError("The provided gene set file must have a .gmt extension")
        gene_set_path = gene_sets  # 直接使用提供的路径
    else:
        gene_set_path = os.path.join(ENRICH_DIR, '{}.gmt'.format(gene_sets))
    
    go_results = gp.enrich(gene_list=gene_list, gene_sets=gene_set_path, background=None,
                    outdir=outdir,
                    cutoff=cutoff,
                    no_plot=True,
                    verbose=True)
    return go_results.results



#需要有网调API，就不用下载gesapy了，不然需要自己下载数据库，可以在软件里面带两个库，但是不能太大
def post_genelist(genelist):
    '''
    Post genelist to Enrichr for comparison against pre-compiled ontologies.

    Parameters
    ----------
    genelist : Iterable
        List of genes

    Returns
    -------
    list_id : str
        ID for genelist. Used to retrieve enrichment results.

    '''
    assert(isinstance(genelist, Iterable)), 'Genelist must be an iterable object'

    payload = {
        'list': (None, '\n'.join(genelist)),
    }

    response = requests.post(ENRICHR_URL + POST_ENDPOINT, files=payload)
    if not response.ok:
        raise Exception('Error analyzing gene list')

    list_id = json.loads(response.text)['userListId']
    return list_id


def enrich_online(list_id, ontology = 'WikiPathways_2019_Human'):
    '''
    Fetch enrichment results from an ontology.

    Parameters
    ----------
    list_id : str
        genelist ID returned by `post_genelist`
    onotology : str, default = "WikiPathways_2019_Human"
        Retrieve results for this ontology. For a full list of 
        ontologies, see `Enrichr <https://maayanlab.cloud/Enrichr/#libraries>`_.

    Returns
    -------
    results : dict
        Dictionary with schema:

        .. code-block::

            {
                <ontology> : {
                    [
                        {'rank' : <rank>,
                        'term' : <term>,
                        'pvalue' : <pval>,
                        'zscore': <zscore>,
                        'combined_score': <combined_score>,
                        'genes': [<gene1>, ..., <geneN>],
                        'adj_pvalue': <adj_pval>},
                        ...,
                    ]
                }
            }   

    '''

    try:
        import charset_normalizer
    except ModuleNotFoundError:
        pass

    url = ENRICHR_URL + GET_ENDPOINT.format(
        list_id = str(list_id),
        ontology = str(ontology)
    )

    response = requests.get(url)
    if not response.ok:
        raise Exception('Error fetching enrichment results: \n' + str(response))
    
    data = json.loads(response.text)[ontology]
    
    return {ontology : [dict(zip(HEADERS, x)) for x in data]}


def fetch_ontologies(list_id, ontologies = LEGACY_ONTOLOGIES):
    '''
    Fetch enrichment results from ontologies.

    Parameters
    ----------
    list_id : str
        genelist ID returned by `post_genelist`
    onotologies : Iterable[str], default = mira.tl.LEGACY_ONTOLOGIES
        Retrieve results for these ontologies. For a full list of 
        ontologies, see `Enrichr <https://maayanlab.cloud/Enrichr/#libraries>`_.

    Returns
    -------
    results : dict
        Dictionary with schema:

        .. code-block::
    
            {
                <ontology> : {
                    [
                        {'rank' : <rank>,
                        'term' : <term>,
                        'pvalue' : <pval>,
                        'zscore': <zscore>,
                        'combined_score': <combined_score>,
                        'genes': [<gene1>, ..., <geneN>],
                        'adj_pvalue': <adj_pval>},
                        ...,
                    ]
                }
            }
            
    '''

    results = {}
    assert(isinstance(ontologies, Iterable)), 'Ontologies must be an iterable object'

    for ontology in ontologies:
        results.update(
            enrich_online(list_id, ontology)
        )

    return results
                            
                            
                            
                            