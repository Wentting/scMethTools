import plotly.express as px
import anndata as ad
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from itertools import groupby
from operator import itemgetter
from scipy.signal import savgol_filter
import scipy.sparse as sparse
import numpy as np
from ..preprocessing.smooth import Smoother

def merge_exons(intervals):

        # Sort intervals based on the start time
    intervals.sort(key=lambda x: x[0])

    # Initialize the list to store merged intervals
    merged = []
    for interval in intervals:
        # Convert interval tuple to list for item assignment
        interval = list(interval)

        # If the merged list is empty or the current interval does not overlap with the previous
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)  # Still append as list
        else:
            # There is overlap, merge the current and previous intervals
            merged[-1][1] = max(merged[-1][1], interval[1])

    # Convert lists back to tuples for final output, if necessary
    return [tuple(interval) for interval in merged]
def queue_reads(dict_per_read):

    sorted_mod = dict(
        sorted(dict_per_read.items(), key=lambda e: e[1][0], reverse=True)
    )
    i = 0
    out_dict = {}

    for k, v in sorted_mod.items():

        if len(out_dict) == 0:

            out_dict[i] = [(k, v)]
            i -= 1
            continue
        for line, reads in out_dict.items():
            t = 0
            for read in reads:
                if overlaps(v[1], read[1][1]) > -5000:
                    t = 1
            if t == 0:
                out_dict[line].append((k, v))
                break

        if t == 1:
            out_dict[i] = [(k, v)]
            i -= 1
    return out_dict
def record_text_plot(record_start, record_end, start, end):
    """
    plot gene models texts (names)
    """
    if (record_start <= start) & (record_end >= end):
        x = [(start + end) / 2]
        textpos = "top center"
    elif (record_start >= start) & (record_end <= end):
        x = [(record_start + record_end) / 2]
        textpos = "top center"
    elif (record_start >= start) & (record_end >= end):
        x = [record_start]
        textpos = "top right"
    elif (record_start <= start) & (record_end <= end):
        x = [record_end]
        textpos = "top left"

    return (x, textpos)
def overlaps(a, b):
    """
    Return the amount of overlap, in bp
    between a and b.
    If >0, the number of bp of overlap
    If 0,  they are book-ended.
    If <0, the distance in bp between them
    """

    return min(a[1], b[1]) - max(a[0], b[0])

def merge_exons(intervals):

        # Sort intervals based on the start time
    intervals.sort(key=lambda x: x[0])

    # Initialize the list to store merged intervals
    merged = []
    for interval in intervals:
        # Convert interval tuple to list for item assignment
        interval = list(interval)

        # If the merged list is empty or the current interval does not overlap with the previous
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)  # Still append as list
        else:
            # There is overlap, merge the current and previous intervals
            merged[-1][1] = max(merged[-1][1], interval[1])

    # Convert lists back to tuples for final output, if necessary
    return [tuple(interval) for interval in merged]
def queue_reads(dict_per_read):

    sorted_mod = dict(
        sorted(dict_per_read.items(), key=lambda e: e[1][0], reverse=True)
    )
    i = 0
    out_dict = {}

    for k, v in sorted_mod.items():

        if len(out_dict) == 0:

            out_dict[i] = [(k, v)]
            i -= 1
            continue
        for line, reads in out_dict.items():
            t = 0
            for read in reads:
                if overlaps(v[1], read[1][1]) > -5000:
                    t = 1
            if t == 0:
                out_dict[line].append((k, v))
                break

        if t == 1:
            out_dict[i] = [(k, v)]
            i -= 1
    return out_dict
def record_text_plot(record_start, record_end, start, end):
    """
    plot gene models texts (names)
    """
    if (record_start <= start) & (record_end >= end):
        x = [(start + end) / 2]
        textpos = "top center"
    elif (record_start >= start) & (record_end <= end):
        x = [(record_start + record_end) / 2]
        textpos = "top center"
    elif (record_start >= start) & (record_end >= end):
        x = [record_start]
        textpos = "top right"
    elif (record_start <= start) & (record_end <= end):
        x = [record_end]
        textpos = "top left"

    return (x, textpos)
def overlaps(a, b):
    """
    Return the amount of overlap, in bp
    between a and b.
    If >0, the number of bp of overlap
    If 0,  they are book-ended.
    If <0, the distance in bp between them
    """

    return min(a[1], b[1]) - max(a[0], b[0])

def SetColor(x):
    if x == 0:
        return "white"
        # return 'white'
    if x == 1:
        return "red"
        # return 'black'
    if x == -1:
        return "blue"
    
def parse_gtf_exons():
    recs = {}  # Dictionary to store the records
    gtf = "D://Test/GSE97693/genome/gencode.v41.basic.annotation.gtf"
    chrom = "chr16"
    start = 1110627
    end = 1112627
    # Open the GTF file
    with open(gtf, 'r') as f:
        # Iterate through each line in the file
        for line in f:
            # Skip header lines
            if line.startswith('#'):
                continue
            
            # Split the line into columns
            columns = line.strip().split('\t')
            
            # Filter by chromosome and feature type
            if columns[0] == chrom and not (int(columns[3]) > end or int(columns[4]) < start):
                # Extract attributes from the last column
                attributes = {attr.strip().split(' ')[0]: attr.strip().split(' ')[1].strip('"') 
                              for attr in columns[8].split(';') if attr}           
                # Construct the key (gene_name and gene_type)
                gene_id = attributes.get('gene_id', 'unknown_gene')
                gene_name = attributes.get('gene_name', 'unknown_name')
                gene_type = attributes.get('gene_type', 'unknown_type')
                gene_key = f"{gene_name} ({gene_type})"
                #gene_key = f"{gene_name}"               
                # Initialize record if not present
                if gene_key not in recs:
                    recs[gene_key] = {'gene': [], 'exons': []}         
                # Add gene and exon information
                if columns[2] == 'gene':
                    #print( columns[2], int(columns[3]), int(columns[4]), columns[6])
                    recs[gene_key]['gene'] = (int(columns[3]), int(columns[4]), columns[6])
                    #print(gene_key,recs[gene_key])
                elif columns[2] == 'exon':         
                    recs[gene_key]['exons'].append((int(columns[3]), int(columns[4])))
                    
    #print(recs)
    out = {}
    for k, v in recs.items():
        # print(k,v)
        coo = (v["gene"][0], v["gene"][1])
        length = v["gene"][1] - v["gene"][0]
        strand = v["gene"][2]
        exons = merge_exons(v["exons"])
        out[k] = [
            length,
            coo,
            record_text_plot(coo[0], coo[1], start, end),
            strand,
            exons,
        ]
    records = queue_reads(out)
    vertical_spacing = 25
    per_line_height = 60
    name_traces = []
    shapes = []
    i = 0
    row = 0
    for row, record_list in records.items():
    
        for record in record_list:
    
            if record[1][3] == "+":
                color = "RoyalBlue"
                fill = "LightSkyBlue"
            elif record[1][3] == "-":
                color = "lightseagreen"
                fill = "mediumaquamarine"
    
            name_traces.append(
                go.Scatter(
                    x=record[1][2][0],
                    y=[(i + 4)],
                    text=[record[0]],
                    mode="text",
                    textposition=record[1][2][1],
                    showlegend=False,
                    textfont=dict(size=14),
                )
            )
    
            shapes.append(
                dict(
                    type="line",
                    x0=record[1][1][0],
                    y0=i,
                    x1=record[1][1][1],
                    y1=i,
                    line=dict(color=color, width=2),
                    fillcolor=fill,
                )
            )
            for exon in record[1][4]:
                shapes.append(
                    dict(
                        type="rect",
                        x0=exon[0],
                        y0=i + 2,
                        x1=exon[1],
                        y1=(i - 2),
                        line=dict(color=color, width=2),
                        fillcolor=fill,
                        opacity=1,
                    )
                )
    
        i -= vertical_spacing
    
        ylim = [i + vertical_spacing / 2, 20]
        height = (abs(row) + 1) * per_line_height

    return ylim, name_traces, shapes, height
    
def parse_npz_methylation(adata,npz,chrom,start,end,clusters,group_by,sample=None,marker_size = 4,single_trace_height=8):
    import plotly.express as px
    start = start - 3000
    end = end + 3000
    cluster_tracks = []
    npz_traces = []
    traces_height = []
    colors = px.colors.qualitative.Dark24

    # window = 50 #step
    # binsize = 100 #bin
    # regions = []
    # het_dict = {"x": [], "y": [], "y_smooth": []}
    # for i in range(start, end, window):
    #     if i + binsize > end:
    #         regions.append((i, end))
    #         break
    #     else:
    #         regions.append((i, i + binsize))
    #         if i + binsize == end:
    #             break
    
    obs = adata.obs
    #clusters = obs[group].unique()
    # 提取指定区域的甲基化数据
    methylation_data = npz[start:end + 1, :]
    #1. better to sample down because single cell experiment will conduct too many traces and it will be slow and  memory consumed
    # obs[obs['sample_region'] == "NC"].sample(n=30, random_state=2)['cell_id']
    #2. clusters is a list

    if len(clusters) < 2:
        clusters.append("Others")
    for i, cluster in enumerate(clusters): #每个cluster一个track
        traces = []
        profile_traces = []
        # freq = plot_frequencies_gl(sample_dict, start, end, color=colors[i])
        # freq_traces.append(freq)
        if sample is not None:                  
            if cluster=="Others":
                print("group:",group_by)
                cell_indices = obs[~(obs[group_by] == clusters[0])].sample(n=sample, random_state=1)['cell_id'] #all others
            else:
                cell_indices = obs[obs[group_by] == cluster].sample(n=sample, random_state=1)['cell_id']
        else:
            if cluster=="Others":
                cell_indices = obs[~(obs[group_by] == clusters[0])]['cell_id'] #all others
            else:
                cell_indices = obs[obs[group_by] == cluster]['cell_id']
        cluster_data = methylation_data[:, cell_indices].T
        print(cluster," shape:",cluster_data.shape[0])
        for cell_id in range(cluster_data.shape[0]): # one trace every cell
            # 获取当前行的所有非零元素
            cols = cluster_data[cell_id,:].nonzero()[1]+ start # 非零元素的列索引
            val_this_row = cluster_data[cell_id,:].data
            # print(col_this_row)
            # 创建每行的trace
            traces.append(go.Scattergl(
                x=cols,
                line=dict(color=colors[i],width=marker_size / 4),
                y= np.full(len(cols), cell_id),  # 让每一行在不同的水平位置
                mode='lines+markers',
                connectgaps=True,
                marker=dict(
                    color=[SetColor(x) for x in val_this_row],
                    size=marker_size,
                    symbol="square",
                ),
                showlegend=False,
            ))
        
        # for region in regions:
        #     hets = []
        #     for cell_id in range(cluster_data.shape[0]): # one trace every cell
        #         # 获取当前行的所有非零元素
        #         signs = []
        #         cols = cluster_data[cell_id,:].nonzero()[1]+ start # 非零元素的列索引
        #         val_this_row = cluster_data[cell_id,:].data
        
        #         for pos, value in zip(cols,val_this_row):
        #             if (pos >= region[0]) & (pos <= region[1]):
        #                 if value == -1:
        #                     signs.append(-1)
        #                 if value == 1:
        #                     signs.append(1)
        #         # print("len(signs)",len(signs),"v:",signs)
        #         if len(signs) <= 1:
        #             continue
        #         else:
        #             zero_crossings = np.where(np.diff(np.sign(signs)))[0]
        #             ranges = []
        #             for k, g in groupby(enumerate(zero_crossings), lambda x: x[0] - x[1]):
        #                 group = list(map(itemgetter(1), g))
        #                 if len(group) > 1:
        #                     ranges.append(range(group[0], group[-1]))
        #                 else:
        #                     ranges.append(group[0])
        
        #             hets.append(len(ranges))
        #     if len(hets) <= 2:
        #         continue
        #     else:
        #         het_dict["x"].append((region[0] + region[1]) / 2)
        #         het_dict["y"].append(np.mean(hets))
        
        # length = len(het_dict["y"])
        # if length <= 5:
        #     smooth_window = 1
        #     poly = 0
        # elif (length > 5) & (length <= 20):
        #     smooth_window = 5
        #     poly = 3
        # elif (length > 20) & (length <= 50):
        #     smooth_window = 21
        #     poly = 3
        # else:
        #     smooth_window = 51
        #     poly = 3
        
        #het_dict["y_smooth"] = savgol_filter(het_dict["y"], smooth_window, poly)

        het_dict = {"x": [], "y": [], "y_smooth": []}
        smooth_csr = methylation_data[:, cell_indices]
        # print(cell_indices)
        # print(smooth_csr)
        sm = Smoother(smooth_csr,bandwidth=300, weigh=True)
        smoothed_chrom = sm.smooth_whole_chrom()
        for pos, smooth_val in smoothed_chrom.items():
                
                het_dict["x"].append(pos+start)
                het_dict["y"].append(smooth_val)
                het_dict["y_smooth"].append(smooth_val)
        
        profile_traces.append(
            go.Scatter(
                x=het_dict["x"],
                y=het_dict["y"],
                mode="markers",
                marker=dict(
                    size=3,
                    color=colors[i],
                ),
                name="",
                showlegend=False,
            )
        )
        
        profile_traces.append(
            go.Scatter(
                x=het_dict["x"],
                y=het_dict["y_smooth"],
                mode="lines",
                marker=dict(
                    size=3,
                    color=colors[i],
                ),
                name="",
                showlegend=False,
            )
        )

        height = cluster_data.shape[0] #how many cells/traces
        npz_traces.append([traces, cluster ,height * single_trace_height]) #profile, cluster_name, trace_number
        cluster_tracks.append([profile_traces,cluster, 150])
    return npz_traces,cluster_tracks

def get_heights(tracks):
    heights = []
    track_names = []
    for track_type, sub_tracks in tracks.items():
        for track in sub_tracks:
            if track_type == "gtf":
                track_names.append("Gene Annotation")
                heights.append(track[-1])
            elif track_type == "npz":
                track_names.append(track[1])
                heights.append(track[-1])
            elif (track_type == "cluster") :
                track_names.append("Methylation Profile")
                heights.append(track[-1])
                break
    plot_height = sum(heights)
    row_heights = [x / plot_height for x in heights]   
    return plot_height, row_heights, track_names

def profile(adata,
            npz_file,
            region,
            annotation_file,
            clusters,
            group_by,
            sample=None,single_trace_height= 12):
    """
    plot mehtylation profiles for genomic regions at single-base resolution

    Args:
        adata (_type_): scm object must have obs and cluster information
        npz_file (_type_): npz file for chrom region
        region (_str_): str of genomic region, e.g. chr1:1000-2000. chr prefix need to be consistent with gtf file
        gtf (_path_): gtf file path
        sample (int, optional): Number of random samples taken from the clusters. Defaults to 30.

    Returns:
        fig
        ----------------
        profile(adata,
            npz_file="./chr1.npz",
            region="1:152100001-152200000",
            annotation_file=gtf_file,
            clusters=["ESC"],group_by="Cell_type",
            sample=None)
            
    """
    chrom = region.strip().split(":")[0]
    start = int(region.strip().split(":")[1].split("-")[0])
    end = int(region.strip().split(":")[1].split("-")[1])

    tracks = {}
    num_tracks = 0
    #maker gene structure annotation
    genes = parse_gtf_exons(gtf=annotation_file,chrom=chrom,start=start,end=end)
    tracks["gtf"] = []
    tracks["gtf"].append(genes)
    #num_tracks += 1
    # methylation profile data
    npz_data = sparse.load_npz(npz_file)
    npz_traces,cluster_tracks = parse_npz_methylation(adata=adata,npz=npz_data,chrom=chrom,start=start,end=end,clusters=clusters,group_by=group_by,sample=sample)
    # npz_traces,cluster_tracks = parse_npz_methylation(adata=adata,npz=npz_data,chrom=chrom,start=start,end=end,clusters=["NC"],group_by="sample_region",sample=sample)
    tracks["cluster"] = []
    tracks["cluster"] = cluster_tracks 
    num_tracks += len(npz_traces)
    tracks["npz"] = []
    tracks["npz"] = npz_traces
    num_tracks += len(npz_traces)
    plot_height, row_heights,track_names = get_heights(tracks)
    print(track_names) 
    fig = make_subplots(
            rows=num_tracks,
            subplot_titles=track_names,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=50 / plot_height,
            row_heights=row_heights, #relative heights percentage
        )
    
    i = 1
    for genes in tracks["gtf"]:
        for name_trace in genes[1]:
            fig.append_trace(name_trace, row=i, col=1)
        for shape in genes[2]:
            fig.add_shape(shape, row=i, col=1)
        fig.update_xaxes(visible=False, row=i, col=1)
        fig.update_yaxes(range=genes[0], visible=False, row=i, col=1)
        i += 1
    for cluster_id in range(len(tracks["cluster"])):
        for trace in tracks["cluster"][cluster_id][0]:
            fig.add_trace(trace, row=i, col=1) #cell trace
        fig.update_xaxes(visible=False, row=i, col=1)
        fig.update_yaxes(visible=False, row=i, col=1)
        fig.update_yaxes(
                            visible=False,
                            row=i,
                            col=1,
                        )
    i += 1
    for cluster_id in range(len(tracks["npz"])):
        for trace in tracks["npz"][cluster_id][0]:
            fig.add_trace(trace, row=i, col=1) #cell trace
        fig.update_xaxes(visible=False, row=i, col=1)
        fig.update_yaxes(visible=False, row=i, col=1)
        fig.update_yaxes(
                            visible=False,
                            row=i,
                            col=1,
                        )
        i += 1
    start_us = start - 3000
    end_ds = end + 3000
    fig.update_xaxes(range=[start_us,end_ds] , visible=True, row=i - 1, col=1)
    fig.update_layout(
                height=plot_height,
                # autosize=False,
                margin=dict(l=10, r=10, t=30, b=60),
            )
    fig.update_layout(plot_bgcolor='#fff')
    fig.show()

