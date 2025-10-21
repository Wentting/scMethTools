import os
import gzip
import glob
import logging
import numpy as np
from scipy import sparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import defaultdict, namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import threading
import fcntl
import time
import scMethtools.logging as logg
from scMethtools.preprocessing.adaptive_smooth import _smoothing_chrom_fast
from scMethtools.preprocessing._methylation_level import _feature_methylation_chrom
from scMethtools.preprocessing._format import reorder_columns_by_index, CoverageFormat


class SingleCellMethylationReader:
    """single cell methylation reader and processor
    """
    
    def __init__(self, input_dir, output_dir, pipeline='bismark', 
                 cpu=None, context="CG", file_pattern="*.bed", 
                 cov_threshold=1, high_threshold=0.9, low_threshold=0.1,
                 chunk_size=10_000_000, buffer_size=5000, adaptive=False,
                 keep_temp=False, chrom_format='keep'):  
        """
        Initialize the single cell methylation processor
        
        input_dir: the directory of single cell methylation data
        output_dir: the directory of output sparse matrix
        pipeline: the format of single cell methylation data,default is 'bismark'.
        cpu: the number of CPUs
        context: the context of methylation data,default is 'CG'.
        file_pattern: the pattern of single cell methylation data,default is '*.bed'.
        cov_threshold: the threshold of coverage,default is 1.
        high_threshold: the threshold of high methylation,default is 0.9.
        low_threshold: the threshold of low methylation,default is 0.1.
        chunk_size: the size of chunk,default is 10_000_000.
        buffer_size: the size of buffer,default is 5000.
        adaptive: whether to use adaptive smoothing,default is False.
        keep_temp: whether to keep the temporary files,default is False.
        chrom_format: the format of chromosome name,default is 'keep'. remove 'chr' prefix or add 'chr' prefix
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = self.output_dir / "tmp"
        self.data_dir = self.output_dir / "data"
        self.file_format = pipeline
        self.context = context
        self.file_pattern = file_pattern
        self.keep_temp = keep_temp
        self.cov_threshold = cov_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.chrom_format = chrom_format  # new parameter: uniform chromosome naming format
        self.adaptive = adaptive  # new parameter: adaptive smoothing
        self.format_config = self._parse_format(pipeline)
        self.cpu = min(cpu or cpu_count() // 2, cpu_count() // 2) 

        self.status_file = self.output_dir / "processing_status.json"
        self.status = self._load_status()
        
        # generate directories
        self._create_directories()
        
        # record the processed files and cells
        self.cell_files = []
        self.cell_names = []
        self.chrom_sizes = {}  # 记录每个染色体的最大位置      
        # chromosome chunk mapping
        self.chunk_map = {}  # {chrom: {chunk_id: [临时文件]}}
        
    def _create_directories(self):
        """create necessary directories"""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def _parse_format(self, pipeline):
        """parse file format"""
        if isinstance(pipeline, str):
            return reorder_columns_by_index(pipeline)
        elif isinstance(pipeline, CoverageFormat):
            return pipeline
        else:
            raise ValueError(f"Invalid pipeline format: {pipeline}")
        
    def _load_status(self):
        """load processing status"""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "find_files": 0,
                "read_cells": 0, 
                "save_cells": 0,
                "smooth_cells": 0,
                "failed_chroms": {
                    "save_cells": [],
                    "smooth_cells": []
                },
                "completed_chroms": {
                    "save_cells": [],
                    "smooth_cells": []
                },
                "failed_cells": [],  # new parameter: record failed cells
                "processed_cells": 0  # 新增：已处理的细胞数
            }
    
    def _update_step_status(self, step_name, status_code):
        """update step status"""
        self.status[step_name] = status_code
        self._save_status()
        logg.info(f"Step '{step_name}' status updated to: {status_code}")
    
    def _save_status(self):
        """save processing status"""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def verify_chromosomes(self, step='save'):
        """verify chromosome data completeness - new function"""
        if step == 'save':
            missing_chroms = []
            for chrom in self.chrom_sizes.keys():
                npz_file = self.data_dir / f"{chrom}.npz"
                if not npz_file.exists():
                    missing_chroms.append(chrom)
            if missing_chroms:
                logg.warning(f"Missing NPZ files for chromosomes: {missing_chroms}")
            return missing_chroms
        return []
            
    def find_cell_files(self):
        """find cell files in input directory"""
        pattern = str(self.input_dir / self.file_pattern)
        self.cell_files = sorted(glob.glob(pattern))
        self.cell_names = [os.path.splitext(os.path.basename(f))[0] for f in self.cell_files]
        
        if not self.cell_files:
            raise ValueError(f"No files found in {self.input_dir} matching {self.file_pattern}")
        
        logg.info(f"Found {len(self.cell_files)} cell files")
        return self.cell_files
    
    def read_cells(self):
        """read cells"""
        chrom_col, pos_col, meth_col, umeth_col, context_col, cov, sep, header = self.format_config
        all_stats, global_chrom_sizes, failed_cells = parallel_process_cells(
            self.cell_files, self.temp_dir, chrom_col, pos_col, meth_col, umeth_col, context_col,
            cov, sep, header, self.cov_threshold, self.high_threshold, self.low_threshold,
            self.chunk_size, self.buffer_size, self.cpu, self.keep_temp, self.chrom_format
        )
        self.chrom_sizes = global_chrom_sizes
        self.all_stats = all_stats
        try:
            import pandas as pd
            csv_path = self.output_dir / "all_stats.csv"

            if isinstance(all_stats, pd.DataFrame):
                all_stats.to_csv(csv_path, index=False)
            elif isinstance(all_stats, list):
                # 如果是 list of dict
                if all(isinstance(x, dict) for x in all_stats):
                    df = pd.DataFrame(all_stats)
                    df.to_csv(csv_path, index=False)
                else:
                    df = pd.DataFrame({"stats": all_stats})
                    df.to_csv(csv_path, index=False)
            elif isinstance(all_stats, dict):
                df = pd.DataFrame([all_stats])
                df.to_csv(csv_path, index=False)
            else:
                logg.warning(f"Unsupported all_stats type ({type(all_stats)}), cannot save as CSV.")
            logg.info(f"cell stats saved as CSV at: {csv_path}")
        except Exception as e:
            logg.warning(f"Failed to save all_stats as CSV: {str(e)}")
        self.status['failed_cells'] = failed_cells  # record failed cells
        self.status['processed_cells'] = len(all_stats)
        self._save_status()
        
        if failed_cells:
            logg.warning(f"Failed to process {len(failed_cells)} cells: {failed_cells}")
        
        return all_stats, global_chrom_sizes
            
    def save_cells(self):
        """save cells"""
        n_jobs = min(self.cpu, len(self.chrom_sizes))  # modify: adjust process number according to chromosome number
        missing_chroms = all_chroms_coo_to_csr_parall(
            self.temp_dir, self.data_dir, self.chrom_sizes, len(self.all_stats), n_jobs
        )
        
        # verify result completeness
        verify_missing = self.verify_chromosomes('save')
        if verify_missing:
            logg.error(f"Verification failed: missing chromosomes {verify_missing}")
            self.status['failed_chroms']['save_cells'].extend(verify_missing)
        else:
            logg.info("All chromosomes saved successfully")
        
        self._save_status()
    
    def smooth_cells(self):
        """smooth cells"""
        n_jobs = min(self.cpu, len(self.chrom_sizes))  # modify: adjust process number according to chromosome number
        all_chroms_smooth_parall(self.data_dir, self.output_dir, self.chrom_sizes, len(self.all_stats),adaptive=self.adaptive, n_jobs=n_jobs) 


def all_chroms_smooth_parall(data_dir, out_dir, chrom_sizes, n_cells, adaptive=False, n_jobs=1):
    """_summary_

    Args:
        data_dir (_type_): npz file directory
        out_dir (_type_): smoothed file directory
        chrom_sizes (_type_): _description_
        n_jobs (int, optional): _description_. Defaults to 1.
    """
    cpu_count = n_jobs
    logg.info(f"using {cpu_count} processes to process {len(chrom_sizes)} chromosomes")
    
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:           
        futures = {
            executor.submit(_smoothing_chrom_fast, chrom, data_dir, out_dir, adaptive): chrom
            for chrom, chrom_size in chrom_sizes.items()
        }
        for future in as_completed(futures):
            chrom = futures[future]
            try:
                chrom = future.result()
            except Exception as e:
                logg.error(f"Error processing chromosome {chrom}: {e}")
                continue
    return 0


def normalize_chrom(chrom, format_type='keep'):
    """
    standardize chromosome name - move to global function, uniform call
    format_type: 'add_chr' (添加chr前缀) 或 'remove_chr' (移除chr前缀) 或 'keep' (保持原样)
    """
    chrom = str(chrom).strip()
    
    if format_type == 'add_chr':
        if not chrom.startswith('chr'):
            return f"chr{chrom}"
        return chrom
    elif format_type == 'remove_chr':
        if chrom.startswith('chr'):
            return chrom[3:]
        return chrom
    else:  # format_type == 'keep'
        return chrom


def parallel_process_cells(
    cells, out_dir, chrom_col, pos_col, meth_col, umeth_col, context_col,
    cov, sep, header, cov_threshold=1, high_threshold=0.9, low_threshold=0.1,
    chunk_size=10_000_000, buffer_size=5000, n_jobs=8, clean_existing=True, chrom_format='keep'
):
    
    os.makedirs(out_dir, exist_ok=True)
    
    if clean_existing:
        print("Cleaning existing temporary and COO files...")
        # 改进：使用更精确的文件清理逻辑
        for file_path in Path(out_dir).rglob('*.tmp'):
            file_path.unlink()
        for file_path in Path(out_dir).rglob('*.coo'):
            file_path.unlink()
        
        # 清理空目录
        for root, dirs, files in os.walk(out_dir, topdown=False):
            try:
                if not files and not dirs and root != str(out_dir):
                    os.rmdir(root)
            except OSError:
                pass
    
    failed_cells = []  # 新增：记录失败的细胞
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(
                process_cell, cell_id, bed_file, out_dir, chrom_col, pos_col, meth_col, umeth_col, context_col,
                cov, sep, header, cov_threshold, high_threshold, low_threshold, chunk_size, buffer_size, chrom_format
            ): (cell_id, bed_file)
            for cell_id, bed_file in enumerate(cells)
        }
        
        # 改进：更好的错误处理
        results = []
        for future in as_completed(futures):
            cell_id, bed_file = futures[future]
            try:
                result = future.result()
                results.append(result)
                if (cell_id + 1) % 100 == 0:
                    logg.info(f"Processed {cell_id + 1}/{len(cells)} cells")
            except Exception as e:
                logg.error(f"Failed to process cell {cell_id} ({bed_file}): {e}")
                failed_cells.append((cell_id, bed_file, str(e)))
                continue
    
    # 收集所有细胞的统计信息和染色体大小
    all_stats = []
    global_chrom_sizes = {}
    
    for stats, chrom_max_pos in results:
        all_stats.append(stats)
        # 更新全局染色体大小
        for chrom, max_pos in chrom_max_pos.items():
            if chrom not in global_chrom_sizes:
                global_chrom_sizes[chrom] = max_pos
            else:
                global_chrom_sizes[chrom] = max(global_chrom_sizes[chrom], max_pos)
    
    logg.info(f"Successfully processed {len(results)} cells, {len(failed_cells)} failed")
    logg.info(f"Found {len(global_chrom_sizes)} chromosomes: {list(global_chrom_sizes.keys())}")
    
    merge_tmp_to_coo(out_dir)
    return all_stats, global_chrom_sizes, failed_cells


def process_cell(cell_id, bed_file, out_dir, chrom_col, pos_col, meth_col, umeth_col=None, context_col=None,
                 cov=False, sep='\t', header=False, cov_threshold=1, high_threshold=0.8, low_threshold=0.2,
                 chunk_size=1000000, buffer_size=10000, chrom_format='keep'):
    """
    Process single cell bed file, compatible with full coverage or simplified 3-column (chr/pos/value) format.
    """
    stats = {'cell_id': cell_id, 'cell_name': os.path.basename(bed_file), 
             'sites': 0, 'meth': 0, 'n_total': 0, 'nonCG': 0, 'CG': 0, 'unmeth': 0}
    chrom_max_pos = {}
    open_func = gzip.open if bed_file.endswith('.gz') else open
    mode = 'rt' if bed_file.endswith('.gz') else 'r'
    buffers = dict()
    file_handles = {}

    def get_chunk_path(chrom, chunk_id):
        filename = f"{chrom}_chunk{chunk_id:07d}_cell{cell_id:04d}.tmp"
        return os.path.join(out_dir, filename)

    try:
        with open_func(bed_file, mode) as f:
            if header:
                next(f)
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split(sep)
                # 检查行是否有足够的列
                max_col = max(chrom_col, pos_col, meth_col, umeth_col or 0, context_col or 0)
                if len(parts) <= max_col:
                    continue

                # 标准化染色体名称
                raw_chrom = parts[chrom_col]
                chrom = normalize_chrom(raw_chrom, chrom_format)

                try:
                    pos = int(parts[pos_col])
                    meth_val = float(parts[meth_col])  # 三列格式直接取比例
                    if umeth_col is not None:
                        meth = int(parts[meth_col])
                        umeth = int(parts[umeth_col])
                except (ValueError, IndexError):
                    continue

                # 更新染色体最大位置
                chrom_max_pos[chrom] = max(pos, chrom_max_pos.get(chrom, 0))

                # 对三列格式 context 可能不存在
                context = parts[context_col] if context_col is not None else 'CG'

                stats['n_total'] += 1
                if context_col is not None and not context.upper().startswith('CG'):
                    stats['nonCG'] += 1
                    continue
                stats['CG'] += 1
                stats['sites'] += 1

                # 根据是否为三列格式选择 meth_ratio
                if umeth_col is None:  # 三列格式，meth_val 直接是比例
                    meth_ratio = meth_val
                else:
                    coverage = umeth + (0 if not cov else meth)
                    if coverage < cov_threshold:
                        continue
                    meth_ratio = meth / coverage if coverage > 0 else np.nan

                if meth_ratio >= high_threshold:
                    meth_value = 1
                    stats['meth'] += 1
                elif meth_ratio <= low_threshold:
                    meth_value = -1
                    stats['unmeth'] += 1
                else:
                    continue

                chunk_id = pos // chunk_size
                key = (chrom, chunk_id)
                if key not in buffers:
                    buffers[key] = []
                buffers[key].append(f"{pos},{cell_id},{meth_value}\n")

                if len(buffers[key]) >= buffer_size:
                    tmp_path = get_chunk_path(chrom, chunk_id)
                    if tmp_path not in file_handles:
                        file_handles[tmp_path] = open(tmp_path, "w")
                    file_handles[tmp_path].writelines(buffers[key])
                    file_handles[tmp_path].flush()
                    buffers[key] = []

        # 写入剩余 buffer
        for (chrom, chunk_id), lines in buffers.items():
            if lines:
                tmp_path = get_chunk_path(chrom, chunk_id)
                if tmp_path not in file_handles:
                    file_handles[tmp_path] = open(tmp_path, "w")
                file_handles[tmp_path].writelines(lines)
                file_handles[tmp_path].flush()

    except Exception as e:
        logg.error(f"Error processing cell {cell_id} file {bed_file}: {e}")
        raise
    finally:
        for handle in file_handles.values():
            handle.close()

    return stats, chrom_max_pos


def merge_tmp_to_coo(out_dir):
    """
    合并所有cell的chunk临时文件为最终coo文件
    修改：简化文件查找逻辑，改进进度显示
    """
    # 修改：简化文件查找，直接在out_dir中查找.tmp文件
    tmp_files = list(Path(out_dir).glob("*.tmp"))
    
    if not tmp_files:
        logg.warning("No temporary files found for merging")
        return
    
    logg.info(f"Found {len(tmp_files)} temporary files to merge")
    
    # 按染色体和chunk分组
    chunk_map = defaultdict(list)
    for tmp_file in tmp_files:
        filename = tmp_file.name
        # 文件名格式: {chrom}_chunk{chunk_id:07d}_cell{cell_id:04d}.tmp
        # 提取chrom和chunk_id
        parts = filename.split('_')
        if len(parts) >= 2 and parts[1].startswith('chunk'):
            chrom = parts[0]
            chunk_part = parts[1]  # chunk0000001
            chunk_key = f"{chrom}_{chunk_part}"
            chunk_map[chunk_key].append(str(tmp_file))
    
    logg.info(f"Found {len(chunk_map)} unique chunks to merge")
    
    merged_count = 0
    for chunk_key, files in chunk_map.items():
        coo_path = os.path.join(out_dir, f"{chunk_key}.coo")
        with open(coo_path, "w") as out_f:
            for tmp_f in files:
                try:
                    with open(tmp_f, "r") as in_f:
                        for line in in_f:
                            out_f.write(line)
                    os.remove(tmp_f)
                except Exception as e:
                    logg.error(f"Error processing temp file {tmp_f}: {e}")
                    
        merged_count += 1
        if merged_count % 100 == 0:
            logg.info(f"Merged {merged_count}/{len(chunk_map)} chunks...")
    
    logg.info(f"Merge completed. Created {len(chunk_map)} COO files.")


def all_chroms_coo_to_csr_parall(data_dir, out_dir, chrom_sizes, n_cells, n_jobs=1):
    """
    批量处理所有染色体，返回 {chrom: csr_matrix}
    修改：改进错误处理和日志
    """
    cpu_count = n_jobs
    logg.info(f"using {cpu_count} processes to process {len(chrom_sizes)} chromosomes")
    
    failed_chroms = []
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:           
        futures = {
            executor.submit(coo_chunks_to_csr, data_dir, out_dir, chrom, n_cells, chrom_size): chrom
            for chrom, chrom_size in chrom_sizes.items()
        }
        
        for future in as_completed(futures):
            chrom = futures[future]  # 修改：直接从字典获取chrom名称
            try:
                result_chrom, mat = future.result()
                logg.info(f"{result_chrom} done. Shape: {mat.shape}, nnz: {mat.nnz}")
            except Exception as e:
                logg.error(f"Error processing chromosome {chrom}: {e}")
                failed_chroms.append(chrom)
                continue
    
    if failed_chroms:
        logg.error(f"Failed to process chromosomes: {failed_chroms}")
    else:
        logg.info("All chromosomes processed successfully")
    
    return failed_chroms


def coo_chunks_to_csr(data_dir, out_dir, chrom, n_cells, chrom_size):
    """
    merge all chunk coo files of a chromosome into a CSR matrix
    - data_dir: the directory of chunk coo files
    - out_dir: the directory of output CSR matrix
    - chrom: the chromosome name
    - n_cells: the number of cells
    - chrom_size: the size of the chromosome
    """
    data_chunks = []
    indices_chunks = []
    indptr = np.zeros(chrom_size + 2, dtype=np.int32)
    last_pos = -1
    indptr_counter = 0
    indptr_i = 0

    # 修改：更精确的文件查找，确保染色体名称匹配
    chunk_files = sorted([
        f for f in os.listdir(data_dir)
        if f.startswith(f"{chrom}_chunk") and f.endswith(".coo")
    ])
    
    if not chunk_files:
        logg.warning(f"No COO files found for chromosome {chrom}")
        # 创建空矩阵
        mat = sparse.csr_matrix(([], ([], [])), shape=(chrom_size + 1, n_cells))
        sparse.save_npz(os.path.join(out_dir, f"{chrom}.npz"), mat)
        return chrom, mat
    
    logg.info(f"Processing {len(chunk_files)} chunks for chromosome {chrom}")
    
    for chunk_file in chunk_files:
        chunk_path = os.path.join(data_dir, chunk_file)
        # 逐行读取，避免大文件一次性读入
        chunk_positions = []
        chunk_cells = []
        chunk_values = []
        
        try:
            with open(chunk_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        pos, cell, value = line.split(',')
                        chunk_positions.append(int(pos))
                        chunk_cells.append(int(cell))
                        chunk_values.append(int(value))
                    except ValueError:
                        logg.warning(f"Invalid line in {chunk_path}: {line}")
                        continue
        except Exception as e:
            logg.error(f"Error reading chunk file {chunk_path}: {e}")
            continue
            
        if not chunk_positions:
            continue
            
        arr = np.array(list(zip(chunk_positions, chunk_cells, chunk_values)), dtype=int)
        # reorder by pos and cell
        sorting_idx = np.lexsort((arr[:, 1], arr[:, 0]))
        arr = arr[sorting_idx]
        
        # build indptr
        for pos in arr[:, 0]:
            while last_pos < pos:
                indptr_i += 1
                indptr[indptr_i] = indptr_counter
                last_pos += 1
            indptr_counter += 1
            
        data_chunks.append(arr[:, 2].astype(np.int8))
        indices_chunks.append(arr[:, 1].astype(np.uint16))
    
    indptr[indptr_i+1:] = indptr_counter
    data = np.concatenate(data_chunks) if data_chunks else np.array([], dtype=np.int8)
    indices = np.concatenate(indices_chunks) if indices_chunks else np.array([], dtype=np.uint16)
    mat = sparse.csr_matrix((data, indices, indptr), shape=(chrom_size + 1, n_cells))
    
    sparse.save_npz(os.path.join(out_dir, f"{chrom}.npz"), mat)
    
    return chrom, mat


def read(input_dir, output_dir, pipeline='bismark', 
         cpu=None, context="CG", file_pattern="*.bed", 
         smooth=False, keep_temp=False, chrom_format='keep'):  # 新增chrom_format参数
    
    """
    import single cell methylation data and build sparse matrix
    input_dir: the directory of single cell methylation data
    output_dir: the directory of output sparse matrix
    pipeline: the format of single cell methylation data,default is 'bismark'.
            format: chrom_col:pos_col:meth_col:umeth_col[c/u]:context_col:sep:header
    cpu: the number of CPUs
    context: the context of methylation data,default is 'CG'.
    file_pattern: the pattern of single cell methylation data,default is '*.bed'.
    keep_temp: whether to keep the temporary files,default is False.
    chrom_format: chromosome naming format ('add_chr', 'remove_chr', 'keep'), default is 'keep'.
    
    Example:
        import_cells(input_dir='/xtdisk/methbank_baoym/zongwt/single/data/GSE56789/raw',
                 output_dir='./',pipeline='1:3:7:8c:4:\t:0',cpu=10,context='CG',
                 file_pattern='*.bed',keep_temp=False)
    """
    try:
        # get scm processor
        processor = SingleCellMethylationReader(
            input_dir=input_dir,
            output_dir=output_dir,
            pipeline=pipeline,
            cpu=cpu,
            context=context,
            file_pattern=file_pattern,
            keep_temp=keep_temp,
            chrom_format=chrom_format  # new parameter
        )
        logg.info(f"Initializing SingleCellMethylationReader with pipeline: {pipeline}")
        logg.info(f"Using {processor.cpu} CPUs")
        logg.info(f"Keep temporary files: {processor.keep_temp}")
        logg.info(f"Context: {processor.context}")
        logg.info(f"File pattern: {processor.file_pattern}")
        logg.info(f"Chromosome format: {processor.chrom_format}")  # new log
        logg.info(f"File format: {processor.format_config}")
        logg.info(f"## Starting importing data...")      
        
        # find cell files
        processor.find_cell_files()
        logg.info(f"## Starting reading cells...")
        processor.read_cells()
        logg.info(f"## Reading data done. Processed {processor.status['processed_cells']} cells.")
        
        logg.info(f"## Starting saving data...")
        processor.save_cells()
        logg.info(f"## Saving data done.")
        
        if smooth:
            logg.info(f"## Starting smoothing data...")
            processor.smooth_cells()
            logg.info(f"## Smoothing data done.")

        if processor.status['failed_cells']:
            logg.warning(f"Processing completed with {len(processor.status['failed_cells'])} failed cells")
        else:
            logg.info("Import all single cell methylation data successfully!")
            
        logg.info(f"## Output directory: {output_dir}")
        logg.info(f"## Chromosomes found: {list(processor.chrom_sizes.keys())}")
        
        return "Success"
        
    except Exception as e:
        logg.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {e}"