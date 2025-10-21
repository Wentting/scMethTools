import os
import gzip
import glob
import shutil
import logging
import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import gc
from collections import defaultdict, namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import json
import mmap
import tempfile
import psutil
from typing import Dict, List, Tuple, Optional, Union

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logg = logging.getLogger(__name__)

class FileFormat(
    namedtuple('CoverageFormat', ['chrom', 'pos', 'meth', 'umeth', 'context', 'coverage', 'sep', 'header'])):
    """Describes the columns in the coverage file."""
    
    def remove_chr_prefix(self):
        """Remove "chr" or "CHR" etc. from chrom."""
        return self._replace(chrom=self.chrom.lower().lstrip("chr"))

def _custom_format(pipeline: str) -> Tuple:
    """Custom format for single cell methylation data."""
    try:
        parts = pipeline.lower().split(":")
        if len(parts) != 7:
            raise Exception("Invalid number of ':'-separated values in custom input format")
     
        indices = [int(p) - 1 for p in parts[:3]]
        chrom, pos, meth = indices
        
        import re
        match = re.match(r'(\d+)([cuCU])', parts[3])
        if not match:
            raise Exception(
                "The 4th column of a custom input format must contain an integer and "
                "either 'c' for coverage or 'u' for unmethylated counts (e.g. '4c'), "
                f"but you provided '{parts[3]}'."
            )
        umeth, info = match.groups()
        umeth = int(umeth) - 1
        coverage = info.lower() == 'c'
        context = int(parts[4]) - 1
        sep = "\t" if parts[5].lower() in ("\\t", "tab", "t") else parts[5]
        header = bool(int(parts[6]))
        return chrom, pos, meth, umeth, context, coverage, sep, header
    except (ValueError, IndexError) as e:
        raise ValueError(f"Format parsing error: {str(e)}")

def reorder_columns_by_index(pipeline: str) -> FileFormat:
    """Parse pipeline format and return FileFormat object."""
    pipeline = pipeline.lower()
    
    format_orders = {
        'bismark': FileFormat(0, 1, 3, 4, 5, False, "\t", False),
        'bsseeker2': FileFormat(0, 2, 6, 7, 3, True, "\t", False),
        'bsseeker': FileFormat(0, 2, 6, 7, 3, True, "\t", False),
        'methylpy': FileFormat(0, 2, 6, 7, 3, True, "\t", False),
        'allc': FileFormat(0, 1, 4, 5, 3, True, "\t", True)
    }
    
    if pipeline == 'allc':
        pipeline = "1:2:5:6c:4:\t:1"
    
    if pipeline in format_orders:
        logg.info(f"## BED column format: {pipeline}")
        return format_orders[pipeline]
    elif ":" in pipeline:
        logg.info("## BED column format: Custom")
        return FileFormat(*_custom_format(pipeline))
    else:
        raise ValueError(f"Invalid format type or custom order {pipeline}")

class OptimizedMethylationProcessor:
    """优化后的单细胞甲基化数据处理器"""
    
    def __init__(self, input_dir: str, output_dir: str, pipeline: str = 'bismark', 
                 cpu: Optional[int] = None, context: str = "CG", 
                 file_pattern: str = "*.bed", cov_threshold: int = 1, 
                 high_threshold: float = 0.9, low_threshold: float = 0.1,
                 chunk_size: int = 10_000_000, buffer_size: int = 50000,
                 keep_temp: bool = False, memory_limit_gb: float = 8.0):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = self.output_dir / "tmp"
        self.data_dir = self.output_dir / "CG_data"
        self.smooth_dir = self.output_dir / "smooth_data"
        
        self.file_format = pipeline
        self.context = context
        self.file_pattern = file_pattern
        self.keep_temp = keep_temp
        self.cov_threshold = cov_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.memory_limit_gb = memory_limit_gb
        
        # 动态调整CPU数量
        available_cpu = cpu_count()
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if cpu is None:
            # 基于内存限制动态调整CPU数量
            max_cpu_by_memory = max(1, int(available_memory_gb / 2))  # 每个进程预估2GB
            self.cpu = min(available_cpu // 2, max_cpu_by_memory)
        else:
            self.cpu = min(cpu, available_cpu)
        
        self.format_config = self._parse_format(pipeline)
        
        # 状态管理
        self.status_file = self.output_dir / "processing_status.json"
        self.status = self._load_status()
        
        # 创建目录
        self._create_directories()
        
        # 数据统计
        self.cell_files = []
        self.cell_names = []
        self.chrom_sizes = {}
        self.all_stats = []
        
        logg.info(f"Processor initialized with {self.cpu} CPUs, memory limit: {memory_limit_gb}GB")
    
    def _create_directories(self):
        """创建必要的目录"""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.smooth_dir.mkdir(parents=True, exist_ok=True)
    
    def _parse_format(self, pipeline: Union[str, FileFormat]) -> FileFormat:
        """解析文件格式"""
        if isinstance(pipeline, str):
            return reorder_columns_by_index(pipeline)
        elif isinstance(pipeline, FileFormat):
            return pipeline
        else:
            raise ValueError(f"Invalid pipeline format: {pipeline}")
    
    def _load_status(self) -> Dict:
        """加载处理状态"""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                return json.load(f)
        return {
            "find_files": 0,
            "read_cells": 0,
            "save_cells": 0,
            "smooth_cells": 0,
            "failed_files": [],
            "completed_chroms": []
        }
    
    def _save_status(self):
        """保存处理状态"""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def find_cell_files(self) -> List[str]:
        """查找输入目录中的细胞文件"""
        pattern = str(self.input_dir / self.file_pattern)
        self.cell_files = sorted(glob.glob(pattern))
        self.cell_names = [os.path.splitext(os.path.basename(f))[0] for f in self.cell_files]
        
        if not self.cell_files:
            raise ValueError(f"No files found in {self.input_dir} matching {self.file_pattern}")
        
        logg.info(f"Found {len(self.cell_files)} cell files")
        self.status["find_files"] = len(self.cell_files)
        self._save_status()
        return self.cell_files
    
    def process_cells_optimized(self):
        """优化的细胞处理流程"""
        # 估算内存需求并调整批次大小
        file_sizes = [os.path.getsize(f) for f in self.cell_files]
        avg_file_size = np.mean(file_sizes)
        
        # 根据内存限制调整批次大小
        memory_per_file_mb = avg_file_size / (1024 * 1024) * 3  # 估算处理时内存放大3倍
        max_batch_size = max(1, int(self.memory_limit_gb * 1024 / memory_per_file_mb))
        batch_size = min(max_batch_size, len(self.cell_files))
        
        logg.info(f"Processing {len(self.cell_files)} files in batches of {batch_size}")
        
        all_stats = []
        global_chrom_sizes = {}
        
        # 分批处理文件
        for i in tqdm(range(0, len(self.cell_files), batch_size), desc="Processing batches"):
            batch_files = self.cell_files[i:i+batch_size]
            batch_stats, batch_chrom_sizes = self._process_batch(batch_files, i)
            
            all_stats.extend(batch_stats)
            
            # 更新全局染色体大小
            for chrom, max_pos in batch_chrom_sizes.items():
                global_chrom_sizes[chrom] = max(global_chrom_sizes.get(chrom, 0), max_pos)
            
            # 强制垃圾回收
            gc.collect()
        
        # 合并临时文件
        self._merge_temp_files()
        
        self.all_stats = all_stats
        self.chrom_sizes = global_chrom_sizes
        self.status["read_cells"] = len(all_stats)
        self._save_status()
        
        return all_stats, global_chrom_sizes
    
    def _process_batch(self, batch_files: List[str], batch_start_idx: int) -> Tuple[List[Dict], Dict]:
        """处理单个批次的文件"""
        with ProcessPoolExecutor(max_workers=self.cpu) as executor:
            futures = [
                executor.submit(
                    self._process_single_cell_optimized,
                    batch_start_idx + i, file_path
                )
                for i, file_path in enumerate(batch_files)
            ]
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logg.error(f"Error processing cell: {e}")
                    continue
        
        # 分离统计信息和染色体大小
        batch_stats = [r[0] for r in results]
        batch_chrom_sizes = {}
        
        for _, chrom_sizes in results:
            for chrom, size in chrom_sizes.items():
                batch_chrom_sizes[chrom] = max(batch_chrom_sizes.get(chrom, 0), size)
        
        return batch_stats, batch_chrom_sizes
    
    def _process_single_cell_optimized(self, cell_id: int, bed_file: str) -> Tuple[Dict, Dict]:
        """优化的单细胞处理函数"""
        stats = {
            'cell_id': cell_id,
            'cell_name': os.path.basename(bed_file).split('.')[0],
            'sites': 0, 'meth': 0, 'unmeth': 0, 'n_total': 0, 'nonCG': 0, 'CG': 0
        }
        
        chrom_max_pos = {}
        buffers = defaultdict(list)
        file_handles = {}
        
        # 使用内存映射优化大文件读取
        try:
            open_func = gzip.open if bed_file.endswith('.gz') else open
            mode = 'rt' if bed_file.endswith('.gz') else 'r'
            
            with open_func(bed_file, mode) as f:
                if self.format_config.header:
                    next(f)
                
                for line_num, line in enumerate(f):
                    if not line.strip():
                        continue
                    
                    try:
                        result = self._parse_line_optimized(line, cell_id, stats, chrom_max_pos)
                        if result:
                            chrom, chunk_id, data_line = result
                            key = (chrom, chunk_id)
                            buffers[key].append(data_line)
                            
                            # 定期刷新缓冲区
                            if len(buffers[key]) >= self.buffer_size:
                                self._flush_buffer(key, buffers[key], file_handles, cell_id)
                                buffers[key] = []
                                
                    except Exception as e:
                        logg.warning(f"Error parsing line {line_num} in {bed_file}: {e}")
                        continue
            
            # 刷新剩余缓冲区
            for key, buffer in buffers.items():
                if buffer:
                    self._flush_buffer(key, buffer, file_handles, cell_id)
        
        finally:
            # 关闭文件句柄
            for handle in file_handles.values():
                handle.close()
        
        return stats, chrom_max_pos
    
    def _parse_line_optimized(self, line: str, cell_id: int, stats: Dict, chrom_max_pos: Dict) -> Optional[Tuple]:
        """优化的行解析函数"""
        parts = line.strip().split(self.format_config.sep)
        
        # 检查列数
        max_col = max(
            self.format_config.chrom, self.format_config.pos, 
            self.format_config.meth, self.format_config.umeth, 
            self.format_config.context
        )
        
        if len(parts) <= max_col:
            return None
        
        try:
            # 标准化染色体名称
            chrom = self._normalize_chrom(parts[self.format_config.chrom])
            pos = int(parts[self.format_config.pos])
            meth = int(parts[self.format_config.meth])
            umeth = int(parts[self.format_config.umeth])
            context = parts[self.format_config.context]
            
            # 更新统计
            stats['n_total'] += 1
            chrom_max_pos[chrom] = max(chrom_max_pos.get(chrom, 0), pos)
            
            # 过滤非CG位点
            if not context.upper().startswith('CG'):
                stats['nonCG'] += 1
                return None
            
            stats['CG'] += 1
            stats['sites'] += 1
            
            # 计算覆盖度和甲基化比例
            coverage = umeth + (0 if self.format_config.coverage else meth)
            if coverage < self.cov_threshold:
                return None
            
            meth_ratio = meth / coverage if coverage > 0 else 0
            
            if meth_ratio >= self.high_threshold:
                meth_value = 1
                stats['meth'] += 1
            elif meth_ratio <= self.low_threshold:
                meth_value = -1
                stats['unmeth'] += 1
            else:
                return None
            
            chunk_id = pos // self.chunk_size
            data_line = f"{pos},{cell_id},{meth_value}\n"
            
            return chrom, chunk_id, data_line
            
        except (ValueError, IndexError):
            return None
    
    def _normalize_chrom(self, chrom: str) -> str:
        """标准化染色体名称"""
        chrom = str(chrom).strip()
        if not chrom.startswith('chr'):
            return f"chr{chrom}"
        return chrom
    
    def _flush_buffer(self, key: Tuple, buffer: List[str], file_handles: Dict, cell_id: int):
        """刷新缓冲区到文件"""
        chrom, chunk_id = key
        file_path = self.temp_dir / f"{chrom}_chunk{chunk_id:07d}_cell{cell_id:04d}.tmp"
        
        if file_path not in file_handles:
            file_handles[file_path] = open(file_path, 'a')
        
        file_handles[file_path].writelines(buffer)
    
    def _merge_temp_files(self):
        """合并临时文件"""
        logg.info("Merging temporary files...")
        
        # 收集所有临时文件
        temp_files = list(self.temp_dir.glob("*.tmp"))
        chunk_map = defaultdict(list)
        
        # 按染色体和chunk分组
        for temp_file in temp_files:
            filename = temp_file.stem
            parts = filename.split('_')
            if len(parts) >= 2:
                chrom = parts[0]
                chunk_info = parts[1]
                chunk_key = f"{chrom}_{chunk_info}"
                chunk_map[chunk_key].append(temp_file)
        
        # 合并文件
        for chunk_key, files in tqdm(chunk_map.items(), desc="Merging chunks"):
            output_file = self.temp_dir / f"{chunk_key}.coo"
            
            with open(output_file, 'w') as out_f:
                for temp_file in files:
                    with open(temp_file, 'r') as in_f:
                        out_f.write(in_f.read())
                    
                    # 删除临时文件
                    if not self.keep_temp:
                        temp_file.unlink()
    
    def save_cells_optimized(self):
        """优化的细胞保存流程"""
        logg.info("Converting COO files to CSR matrices...")
        
        # 使用线程池进行I/O密集型操作
        with ThreadPoolExecutor(max_workers=min(self.cpu, 8)) as executor:
            futures = {
                executor.submit(self._convert_chrom_to_csr, chrom, chrom_size): chrom
                for chrom, chrom_size in self.chrom_sizes.items()
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting chromosomes"):
                try:
                    chrom, success = future.result()
                    if success:
                        logg.info(f"Successfully converted {chrom}")
                    else:
                        logg.error(f"Failed to convert {chrom}")
                except Exception as e:
                    chrom = futures[future]
                    logg.error(f"Error converting {chrom}: {e}")
        
        self.status["save_cells"] = 1
        self._save_status()
    
    def _convert_chrom_to_csr(self, chrom: str, chrom_size: int) -> Tuple[str, bool]:
        """将单个染色体的COO文件转换为CSR矩阵"""
        try:
            # 收集该染色体的所有chunk文件
            chunk_files = sorted(self.temp_dir.glob(f"{chrom}_chunk*.coo"))
            
            if not chunk_files:
                logg.warning(f"No COO files found for {chrom}")
                return chrom, False
            
            # 构建CSR矩阵
            all_positions = []
            all_cells = []
            all_values = []
            
            for chunk_file in chunk_files:
                with open(chunk_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            pos, cell, value = map(int, line.strip().split(','))
                            all_positions.append(pos)
                            all_cells.append(cell)
                            all_values.append(value)
            
            if not all_positions:
                logg.warning(f"No data found for {chrom}")
                return chrom, False
            
            # 创建CSR矩阵
            n_cells = len(self.all_stats)
            data = np.array(all_values, dtype=np.int8)
            row_ind = np.array(all_positions, dtype=np.uint32)
            col_ind = np.array(all_cells, dtype=np.uint16)
            
            # 构建CSR矩阵
            coo_matrix = sparse.coo_matrix((data, (row_ind, col_ind)), 
                                         shape=(chrom_size + 1, n_cells))
            csr_matrix = coo_matrix.tocsr()
            
            # 保存矩阵
            output_file = self.data_dir / f"{chrom}.npz"
            sparse.save_npz(output_file, csr_matrix)
            
            # 清理临时文件
            if not self.keep_temp:
                for chunk_file in chunk_files:
                    chunk_file.unlink()
            
            return chrom, True
            
        except Exception as e:
            logg.error(f"Error converting {chrom}: {e}")
            return chrom, False
    
    def run_complete_pipeline(self, smooth: bool = False):
        """运行完整的处理流程"""
        try:
            # 1. 查找文件
            logg.info("Step 1: Finding cell files...")
            self.find_cell_files()
            
            # 2. 处理细胞数据
            logg.info("Step 2: Processing cell data...")
            self.process_cells_optimized()
            
            # 3. 保存数据
            logg.info("Step 3: Saving data...")
            self.save_cells_optimized()
            
            # 4. 可选的平滑处理
            if smooth:
                logg.info("Step 4: Smoothing data...")
                self.smooth_cells()
            
            logg.info("Pipeline completed successfully!")
            return True
            
        except Exception as e:
            logg.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def smooth_cells(self):
        """平滑处理 - 占位符函数"""
        logg.info("Smoothing functionality not implemented in this optimization")
        self.status["smooth_cells"] = 1
        self._save_status()

# 使用示例
def import_cells_optimized(input_dir: str, output_dir: str, 
                         pipeline: str = 'bismark', cpu: Optional[int] = None,
                         context: str = "CG", file_pattern: str = "*.bed", 
                         smooth: bool = False, keep_temp: bool = False,
                         memory_limit_gb: float = 8.0):
    """
    优化版本的单细胞甲基化数据导入函数
    
    Args:
        input_dir: 输入数据目录
        output_dir: 输出数据目录
        pipeline: 数据格式
        cpu: CPU数量
        context: 甲基化上下文
        file_pattern: 文件模式
        smooth: 是否进行平滑处理
        keep_temp: 是否保留临时文件
        memory_limit_gb: 内存限制(GB)
    """
    
    processor = OptimizedMethylationProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        pipeline=pipeline,
        cpu=cpu,
        context=context,
        file_pattern=file_pattern,
        keep_temp=keep_temp,
        memory_limit_gb=memory_limit_gb
    )
    
    success = processor.run_complete_pipeline(smooth=smooth)
    return 0 if success else 1

if __name__ == "__main__":
    # 示例使用
    result = import_cells_optimized(
        input_dir='/path/to/input',
        output_dir='/path/to/output',
        pipeline='1:3:7:8c:4:\t:0',
        cpu=10,
        context='CG',
        file_pattern='*.bed',
        keep_temp=False,
        memory_limit_gb=16.0
    )
