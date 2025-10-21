import os
import glob
import multiprocessing as mp
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats  
from numba import njit


class BiologicallyInformedSegmentation:
    def __init__(self,
                 # 基本统计参数
                 alpha=0.01, # 统计显著性水平

                 # 生物学约束参数
                 min_sites_per_segment=2, # 每个片段最少包含的位点数
                 max_site_distance=5000, # 同一片段内相邻位点间的最大距离 (bp)
                 max_segment_length=10_000_000, # 片段的最大长度 (bp)
                 min_segment_length=100, # 片段的最小长度 (bp)
                 # 新增：合并相邻片段时允许的最大间隙，用于更灵活的合并
                 max_merge_gap=500, # 合并相邻片段时允许的最大基因组间隙 (bp)

                 # 单细胞甲基化特异性参数
                 coverage_threshold=3, # 过滤低覆盖度位点的阈值
                 dropout_tolerance=0.3, # 片段内允许的最大数据缺失率
                 sparse_region_threshold=0.1, # 稀疏区域的位点密度阈值 (sites/kb)

                 # 甲基化水平阈值
                 fully_methylated_threshold=0.8, # 完全甲基化区域的平均甲基化水平阈值
                 fully_unmethylated_threshold=0.2, # 完全未甲基化区域的平均甲基化水平阈值
                 variance_threshold=0.05, # 低变异区域的甲基化水平方差阈值

                 # 预分割参数
                 gap_threshold=10000, # 用于预分割的大间隙阈值 (bp)

                 # 质量控制参数
                 min_data_points=10, # 加载数据时所需的最小数据点数
                 outlier_percentile=95): # 异常值检测百分位数，目前未严格使用

        self.alpha = alpha
        self.min_sites_per_segment = min_sites_per_segment
        self.max_site_distance = max_site_distance
        self.max_segment_length = max_segment_length
        self.min_segment_length = min_segment_length
        self.max_merge_gap = max_merge_gap # 新增参数
        self.coverage_threshold = coverage_threshold
        self.dropout_tolerance = dropout_tolerance
        self.sparse_region_threshold = sparse_region_threshold
        self.fully_methylated_threshold = fully_methylated_threshold
        self.fully_unmethylated_threshold = fully_unmethylated_threshold
        self.variance_threshold = variance_threshold
        self.gap_threshold = gap_threshold
        self.min_data_points = min_data_points
        self.outlier_percentile = outlier_percentile

        self.segmentation_stats = {}
        # 模拟CpG岛区域，实际应从文件加载
        self.preloaded_cpg_islands = []
        # 用于存储原始DataFrame的引用，以便在合并时重新计算统计量
        self._original_df = None 

    def load_data(self, filename: str) -> pd.DataFrame:
        """加载并预处理数据"""
        try:
            df = pd.read_csv(filename, header=None, names=['position', 'smooth_value'])
        except FileNotFoundError:
            print(f"Error: File not found at {filename}. Generating dummy data for demonstration.")
            np.random.seed(42)
            positions = np.sort(np.random.randint(1, 2_000_000, 2000)) # 更多数据点
            smooth_values = np.random.rand(2000) * 0.6 + 0.2
            # 模拟一些变化
            smooth_values[100:200] = np.random.rand(100) * 0.1 + 0.9 # 高甲基化区域
            smooth_values[500:600] = np.random.rand(100) * 0.1 # 低甲基化区域
            smooth_values[1500:1600] = np.random.rand(100) * 0.1 + 0.9 # 另一个高甲基化区域
            # 模拟一些大间隙
            positions = np.concatenate([
                positions[:500],
                positions[500:1000] + 50000, # 制造一个大间隙
                positions[1000:] + 100000 # 制造另一个大间隙
            ])
            smooth_values = np.concatenate([
                smooth_values[:500],
                smooth_values[500:1000],
                smooth_values[1000:]
            ])
            df = pd.DataFrame({'position': positions, 'smooth_value': smooth_values})

        df = df.sort_values('position').reset_index(drop=True)
        if 'coverage' not in df.columns:
            df['coverage'] = np.random.randint(5, 20, len(df))
        df = df[df['coverage'] >= self.coverage_threshold].reset_index(drop=True)
        
        if len(df) < self.min_data_points:
            raise ValueError(f"数据点太少: {len(df)} < {self.min_data_points}")
        
        return df

    def calculate_site_density(self, positions: np.ndarray) -> float:
        """计算位点密度 (sites/kb)"""
        if len(positions) < 2:
            return 0.0
        span = positions[-1] - positions[0]
        return len(positions) / (span / 1000) if span > 0 else float('inf')

    def cbs_core_segmentation(self, smooth_values: np.ndarray,
                              positions: np.ndarray) -> List[int]:
        """
        简化的CBS核心算法。通过递归地寻找甲基化水平的显著变化点来进行分割。
        核心思想：尝试在所有可能的位置 k 进行分割，计算左右两部分的均值，
        并使用Welch's t-test判断这两部分均值是否存在显著差异。
        选择最显著的分割点，然后对子区域递归进行此过程。
        """
        if len(smooth_values) < 2 * self.min_sites_per_segment:
            return []

        changepoints = []
        n = len(smooth_values)

        max_stat = 0 # 记录最大的t统计量绝对值
        best_changepoint = -1 # 记录最佳分割点索引

        for k in range(self.min_sites_per_segment, n - self.min_sites_per_segment + 1):
            left_valid = smooth_values[:k][~np.isnan(smooth_values[:k])]
            right_valid = smooth_values[k:][~np.isnan(smooth_values[k:])]

            # 增强鲁棒性：检查子段是否有足够的有效数据点且方差不为零
            if len(left_valid) > 1 and len(right_valid) > 1 and np.var(left_valid) > 0 and np.var(right_valid) > 0:
                try:
                    stat, p_value = stats.ttest_ind(left_valid, right_valid, equal_var=False)
                    if abs(stat) > max_stat and p_value < self.alpha:
                        max_stat = abs(stat)
                        best_changepoint = k
                except ValueError: # ttest_ind可能会因为数据同质性等问题抛出ValueError
                    continue
            # 如果方差为零，则无法进行t检验，跳过此分割点
            elif (len(left_valid) > 1 and np.var(left_valid) == 0) or \
                 (len(right_valid) > 1 and np.var(right_valid) == 0):
                continue

        if best_changepoint != -1:
            changepoints.append(best_changepoint) # 找到一个显著变化点

            # 递归处理左右子区域
            left_cp = self.cbs_core_segmentation(
                smooth_values[:best_changepoint],
                positions[:best_changepoint]
            )
            right_cp = self.cbs_core_segmentation(
                smooth_values[best_changepoint:],
                positions[best_changepoint:]
            )

            # 调整右侧变化点索引至原始区域的相对位置
            right_cp = [cp + best_changepoint for cp in right_cp]

            changepoints.extend(left_cp)
            changepoints.extend(right_cp)

        return sorted(list(set(changepoints))) # 去重并排序
        
    def fast_sliding_segmentation(self, smooth_values: np.ndarray, positions: np.ndarray,
                       window_size: int = 500, step_size: int = 100) -> List[int]:
        """修复版本：动态调整窗口大小，更宽松的参数"""
        n = len(smooth_values)
        
        # 动态调整窗口大小 - 更加宽松
        if n < 100:
            window_size = max(self.min_sites_per_segment, n // 20)  # 降低从n//10到n//20
            step_size = max(1, window_size // 3)  # 降低从window_size//5到window_size//3
        elif n < 1000:
            window_size = max(self.min_sites_per_segment * 3, n // 20)  # 降低分母
            step_size = max(1, window_size // 3)
        
        # 确保窗口大小不超过数据长度的1/3（之前是1/4）
        window_size = min(window_size, n // 3)
        
        print(f"Region size: {n}, Using window_size: {window_size}, step_size: {step_size}")
        
        changepoints = []
        
        for i in range(0, n - 2 * window_size, step_size):
            left = smooth_values[i:i+window_size]
            right = smooth_values[i+window_size:i+2*window_size]
            
            # 移除NaN值
            left_valid = left[~np.isnan(left)]
            right_valid = right[~np.isnan(right)]
            
            if len(left_valid) < 2 or len(right_valid) < 2:
                continue
                
            try:
                stat, pval = stats.ttest_ind(left_valid, right_valid, equal_var=False)
                if pval < self.alpha and abs(stat) > 1.0:  # 添加效应量阈值
                    changepoints.append(i + window_size)
            except:
                continue
        
        # 去重和过滤
        changepoints = sorted(list(set(changepoints)))
        
        # 移除过于接近的变化点 - 更宽松的距离要求
        filtered_changepoints = []
        min_distance = max(window_size // 3, self.min_sites_per_segment)  # 降低从window_size//2
        
        for cp in changepoints:
            if not filtered_changepoints or (cp - filtered_changepoints[-1]) >= min_distance:
                filtered_changepoints.append(cp)
        
        print(f"Found {len(filtered_changepoints)} changepoints after filtering")
        return filtered_changepoints

    def _calculate_segment_metrics(self,
                                   positions: np.ndarray,
                                   smooth_values: np.ndarray) -> Dict:
        """
        辅助函数：计算片段的各项统计指标。
        """
        num_sites = len(positions)
        segment_length = positions[-1] - positions[0] if num_sites > 1 else 0

        valid_values = smooth_values[~np.isnan(smooth_values)]
        data_completeness = len(valid_values) / num_sites if num_sites > 0 else 0
        
        mean_smooth = np.mean(valid_values) if len(valid_values) > 0 else np.nan
        std_smooth = np.std(valid_values) if len(valid_values) > 0 else np.nan
        median_smooth = np.median(valid_values) if len(valid_values) > 0 else np.nan
        site_density = self.calculate_site_density(positions)
        max_site_gap = np.max(np.diff(positions)) if num_sites > 1 else 0

        return {
            'length_bp': int(segment_length),
            'num_sites': num_sites,
            'mean_smooth': mean_smooth,
            'std_smooth': std_smooth,
            'median_smooth': median_smooth,
            'site_density': site_density,
            'data_completeness': data_completeness,
            'max_site_gap': max_site_gap
        }

    def validate_and_classify_segment(self,
                                      positions: np.ndarray,
                                      smooth_values: np.ndarray,
                                      segment_global_start_idx: int,
                                      segment_global_end_idx: int) -> Optional[Dict]:
        """
        验证和分类一个潜在的基因组片段，应用所有生物学和单细胞特异性约束。
        此步骤在CBS分割后对每个候选片段进行筛选和特征提取。
        """
        metrics = self._calculate_segment_metrics(positions, smooth_values)

        # 约束验证
        if metrics['num_sites'] < self.min_sites_per_segment:
            return None
        if not (self.min_segment_length <= metrics['length_bp'] <= self.max_segment_length):
            return None
        if metrics['num_sites'] > 1 and metrics['max_site_gap'] > self.max_site_distance:
            return None
        if metrics['data_completeness'] < (1 - self.dropout_tolerance):
            return None

        # Segment 类型分类
        segment_type = 'variable' # 默认类型

        if metrics['site_density'] < self.sparse_region_threshold:
            segment_type = 'sparse'
        elif not np.isnan(metrics['mean_smooth']):
            if metrics['mean_smooth'] >= self.fully_methylated_threshold:
                segment_type = 'fully_methylated'
            elif metrics['mean_smooth'] <= self.fully_unmethylated_threshold:
                segment_type = 'fully_unmethylated'
            elif not np.isnan(metrics['std_smooth']) and metrics['std_smooth'] < self.variance_threshold:
                segment_type = 'low_variance'
        # CpG岛判断 (实际应用中，会加载预先注释的CpG岛区域)
        for cpg_start, cpg_end in self.preloaded_cpg_islands:
            if max(positions[0], cpg_start) < min(positions[-1], cpg_end): # 有重叠
                segment_type = 'cpg_island'
                break

        # 构建最终的segment信息字典
        segment_info = {
            'start_idx': segment_global_start_idx,
            'end_idx': segment_global_end_idx,
            'start_pos': int(positions[0]),
            'end_pos': int(positions[-1]),
            'type': segment_type
        }
        segment_info.update(metrics) # 合并计算出的指标

        return segment_info

    def _merge_adjacent_segments(self, segments: List[Dict]) -> List[Dict]:
        """简化的合并逻辑：减少复杂性，专注于基本合并"""
        if not segments:
            return []
        
        print(f"开始合并 {len(segments)} 个片段")
        
        # 按起始位置排序
        sorted_segments = sorted(segments, key=lambda x: x['start_pos'])
        
        merged_segments = []
        
        for current_seg in sorted_segments:
            if not merged_segments:
                merged_segments.append(current_seg)
                continue
                
            last_seg = merged_segments[-1]
            
            # 检查是否可以合并：相邻且同类型
            gap = current_seg['start_pos'] - last_seg['end_pos']
            can_merge = (gap <= self.max_merge_gap and 
                        current_seg['type'] == last_seg['type'])
            
            if can_merge and self._original_df is not None:
                # 尝试合并
                try:
                    merged_start_idx = last_seg['start_idx']
                    merged_end_idx = current_seg['end_idx']
                    
                    merged_positions = self._original_df['position'].iloc[merged_start_idx:merged_end_idx].values
                    merged_smooth_values = self._original_df['smooth_value'].iloc[merged_start_idx:merged_end_idx].values
                    
                    # 重新计算合并后的指标
                    recalculated_metrics = self._calculate_segment_metrics(
                        merged_positions, merged_smooth_values
                    )
                    
                    # 检查合并后的片段是否仍然满足约束
                    if (self.min_segment_length <= recalculated_metrics['length_bp'] <= self.max_segment_length and
                        recalculated_metrics['num_sites'] >= self.min_sites_per_segment):
                        
                        # 更新最后一个片段为合并后的片段
                        merged_segments[-1] = {
                            'start_idx': merged_start_idx,
                            'end_idx': merged_end_idx,
                            'start_pos': int(merged_positions[0]),
                            'end_pos': int(merged_positions[-1]),
                            'type': last_seg['type']
                        }
                        merged_segments[-1].update(recalculated_metrics)
                    else:
                        # 合并后不满足约束，保持分离
                        merged_segments.append(current_seg)
                except Exception as e:
                    print(f"合并过程中出错: {e}")
                    merged_segments.append(current_seg)
            else:
                # 无法合并，直接添加
                merged_segments.append(current_seg)
        
        print(f"合并后剩余 {len(merged_segments)} 个片段")
        return merged_segments

    def generate_segmentation_stats(self, segments: List[Dict]):
        """生成并存储分割结果的统计报告。"""
        self.segmentation_stats = {
            'total_segments': len(segments),
            'segment_types': {},
            'length_distribution': {},
            'site_count_distribution': {},
            'quality_metrics': {}
        }
        for segment in segments:
            seg_type = segment['type']
            if seg_type not in self.segmentation_stats['segment_types']:
                self.segmentation_stats['segment_types'][seg_type] = 0
            self.segmentation_stats['segment_types'][seg_type] += 1
        if segments:
            lengths = [s['length_bp'] for s in segments]
            self.segmentation_stats['length_distribution'] = {
                'mean': np.mean(lengths), 'median': np.median(lengths), 'std': np.std(lengths),
                'min': np.min(lengths), 'max': np.max(lengths)
            }
            site_counts = [s['num_sites'] for s in segments]
            self.segmentation_stats['site_count_distribution'] = {
                'mean': np.mean(site_counts), 'median': np.median(site_counts), 'std': np.std(site_counts),
                'min': np.min(site_counts), 'max': np.max(site_counts)
            }
            completeness = [s['data_completeness'] for s in segments if not np.isnan(s['data_completeness'])]
            max_site_gaps = [s['max_site_gap'] for s in segments if not np.isnan(s['max_site_gap'])]
            self.segmentation_stats['quality_metrics'] = {
                'mean_completeness': np.mean(completeness) if completeness else np.nan,
                'segments_with_high_quality': sum(c > 0.8 for c in completeness),
                'constraint_violations': sum(1 for gap in max_site_gaps if gap > self.max_site_distance)
            }
        else:
            self.segmentation_stats['length_distribution'] = {k: 0 for k in ['mean', 'median', 'std', 'min', 'max']}
            self.segmentation_stats['site_count_distribution'] = {k: 0 for k in ['mean', 'median', 'std', 'min', 'max']}
            self.segmentation_stats['quality_metrics'] = {'mean_completeness': np.nan, 'segments_with_high_quality': 0, 'constraint_violations': 0}


    def print_segmentation_report(self):
        """打印分割报告"""
        if not self.segmentation_stats:
            print("请先运行分割算法以生成报告。")
            return
        stats = self.segmentation_stats
        print("\n=== 生物学约束分割报告 ===")
        print(f"总segment数: {stats['total_segments']}")
        print(f"平均长度: {stats['length_distribution']['mean']:.0f} bp")
        print(f"平均位点数: {stats['site_count_distribution']['mean']:.1f}")
        print(f"平均数据完整性: {stats['quality_metrics']['mean_completeness']:.2f}")
        print("\nSegment类型分布:")
        if stats['segment_types']:
            for seg_type, count in stats['segment_types'].items():
                print(f"  {seg_type}: {count}")
        else:
            print("  无特定类型segment。")
        print(f"\n约束违反 (最大位点间隙): {stats['quality_metrics']['constraint_violations']} segments")
        print(f"高质量segments (完整度 > 0.8): {stats['quality_metrics']['segments_with_high_quality']}")
        
    def segment_methylation_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 核心方法: 对甲基化数据执行生物学知情分割。
            这是一个多阶段过程，结合了统计学分割和生物学约束过滤。
        """
        self._original_df = df.copy()
        
        positions = df['position'].values
        smooth_values = df['smooth_value'].values
        
        print(f"开始分割 {len(df)} 个位点")
        
        # 步骤1: 大间隙预分割
        gaps = np.diff(positions)
        large_gaps_indices = np.where(gaps > self.gap_threshold)[0]
        
        continuous_regions = []
        current_start_idx = 0
        for gap_idx in large_gaps_indices:
            if (gap_idx + 1) - current_start_idx >= self.min_sites_per_segment:
                continuous_regions.append((current_start_idx, gap_idx + 1))
            current_start_idx = gap_idx + 1
        if len(positions) - current_start_idx >= self.min_sites_per_segment:
            continuous_regions.append((current_start_idx, len(positions)))
        
        print(f"发现 {len(continuous_regions)} 个连续区域进行独立CBS")
        
        all_segments_info = []
        
        # 步骤2 & 3: 对每个连续区域执行CBS并进行生物学约束验证
        for region_start_idx, region_end_idx in continuous_regions:
            region_positions = positions[region_start_idx:region_end_idx]
            region_smooth = smooth_values[region_start_idx:region_end_idx]
        
            if len(region_positions) < 2 * self.min_sites_per_segment:
                segment_info = self.validate_and_classify_segment(
                    region_positions, region_smooth, region_start_idx, region_end_idx
                )
                if segment_info:
                    all_segments_info.append(segment_info)
                continue
        
            changepoints_in_region = self.fast_sliding_segmentation(region_smooth, region_positions, 
                                                        window_size=300, step_size=50)
            
            # 修复：正确创建片段索引
            segment_indices = []
            if not changepoints_in_region:
                # 没有变化点，整个区域作为一个片段
                segment_indices.append((0, len(region_smooth)))
            else:
                # 有变化点，根据变化点分割
                changepoints_sorted = sorted(changepoints_in_region)
                
                # 添加起始和结束点
                all_breakpoints = [0] + changepoints_sorted + [len(region_smooth)]
                
                # 创建连续的片段
                for i in range(len(all_breakpoints) - 1):
                    start_idx = all_breakpoints[i]
                    end_idx = all_breakpoints[i + 1]
                    
                    if end_idx > start_idx:  # 确保片段有效
                        segment_indices.append((start_idx, end_idx))
    
            # 使用修复的非重叠片段创建方法
            region_segments = self._create_non_overlapping_segments(
                segment_indices, region_positions, region_smooth, region_start_idx
            )
            all_segments_info.extend(region_segments)
        
        # 步骤4: 尝试合并相邻的同质segments（使用修复版本）
        final_segments = self._merge_adjacent_segments(all_segments_info)
        
        # 步骤5: 生成统计信息
        self.generate_segmentation_stats(final_segments)
        
        return pd.DataFrame(final_segments)
    
    def _create_non_overlapping_segments(self, segment_indices: List[Tuple[int, int]], 
                                   region_positions: np.ndarray, 
                                   region_smooth: np.ndarray,
                                   region_start_idx: int) -> List[Dict]:
        """修复版本：正确处理片段索引，不应该合并CBS产生的片段"""
        if not segment_indices:
            print("没有片段索引传入")
            return []
        
        # 排序片段索引
        segment_indices = sorted(segment_indices)
        
        # CBS算法产生的片段应该是连续不重叠的，如果有重叠说明算法有问题
        # 这里只做基本的去重和有效性检查
        valid_indices = []
        
        for start, end in segment_indices:
            # 确保索引有效
            start = max(0, start)
            end = min(len(region_positions), end)
            
            if end > start:  # 确保片段有效
                valid_indices.append((start, end))
        
        # 验证每个片段
        valid_segments = []
        for i, (seg_start_in_region, seg_end_in_region) in enumerate(valid_indices):
            seg_positions = region_positions[seg_start_in_region:seg_end_in_region]
            seg_smooth = region_smooth[seg_start_in_region:seg_end_in_region]
    
            global_seg_start_idx = region_start_idx + seg_start_in_region
            global_seg_end_idx = region_start_idx + seg_end_in_region
            
            segment_info = self.validate_and_classify_segment(
                seg_positions, seg_smooth, global_seg_start_idx, global_seg_end_idx
            )
            if segment_info:
                valid_segments.append(segment_info)
        
        print(f"最终验证通过 {len(valid_segments)} 个片段")
        return valid_segments    positions: np.ndarray,
                                    depth: int = 0,
                                    offset: int = 0,
                                    trace: Optional[List[Dict]] = None) -> Tuple[List[int], List[Dict]]:
        """
        CBS递归分割 + 记录trace信息
        """
        if trace is None:
            trace = []

        n = len(smooth_values)
        if n < 2 * self.min_sites_per_segment:
            return [], trace

        best_stat = 0.0
        best_p = 1.0
        best_k = -1
        cand_stats = []

        for k in range(self.min_sites_per_segment, n - self.min_sites_per_segment + 1):
            left = smooth_values[:k][~np.isnan(smooth_values[:k])]
            right = smooth_values[k:][~np.isnan(smooth_values[k:])]
            if len(left) > 1 and len(right) > 1 and np.var(left) > 0 and np.var(right) > 0:
                t, p = stats.ttest_ind(left, right, equal_var=False)
                cand_stats.append((offset + k, t, p))
                if p < self.alpha and abs(t) > abs(best_stat):
                    best_stat = t
                    best_p = p
                    best_k = k
            else:
                cand_stats.append((offset + k, np.nan, np.nan))

        trace.append({
            'depth': depth,
            'global_start_idx': offset,
            'global_end_idx': offset + n,
            'candidate_stats': cand_stats,
            'best_global_k': (offset + best_k) if best_k != -1 else None,
            'best_t': best_stat if best_k != -1 else None,
            'best_p': best_p if best_k != -1 else None
        })

        if best_k == -1:
            return [], trace

        left_cp, trace = self.cbs_core_segmentation_trace(
            smooth_values[:best_k], positions[:best_k], depth=depth+1, offset=offset, trace=trace
        )
        right_cp, trace = self.cbs_core_segmentation_trace(
            smooth_values[best_k:], positions[best_k:], depth=depth+1, offset=offset+best_k, trace=trace
        )

        changepoints = [offset + best_k] + left_cp + right_cp
        return sorted(set(changepoints)), tracedef plot_segments_track(self, df: pd.DataFrame, changepoints: List[int], ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(df['position'], df['smooth_value'], 'o-', ms=2, alpha=0.6)
        for cp in changepoints:
            ax.axvline(df['position'].iloc[cp], color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Genomic Position (bp)')
        ax.set_ylabel('Smooth Methylation')
        ax.set_title('CBS Segmentation')
        return ax

    def plot_cbs_scan(self, trace: List[Dict], df: pd.DataFrame, show_all=True, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,2))
        cmap = plt.get_cmap('tab10')
        depths = sorted({t['depth'] for t in trace}) if show_all else [0]

        for i, d in enumerate(depths):
            for tdict in trace:
                if tdict['depth'] != d:
                    continue
                xs = [df['position'].iloc[k] for k, _, _ in tdict['candidate_stats']]
                ys = [abs(t) if t is not None and not np.isnan(t) else 0 for _, t, _ in tdict['candidate_stats']]
                ax.plot(xs, ys, label=f"depth {d}", color=cmap(i))
                if tdict['best_global_k'] is not None:
                    bx = df['position'].iloc[tdict['best_global_k']]
                    ax.axvline(bx, color=cmap(i), linestyle='--', alpha=0.7)

        ax.set_ylabel('|t|')
        ax.set_xlabel('Genomic position')
        if show_all:
            ax.legend()
        ax.set_title('CBS candidate split scan')
        return ax

    def plot_cbs_recursion_tree(self, trace: List[Dict], df: pd.DataFrame, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,3))
        for tdict in trace:
            d = tdict['depth']
            start = df['position'].iloc[tdict['global_start_idx']]
            end = df['position'].iloc[tdict['global_end_idx']-1]
            ax.hlines(y=d, xmin=start, xmax=end, lw=4, color='0.7')
            if tdict['best_global_k'] is not None:
                bx = df['position'].iloc[tdict['best_global_k']]
                ax.vlines(x=bx, ymin=d-0.3, ymax=d+0.3, color='r')
        ax.set_ylim(-1, max(t['depth'] for t in trace)+1)
        ax.set_xlabel('Genomic position')
        ax.set_ylabel('CBS depth')
        ax.set_title('CBS recursion path')
        return ax

    def animate_cbs(self, df: pd.DataFrame, trace: List[Dict], interval=1000, save=None):
        from matplotlib.animation import FuncAnimation
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(df['position'], df['smooth_value'], 'o-', ms=2, alpha=0.5)
        lines = []
        text = ax.text(0.02,0.95,'', transform=ax.transAxes, va='top')

        def update(frame):
            tdict = trace[frame]
            if tdict['best_global_k'] is not None:
                bx = df['position'].iloc[tdict['best_global_k']]
                line = ax.axvline(bx, color='r', alpha=0.6)
                lines.append(line)
            text.set_text(f"Step {frame+1}/{len(trace)} depth={tdict['depth']}")
            return lines + [text]

        anim = FuncAnimation(fig, update, frames=len(trace), interval=interval, blit=False, repeat=False)
        if save:
            anim.save(save, dpi=150)
        return anim


def process_chromosome_segmentation(input_dir: str, chromosome: str, segmentation_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理单个染色体的分割任务 - 这是缺失的函数
    
    Args:
        input_dir: 输入目录路径
        chromosome: 染色体名称
        segmentation_params: 分割参数字典
        
    Returns:
        包含染色体分割结果的字典
    """
    try:
        # 构建smooth文件路径
        smooth_file = os.path.join(input_dir, "smooth", f"{chromosome}.csv")
        
        if not os.path.exists(smooth_file):
            return {
                'chromosome': chromosome,
                'segments': pd.DataFrame(),
                'segmentation_stats': {},
                'status': 'error',
                'error': f'Smooth file not found: {smooth_file}'
            }
        
        # 创建分割器实例
        segmenter = BiologicallyInformedSegmentation(**segmentation_params)
        
        # 加载数据
        df = segmenter.load_data(smooth_file)
        
        # 执行分割
        segments_df = segmenter.segment_methylation_data(df)
        
        # 添加染色体信息
        segments_df['chromosome'] = chromosome
        
        return {
            'chromosome': chromosome,
            'segments': segments_df,
            'segmentation_stats': segmenter.segmentation_stats,
            'status': 'success',
            'error': None
        }
        
    except Exception as e:
        return {
            'chromosome': chromosome,
            'segments': pd.DataFrame(),
            'segmentation_stats': {},
            'status': 'error',
            'error': str(e)
        }

@njit
def welch_ttest_numba(left: np.ndarray, right: np.ndarray) -> Tuple[float, float]:
    n1 = len(left)
    n2 = len(right)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0  # 无法检验

    mean1 = np.mean(left)
    mean2 = np.mean(right)
    var1 = np.var(left, ddof=1)
    var2 = np.var(right, ddof=1)

    if var1 == 0.0 and var2 == 0.0:
        return 0.0, 1.0

    t_stat = (mean1 - mean2) / np.sqrt(var1/n1 + var2/n2)

    # 自由度近似 (Welch–Satterthwaite equation)
    df = ((var1/n1 + var2/n2)**2) / ((var1**2)/((n1**2)*(n1 - 1)) + (var2**2)/((n2**2)*(n2 - 1)))

    # 这里使用正态近似而不是精确t分布计算p值以兼容numba
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))

    return t_stat, p

# 在 denovo_segmentation 函数中添加结果存储和简化返回
def denovo_segmentation(input_dir: str, 
                       exclude_chroms: Optional[List[str]] = None,
                       window_size: int = 3000,
                       step_size: int = 500,
                       cpu: int = 10,
                       segmentation_params: Optional[Dict[str, Any]] = None,
                       save_results: bool = True,
                       output_dir: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, List[List[int]]]]:
    """
    在所有染色体中进行de novo识别分割区域
    
    Args:
        input_dir: 输入目录路径，应包含smooth和data文件夹
        exclude_chroms: 要排除的染色体列表
        window_size: 窗口大小
        step_size: 步长
        cpu: 使用的CPU核心数
        segmentation_params: 分割参数字典，如果为None则使用默认参数
        save_results: 是否保存详细结果到文件
        output_dir: 输出目录，如果为None则使用input_dir
    
    Returns:
        Tuple[Dict[str, Any], Dict[str, List[List[int]]]]: (详细结果字典, 简化的segments字典)
    """
    
    # 设置输出目录
    if output_dir is None:
        output_dir = input_dir
    
    # 检查必要的文件夹是否存在
    smooth_dir = os.path.join(input_dir, "smooth")
    
    if not os.path.exists(smooth_dir):
        print(f"Could not find smooth folder at {smooth_dir}, please run smooth first.")
        return {}, pd.DataFrame()
    
    # 创建输出目录
    if save_results:
        segments_output_dir = os.path.join(output_dir, "segments")
        os.makedirs(segments_output_dir, exist_ok=True)
    
    # 设置默认分割参数
    if segmentation_params is None:
        segmentation_params = {
            'alpha': 0.01,
            'min_sites_per_segment': 2,
            'max_site_distance': 10000,
            'max_segment_length': 10_000_000,
            'min_segment_length': 100,
            'max_merge_gap': 500,
            'dropout_tolerance': 0.3,
            'sparse_region_threshold': 0.1,
            'fully_methylated_threshold': 0.8,
            'fully_unmethylated_threshold': 0.2,
            'variance_threshold': 0.05,
            'gap_threshold': 10000
        }
    
    # 获取所有染色体文件
    csv_files = glob.glob(os.path.join(smooth_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {smooth_dir}")
        return {}, {}
    
    # 提取染色体名称
    chromosomes = []
    for csv_file in csv_files:
        chrom = os.path.basename(os.path.splitext(csv_file)[0])
        if exclude_chroms and chrom in exclude_chroms:
            print(f"Skipping chromosome {chrom}")
            continue
        chromosomes.append(chrom)
    
    print(f"Processing {len(chromosomes)} chromosomes with {min(cpu, 24)} CPU cores")
    
    # 限制CPU核心数
    cpu = min(cpu, 24)
    
    # 使用多进程处理
    features = {}
    results = []
    
    with mp.Pool(processes=cpu) as pool:
        # 提交所有任务
        for chrom in chromosomes:
            result = pool.apply_async(
                process_chromosome_segmentation,
                args=(input_dir, chrom, segmentation_params)
            )
            results.append((chrom, result))
        
        # 收集结果
        for chrom, result in results:
            try:
                features[chrom] = result.get()
                print(f"Completed segmentation for chromosome {chrom}")
            except Exception as e:
                print(f"Error retrieving result for {chrom}: {e}")
                features[chrom] = {
                    'chromosome': chrom,
                    'segments': pd.DataFrame(),
                    'status': 'error',
                    'error': str(e)
                }
    
    # 保存详细结果和收集简化结果
    simple_segments_dict = {}
    
    for chrom, result in features.items():
        if result['status'] == 'success' and not result['segments'].empty:
            segments_df = result['segments']
            
            # 保存详细的segments结果
            if save_results:
                segments_file = os.path.join(segments_output_dir, f"{chrom}_segments.csv")
                segments_df.to_csv(segments_file, index=False)
                
                # 保存统计信息
                stats_file = os.path.join(segments_output_dir, f"{chrom}_stats.json")
                import json
                with open(stats_file, 'w') as f:
                    json.dump(result['segmentation_stats'], f, indent=2, default=str)
            
            # 创建简化的segments信息 - 字典格式
            chrom_segments = []
            for _, row in segments_df.iterrows():
                chrom_segments.append((int(row['start_pos']), int(row['end_pos'])))
            
            simple_segments_dict[chrom] = chrom_segments
    
    # 保存简化的总结果
    if save_results and simple_segments_dict:
        # 将字典格式转换为DataFrame保存
        simple_segments_list = []
        for chrom, segments in simple_segments_dict.items():
            for start_pos, end_pos in segments:
                simple_segments_list.append({
                    'chromosome': chrom,
                    'start_pos': start_pos,
                    'end_pos': end_pos
                })
        
        simple_segments_df = pd.DataFrame(simple_segments_list)
        simple_segments_file = os.path.join(output_dir, "segments_summary.csv")
        simple_segments_df.to_csv(simple_segments_file, index=False)
        print(f"Saved simple segments summary to: {simple_segments_file}")
    
    # 保存总体统计信息
    if save_results:
        # 生成总结报告
        total_segments = 0
        successful_chroms = 0
        failed_chroms = 0
        
        for chrom, result in features.items():
            if result['status'] == 'success':
                successful_chroms += 1
                total_segments += result['segmentation_stats']['total_segments']
            else:
                failed_chroms += 1
        
        summary_stats = {
            'total_chromosomes_processed': len(chromosomes),
            'successful_chromosomes': successful_chroms,
            'failed_chromosomes': failed_chroms,
            'total_segments_identified': total_segments,
            'average_segments_per_chromosome': total_segments/successful_chroms if successful_chroms > 0 else 0,
            'segmentation_parameters': segmentation_params
        }
        
        summary_file = os.path.join(output_dir, "segmentation_summary.json")
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        print(f"Saved segmentation summary to: {summary_file}")
    
    # 生成控制台总结报告
    total_segments = 0
    successful_chroms = 0
    failed_chroms = 0
    
    for chrom, result in features.items():
        if result['status'] == 'success':
            successful_chroms += 1
            total_segments += result['segmentation_stats']['total_segments']
        else:
            failed_chroms += 1
    
    print(f"\n=== De Novo Segmentation Summary ===")
    print(f"Total chromosomes processed: {len(chromosomes)}")
    print(f"Successful: {successful_chroms}")
    print(f"Failed: {failed_chroms}")
    print(f"Total segments identified: {total_segments}")
    print(f"Average segments per chromosome: {total_segments/successful_chroms if successful_chroms > 0 else 0:.2f}")
    
    if save_results:
        print(f"Results saved to: {output_dir}")
        print(f"  - Detailed segments: {os.path.join(output_dir, 'segments')}")
        print(f"  - Simple summary: {os.path.join(output_dir, 'segments_summary.csv')}")
    
    return features, simple_segments_dict