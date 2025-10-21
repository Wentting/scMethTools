import pandas as pd
import numpy as np
from scipy import stats
import warnings
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

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
        """
        合并相邻且满足合并条件的segments。
        优化：在合并后，从原始DataFrame重新计算合并片段的统计量，确保准确性。
        """
        if not segments:
            return []
        if self._original_df is None:
            warnings.warn("Original DataFrame not set. Cannot accurately re-calculate metrics for merged segments.")
            return segments # 无法重新计算，返回原 segments

        # 按起始位置排序
        sorted_segments = sorted(segments, key=lambda x: x['start_pos'])
        merged = [sorted_segments[0]]

        for i in range(1, len(sorted_segments)):
            current_seg = sorted_segments[i]
            last_merged_seg = merged[-1]

            # 检查是否相邻（允许小间隙）且类型相同
            # 相邻条件：当前片段的起始位置 <= 上一个合并片段的结束位置 + 允许的最大合并间隙
            if (current_seg['start_pos'] <= last_merged_seg['end_pos'] + self.max_merge_gap) and \
               current_seg['type'] == last_merged_seg['type']:

                # 确定合并后的全局索引范围
                merged_global_start_idx = last_merged_seg['start_idx']
                merged_global_end_idx = current_seg['end_idx']

                # 从原始DataFrame中提取合并后的完整数据点
                # 注意：这里假设原始DataFrame的索引与全局CpG位点索引一致
                merged_positions_raw = self._original_df['position'].iloc[merged_global_start_idx:merged_global_end_idx].values
                merged_smooth_values_raw = self._original_df['smooth_value'].iloc[merged_global_start_idx:merged_global_end_idx].values

                # 重新计算合并后片段的所有统计量
                recalculated_metrics = self._calculate_segment_metrics(
                    merged_positions_raw, merged_smooth_values_raw
                )
                
                # 检查合并后的片段是否仍然满足生物学长度约束
                if (self.min_segment_length <= recalculated_metrics['length_bp'] <= self.max_segment_length and
                    recalculated_metrics['num_sites'] >= self.min_sites_per_segment and
                    (recalculated_metrics['num_sites'] == 1 or recalculated_metrics['max_site_gap'] <= self.max_site_distance)):
                    
                    # 更新合并片段的信息
                    merged[-1] = {
                        'start_idx': merged_global_start_idx,
                        'end_idx': merged_global_end_idx,
                        'start_pos': int(merged_positions_raw[0]),
                        'end_pos': int(merged_positions_raw[-1]),
                        'type': last_merged_seg['type'] # 类型保持不变
                    }
                    merged[-1].update(recalculated_metrics) # 更新所有重新计算的指标
                else:
                    # 合并后不符合约束，则不合并，将当前片段添加到结果中
                    merged.append(current_seg)
            else:
                # 不相邻或类型不同，直接添加当前片段
                merged.append(current_seg)
        return merged

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


    def plot_segmentation_with_constraints(self, df: pd.DataFrame, segments_df: pd.DataFrame, save_path: Optional[str] = None):
        """可视化分割结果和约束"""
        if segments_df.empty:
            print("没有segments可供绘制。")
            return
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        ax1 = axes[0]
        ax1.scatter(df['position'], df['smooth_value'], alpha=0.3, s=1, color='gray')
        colors = {
            'variable': 'blue', 'fully_methylated': 'red', 'fully_unmethylated': 'green',
            'low_variance': 'orange', 'sparse': 'purple', 'cpg_island': 'cyan'
        }
        for _, segment in segments_df.iterrows():
            color = colors.get(segment['type'], 'black')
            ax1.axhspan(segment['mean_smooth'] - 0.02, segment['mean_smooth'] + 0.02,
                        xmin=(segment['start_pos'] - df['position'].min()) / (df['position'].max() - df['position'].min()),
                        xmax=(segment['end_pos'] - df['position'].min()) / (df['position'].max() - df['position'].min()),
                        color=color, alpha=0.7)
        ax1.set_ylabel('Smooth Value')
        ax1.set_title('Segmentation Results with Biological Constraints')
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        lengths = segments_df['length_bp'] / 1000
        ax2.hist(lengths, bins=min(30, len(lengths) // 5) if len(lengths) > 0 else 1, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=self.max_segment_length/1000, color='red', linestyle='--', label=f'Max Length: {self.max_segment_length/1000:.0f}kb')
        ax2.axvline(x=self.min_segment_length/1000, color='red', linestyle='--', label=f'Min Length: {self.min_segment_length/1000:.1f}kb')
        ax2.set_xlabel('Segment Length (kb)')
        ax2.set_ylabel('Count')
        ax2.set_title('Segment Length Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[2]
        ax3.hist(segments_df['num_sites'], bins=min(20, len(segments_df['num_sites']) // 5) if len(segments_df['num_sites']) > 0 else 1, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.axvline(x=self.min_sites_per_segment, color='red', linestyle='--', label=f'Min Sites: {self.min_sites_per_segment}')
        ax3.set_xlabel('Number of Sites per Segment')
        ax3.set_ylabel('Count')
        ax3.set_title('Sites per Segment Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

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
        """
        核心方法: 对甲基化数据执行生物学知情分割。
        这是一个多阶段过程，结合了统计学分割和生物学约束过滤。
        """
        # 存储原始DataFrame的引用，以便在_merge_adjacent_segments中重新计算统计量
        self._original_df = df.copy() # 使用copy避免修改原始df

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

            changepoints_in_region = self.cbs_core_segmentation(region_smooth, region_positions)

            segment_indices = []
            if not changepoints_in_region:
                segment_indices.append((0, len(region_smooth)))
            else:
                current_seg_start = 0
                for cp_idx in changepoints_in_region:
                    segment_indices.append((current_seg_start, cp_idx))
                    current_seg_start = cp_idx
                segment_indices.append((current_seg_start, len(region_smooth)))

            for seg_start_in_region, seg_end_in_region in segment_indices:
                seg_positions = region_positions[seg_start_in_region:seg_end_in_region]
                seg_smooth = region_smooth[seg_start_in_region:seg_end_in_region]

                global_seg_start_idx = region_start_idx + seg_start_in_region
                global_seg_end_idx = region_start_idx + seg_end_in_region

                segment_info = self.validate_and_classify_segment(
                    seg_positions, seg_smooth, global_seg_start_idx, global_seg_end_idx
                )
                if segment_info:
                    all_segments_info.append(segment_info)

        # 步骤4: 尝试合并相邻的同质segments
        final_segments = self._merge_adjacent_segments(all_segments_info)

        # 步骤5: 生成统计信息
        self.generate_segmentation_stats(final_segments)

        return pd.DataFrame(final_segments)