import scMethtools.logging as logg
from collections import namedtuple

import re

def detect_sep(line):
    if '\t' in line:
        return '\t'
    elif ',' in line:
        return ','
    else:
        return ' '

def is_int(x):
    try:
        return float(x).is_integer()
    except:
        return False

def is_float(x):
    try:
        float(x)
        return True
    except:
        return False
    
class CoverageFormat(
    namedtuple('CoverageFormat', ['chrom', 'pos', 'meth', 'umeth', 'context', 'coverage', 'sep', 'header'])):
    """Describes the columns in the coverage file.
    chrom, pos, meth, umeth, context, coverage, sep, header

    Params:
    chrom (int): 染色体列索引
    pos (int): 位置列索引
    meth (int): 甲基化列索引
    umeth (int): 未甲基化列索引
    context (int): 上下文列索引
    coverage (bool): 是否使用覆盖度计数
    sep (str): 分隔符
    header (bool): 是否有表头
   
    """
    def remove_chr_prefix(self):
        """Remove "chr" or "CHR" etc. from chrom."""
        return self._replace(chrom=self.chrom.lower().lstrip("chr"))
    
def format_check(bed_file, sample_size=10000):
    """_summary_

    Args:
        bed_file (_str_): bed file path
        sample_size (int, optional): Defaults to 10000.
    Returns:
        _str_: format string or format name
    Example:
        format_check("test.bed", 1000)
  
    """
    
    chrom_col = pos_col = meth_col = umeth_col = value_col = context_col = None
    sep = None
    header = False
    coverage = False

    try:
        with open(bed_file, 'r') as f:
            first_lines = []
            for _ in range(sample_size):
                line = f.readline()
                if not line:
                    break
                if sep is None:
                    sep = detect_sep(line)
                first_lines.append(line.strip().split(sep))
    except FileNotFoundError:
        raise ValueError(f"The file {bed_file} was not found.")
    
    if not first_lines:
        raise ValueError("File is empty or unreadable.")

    # 判断 header（首行是否含有英文字符）
    if any('chrom' in val.lower() for val in first_lines[0]):
        header = True
        data_rows = first_lines[1:]
    else:
        data_rows = first_lines

    n_cols = len(data_rows[0])
    columns = list(zip(*data_rows))  # 转置，变成列结构

    # === 染色体列 ===
    for idx, col in enumerate(columns):
        if all(re.match(r'^chr[\w\d]+$|^[1-9XYM]+$', c, re.IGNORECASE) for c in col[:50]):
            chrom_col = idx
            break
    if chrom_col is None:
        chrom_col = 0  # 默认设为第一列

    # === 位置列：大于100的整数 ===
    for idx, col in enumerate(columns):
        if all(is_int(c) and int(float(c)) > 100 for c in col[:100] if c.strip().isdigit()):
            pos_col = idx
            break

    # === context列 ===
    for idx, col in enumerate(columns):
        if all(c.upper() in ['CG', 'CHG', 'CHH', 'CGN', 'CpG'] for c in col[:100]):
            context_col = idx
            break

    # === value列：出现小数，或全是0/1 ===
    for idx, col in enumerate(columns):
        if idx == 0:
            continue  # 第一列不可能是 value 列
        if not all(is_float(x) for x in col[:100]):
            continue  # 非float列，跳过

        floats = [float(x) for x in col[:100]]
        if any('.' in str(x) for x in col[:100]) or all(x in [0.0, 1.0] for x in floats):
            value_col = idx
            break
        
    # Raise error if unable to determine value column
    if value_col is None:
        raise ValueError(
            "Unable to automatically detect the methylation value column (usually a proportion).\n"
            "Please check the file format manually or specify a custom order string like:\n"
            "  custom order string: \"1:2:3:4c:5:\\t:0\"\n"
            "  (chrom:position:methylated_C:coverage(c)/unmethylated_C(u):context:sep:header)\n"
            "Note: The value column typically contains decimal numbers or binary values (0/1), "
            "and is unlikely to be the first column."
        )

       # 改进后的 position 列判断
    for idx, col in enumerate(columns):
        # 尝试转换前 100 个值为 int，失败就跳过
        parsed = []
        for c in col[:100]:
            try:
                val = float(c)
                if val.is_integer():
                    parsed.append(int(val))
            except:
                continue
        # 判断条件：有效数值数量足够，值大于100，且不是染色体列
        if len(parsed) >= 50 and all(x > 100 for x in parsed):
            pos_col = idx
            break
            
   # 所有正整数列
    int_cols = []
    for idx, col in enumerate(columns):
        # 确保不是 value_col，且所有值都是正整数
        if idx != value_col and all(is_int(x) and int(float(x)) >= 0 for x in col[:100]):
            int_cols.append((idx, [int(float(x)) for x in col[:100]]))

    # meth / unmeth 列推断
    if len(int_cols) >= 2:
        # 确保不是 position 列
        valid_int_cols = [col for col in int_cols if col[0] != pos_col and col[0] != chrom_col]
        if len(valid_int_cols) >= 2:
            c1_idx, c1_vals = valid_int_cols[0]
            c2_idx, c2_vals = valid_int_cols[1]

           # 检查甲基化与未甲基化列的关系
            # 假设较小的是meth列，较大的未甲基化列
            if sum(c1_vals) <= sum(c2_vals):
                meth_idx, unmeth_idx = c1_idx, c2_idx
            else:
                meth_idx, unmeth_idx = c2_idx, c1_idx
            
            # 若有value列，尝试验证 coverage = True
            if value_col is not None:
                meth_vals = [int(float(x)) for x in columns[meth_idx][:1000]]
                unmeth_vals = [int(float(x)) for x in columns[unmeth_idx][:1000]]
                value_vals = [float(x) for x in columns[value_col][:1000]]
                matched = 0
                for m, u, v in zip(meth_vals, unmeth_vals, value_vals):
                    total = m + u
                    if total == 0:
                        continue
                    ratio = m / total
                    if abs(ratio - v) < 0.01:
                        matched += 1
                if matched > 500:  # 半数以上匹配
                    coverage = True
                else:
                    coverage = False

            meth_col, umeth_col = meth_idx, unmeth_idx
    # 格式化列顺序判断
    format_orders = {
        'bismark': (0, 1, 3, 4, 5, False, "\t", False),
        'bsseeker2': (0, 2, 6, 7, 3, True, "\t", False),
        'bsseeker': (0, 2, 6, 7, 3, True, "\t", False),
        'methylpy': (0, 2, 6, 7, 3, True, "\t", False)
    }
    print({
        "chrom_col": chrom_col,
        "pos_col": pos_col,
        "context_col": context_col,
        "meth_col": meth_col,
        "umeth_col": umeth_col,
        "value_col": value_col,
        "sep": sep,
        "header": header,
        "coverage": coverage
    })
            
    # 检查是否是预设格式
    for format_name, order in format_orders.items():
        if (chrom_col, pos_col, meth_col, umeth_col, context_col, coverage, sep, header) == order[:8]:
            return format_name  # 返回符合的格式名
    # 根据 coverage 判断 unmeth 后面应该是 'c' 还是 'u'
    unmeth_suffix = 'c' if coverage else 'u'
    header_suffix = 1 if header else 0
    # 返回自定义格式字符串
    custom_order = f"{chrom_col + 1}:{pos_col + 1}:{meth_col + 1}:{umeth_col + 1}{unmeth_suffix}:{context_col + 1}:{sep}:{header_suffix}"
    return f"Custom order string: \"{custom_order}\""
    

def _custom_format(format_string):
    """
    Create from user specified string. Adapted from scbs function
    
     Args:
        format_string: e.g chrom:pos:meth:coverage/unmeth:context:sep:header
    """
    try:
        parts = format_string.lower().split(":")
        if len(parts) != 7:
            raise Exception("Invalid number of ':'-separated values in custom input format")
         # 使用列表推导式简化索引转换
        indices = [int(p) - 1 for p in parts[:3]]
        chrom, pos, meth = indices
         # 使用正则表达式解析coverage/unmeth
        import re
        match = re.match(r'(\d+)([cuCU])', parts[3])
        if not match:
            raise Exception(
                "The 4th column of a custom input format must contain an integer and "
                "either 'c' for coverage or 'u' for unmethylated counts (e.g. '4c'), "
                f"but you provided '{format_string[3]}'."
            )
        umeth, info = match.groups()
        umeth = int(umeth) - 1 
        coverage = info == 'c'
        #umeth = int(format_string[3][0:-1]) - 1
        context = int(parts[4])-1
        sep = "\t" if parts[5].lower() in ("\\t", "tab", "t") else parts[5]
        header = bool(int(parts[6]))
        return chrom, pos, meth, umeth, context, coverage, sep, header
    except (ValueError, IndexError) as e:
        raise ValueError(f"Format parsing error : {str(e)}")


def reorder_columns_by_index(pipeline):
    """_summary_

    Args:
        pipeline (_str_): software pipeline used to generate the methylation coverage file
        software name or custom order string can be accepted
        software name: bismark, bsseeker2, methylpy
        custom order string: "1:2:3:4c:5:\t:0" (chrom:position:methylated_C:coverage(c)/unmethylated_C(u):context:sep:header) note: 1-based index                 
    Returns:
        _tuple_: the order of the columns in the coverage file
    """
    pipeline = pipeline.lower()
    # bismark format: chr1 10004 + 0 0 CHH CCC
    # bisseeker format: chr1 C 10060 CHH CT 1.00 1 1
    # methylpy format:
    # order: chr pos(1-based) meth unmeth context coverage(all-C) sep header
    format_orders = {
        'bismark': CoverageFormat(0, 1, 3, 4, 5, False, "\t", False),
        'bsseeker2': CoverageFormat(0, 2, 6, 7, 3, True, "\t", False),
        'bsseeker': CoverageFormat(0, 2, 6, 7, 3, True, "\t", False),
        'methylpy': CoverageFormat(0, 2, 6, 7, 3, True, "\t", False)
    }
    # 根据指定的格式调整列顺序
    if pipeline in format_orders:
        logg.info("## BED column format: " + pipeline)
        new_order = format_orders[pipeline]
        return new_order
    elif ":" in pipeline:
        logg.info("## BED column format:  Custom")
        new_order = _custom_format(pipeline)
        return new_order
    else:
        raise ValueError(f"Invalid format type or custom order {pipeline}.")