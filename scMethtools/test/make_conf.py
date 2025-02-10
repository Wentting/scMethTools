# Description: Generate a configuration file for scMethtools.
#
# Usage: python make_conf.py conf_file
#
# Parameters:
#     conf_file: str
#         Path to the configuration file.
#               
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List
import gzip
import configparser




def main():
    parser = get_parser()
    args = parser.parse_args()
    # 解析用户提供的设置

    conf = modify_config(args.input, args.outfile, args)
    #write_conf(args.outfile, conf)
    
# def locate_static_conf():
#     package_path = os.path.dirname(sys.modules['your_package_name'].__file__)
#     conf_path = os.path.join(package_path, 'config/tmp.conf')
#     return conf_path

def locate_static_conf():
    # 假设当前文件是包中的某个模块
    current_dir = Path(__file__).parent
    conf_path = current_dir / "config/tmp.conf"
    return conf_path

def make_conf(args: argparse.Namespace) -> List[str]:
    """
    Generate a configuration file for scMethtools.
    """

    conf = [f"reference = reference/{Path(args.reference).name}"]
    
    if args.benchmark_mode:
        conf.append("benchmark_mode = true")
    if args.include_file:
        conf.append(f"include {args.include_file}")

    return conf

def modify_config(input_file: str, output_file: str, args: argparse.Namespace):
    """
    修改配置文件并生成新的文件。

    :param input_file: 原始配置文件路径
    :param output_file: 修改后的配置文件路径
    :param args: 用户通过命令行传入的参数，作为 key-value 修改配置文件
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    # 检查输入文件是否存在
    if not input_path.exists():
        raise FileNotFoundError(f"配置文件 {input_file} 不存在。")

    # 初始化配置解析器
    config = configparser.ConfigParser()
    config.read(input_path)
    
    for key, value in vars(args).items():  # 将 args 转换为字典并迭代
        if value is None:  # 跳过未指定的参数
            continue
        section, option = key.split(".", 1) if "." in key else ("DEFAULT", key)
        if section not in config:
            config.add_section(section)
        config[section][option] = str(value)  # 配置值必须是字符串

    # 写入新的配置文件
    try:
        with open(output_path, "w") as f:
            config.write(f)
        print(f"The modified configuration file has been saved to {output_file}")
    except IOError as e:
        raise IOError(f"Generate config file ERROE：{e}")

def write_conf(outfile: str, conf: List[str]):
    with open(outfile, "w") as f:
        for line in conf:
            f.write(line + "\n")
    return

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Path of config files", default="./tmp.conf", required=False
    )
    parser.add_argument(
        "-o",
        "--outfile",
        help="Name of file to write generated conf to",
        default="methDot.conf",
    )   
    parser.add_argument(
        "-t", "--num-threads", help="Number of threads", type=int, default=8
    )
    parser.add_argument(
        "-j", "--num-jobs", help="Number of jobs", type=int, default=3
    )
    parser.add_argument(
        "-r", "--reference", help="File path of reference fasta", required=True
    )
    parser.add_argument(
        "-m", "--end-mod", help="Sequencing layout,Single-end(SE) or Paired-end(PE)",  type=str, default='PE', required=True
    )
    parser.add_argument(
        "--output-dir", help="Result path", default="./", required=True
    )
    parser.add_argument(
        "--path-to-bismark",   type=Path, required=False
    )
    parser.add_argument(
        "--path-to-trimmomatic",   type=Path, required=False
    )
    return parser

if __name__ == "__main__":
    main()


