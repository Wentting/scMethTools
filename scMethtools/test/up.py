
import os
import subprocess
from ..io import *
import scMethtools.logging as logg
import glob as glob

def get_remaining_files(file_list, finished_file_list):
    # 从文件中读取已完成的文件路径
    with open(finished_file_list, "r") as f:
        finished_files = set(f.read().splitlines())
    
    # 过滤出尚未完成的文件
    remaining_files = [file for file in file_list if file not in finished_files]
    return remaining_files

def prepare_ref_genome(ref_genome, output_dir):
    """准备参考基因组"""

    if not os.path.exists(ref_genome):
        raise FileNotFoundError(f"Reference genome not found: {ref_genome}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    title("Preparing bisulfite ref genome... ")
    
    # 复制参考基因组到输出目录
    ref_genome_name = os.path.basename(ref_genome)
    ref_genome_out = os.path.join(output_dir, ref_genome_name)
    run_command(f"cp {ref_genome} {ref_genome_out}", "Copy Reference Genome")
    return ref_genome_out

def format_convert(input_file, output_file):
    """convert the format of fasta file to fastq file
    fastq-dump  --split-3 /data40T/zongwt/singledata/GSE81233_H/raw/SRR5241070.sra -O /data40T/zongwt/singledata/GSE81233_H/fastq/

    Args:
        input_file (_type_): _description_
        output_file (_type_): _description_
    """
    title("Converting RAW file from .SRA to .fastq format... ")
    try:
        run_command(f"fastq-dump  --split-3 {input_file} -O {output_file}", "Format Convert")
    except Exception as e:
        logg.warn(e.message)

def run_fastqc(fastq_path):
    """运行 FastQ"""
    """
    
    fastqc /data40T/zongwt/singledata/GSE81233_H/fastq/SRR5241070_1.fastq -o /data40T/zongwt/singledata/GSE81233_H/fastq/
    """
    title("Converting RAW file from .SRA to .fastq format... ")
    fastq_files = glob.glob(f"{fastq_path}/*.fastq.gz")
    run_command("Check Quality" , f"fastq-dump  --split-3 {fastq_path} -O {output_file}")
        

def run_command(command, step_name):
    """运行命令行命令并记录日志"""
    title(f"\n++ Running {step_name}: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logg.warn(f"Error in {step_name}: {result.stderr}")
        raise Exception(f"Step {step_name} failed.")
    print(result.stdout)

def read_config(section, key):
    """读取配置文件中的状态，模拟实现"""
    # 示例：从文件或字典中读取状态
    # 实际代码可以用 ConfigParser 或 JSON 解析
    config = {
        "STATUS": {
            "st_trim": "0",
            "st_fastq": "0",
            "st_bismark": "0",
            "st_bisdedup": "0",
            "st_dedsort": "0"
        },
        "EMAIL": {"active": "false"},
        "Others": {"tmp_meth_out": "output/results"}
    }
    return config.get(section, {}).get(key, "0")

def write_config(section, key, value):
    """写入配置文件中的状态，模拟实现"""
    print(f"Updating config: [{section}] {key} = {value}")
    # 实际实现需要将值写回配置文件

def run_trim_galore(fastq_file, output_dir):
    """运行 Trim Galore"""
    run_command(f"trim_galore {fastq_file} -o {output_dir}", "Trim Galore")
    write_config("STATUS", "st_trim", "2")

def run_fastqc(fastq_file, output_dir):
    """运行 FastQC"""
    run_command(f"fastqc {fastq_file} -o {output_dir}", "FastQC")
    write_config("STATUS", "st_fastq", "2")

def run_bismark(fastq_file, reference_genome, output_dir):
    """运行 Bismark 比对"""
    run_command(f"bismark {reference_genome} -o {output_dir} {fastq_file}", "Bismark")
    write_config("STATUS", "st_bismark", "2")

def run_bismark_dedup(bam_file, output_dir):
    """运行 Bismark Deduplication"""
    run_command(f"deduplicate_bismark -o {output_dir} {bam_file}", "Bismark Deduplication")
    write_config("STATUS", "st_bisdedup", "2")

def run_methimpute(bam_file, output_dir):
    """运行 Methimpute"""
    run_command(f"methimpute --input {bam_file} --output {output_dir}", "Methimpute")
    write_config("STATUS", "st_dedsort", "2")

def main():
    # 输入文件和目录
    raw_fastq = "path/to/raw_data.fastq"
    reference_genome = "path/to/reference_genome.fa"
    output_dir = "output"
    trimmed_fastq = os.path.join(output_dir, "trimmed.fastq")
    bam_file = os.path.join(output_dir, "aligned.bam")
    os.makedirs(output_dir, exist_ok=True)

    # 按状态检查和运行各步骤
    if read_config("STATUS", "st_trim") != "2":
        run_trim_galore(raw_fastq, output_dir)

    if read_config("STATUS", "st_fastq") != "2":
        run_fastqc(trimmed_fastq, output_dir)

    if read_config("STATUS", "st_bismark") != "2":
        run_bismark(trimmed_fastq, reference_genome, output_dir)

    if read_config("STATUS", "st_bisdedup") != "2":
        run_bismark_dedup(bam_file, output_dir)

    if read_config("STATUS", "st_dedsort") != "2":
        run_methimpute(bam_file, output_dir)

    print("Processing files are finished. Results are in:", read_config("Others", "tmp_meth_out"))

if __name__ == "__main__":
    main()
