import os
import subprocess
#from scMethtools.logging import logg
import configparser


def read_config(config_file, section, get_string):
    config = configparser.ConfigParser()
    config.read(config_file)
    val_str = config.get(section, get_string)
    return val_str

def show_config():
    print("\n" + "==" * 30)
    print("Here is summary of configuration parameters: \n")
    print("- RAW files location: " + read_config("GENERAL", "raw_dataset"))
    print("- Number and Size of the data-set: " + read_config("GENERAL", "number_of_dataset") + " Files and Total size: " + read_config("GENERAL", "dataset_size") + " Gigabyte")
    print("- The directory of results: " + read_config("GENERAL", "result_pipeline"))
    print("- Genome type: " + read_config("GENERAL", "genome_type"))
    print("- Genome folder location: " + read_config("GENERAL", "genome_ref"))
    print("     -- Genome Reference name: " + read_config("GENERAL", "genome_name"))
    print("- Paired End: " + read_config("GENERAL", "pairs_mode"))
    print("- Trimmomatic location: " + read_config("Trimmomatic", "trim_path"))
    print("     -- JAVA path: " + read_config("Trimmomatic", "java_path"))
    print("     -- ILLUMINACLIP: " + read_config("Trimmomatic", "name_adap") + ":" + read_config("Trimmomatic", "ill_clip"))
    print("     -- LEADING: " + read_config("Trimmomatic", "LEADING"))
    print("     -- TRAILING: " + read_config("Trimmomatic", "TRAILING"))
    print("     -- SLIDINGWINDOW: " + read_config("Trimmomatic", "SLIDINGWINDOW"))
    print("     -- MINLEN: " + read_config("Trimmomatic", "MINLEN"))
    print("     -- Number of Threads: " + read_config("Trimmomatic", "n_th"))   
    print("- QC-Fastq path: " + read_config("GENERAL", "fastq_path"))
    print("- Bismark parameters: " + read_config("Bismark", "bismark_path"))
    print("     -- scBS-Seq (--pbat)? " + read_config("Bismark", "single_cell"))
    if read_config("Bismark", "single_cell") == "true":
        if read_config("Bismark", "directional") == '':
            txt = "directional"
        else:
            txt = read_config("Bismark", "directional")
        print("     -- directional status: " + txt)
    print("     -- Nucleotide status: " + read_config("Bismark", "nucleotide"))
    print("     -- Number of Parallel: " + read_config("Bismark", "bis_parallel") + " Threads.")
    print("     -- Buffer size: " + read_config("Bismark", "buf_size") + " Gigabyte.")
    print("     -- Samtools Path: " + read_config("Bismark", "samtools_path"))
    print("     -- Bedtools Path: " + read_config("Bismark", "bedtools_path"))
    print("     -- Intermediate for MethExtractor: " + read_config("Bismark", "intermediate_files"))
    print("- Methylation extraction parameters( Only for quick run)")
    print("     -- Minimum read coverage: " + read_config("Methimpute", "mincov"))

def update_config(config_file: str, software_paths: dict):
    """
    更新配置文件中的软件路径。
    :param config_file: 配置文件路径
    :param software_paths: 软件路径的字典
    """
    config = configparser.ConfigParser()
    config.read(config_file)

    for software, path in software_paths.items():
        section, option = software.split(".", 1) if "." in software else ("TOOLS", software)
        if section not in config:
            config.add_section(section)
        if path:
            config[section][option] = path
            
def check_software_availability(software_name, path):
    try:
        if path is None:
            subprocess.check_output([software_name, '--version'])
        else:
            subprocess.check_output([path, '--version'])
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False
def check_path_availability(path):
    try:
        subprocess.check_output([path])
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False

def main():

    # 配置文件路径和日志文件路径
    config_file = "./my.conf"
    log_file = "config_check.log"
    if not os.path.exists(config_file):
        print(f"配置文件 {config_file} 不存在！")
        #logg.error(f"配置文件 {config_file} 不存在！")
        return
    # 检查执行上游比对所需要的命令行软件在系统中是否安装以及版本问题
    software_list = {
        "fastqc": None,
        "trimomatic": None,
        "bowtie2": None,
        "bismark": None,
        "samtools": None,
        "bedtools": None,
    }
    # 从配置文件读取参数
    config = configparser.ConfigParser()
    config.read(config_file)

    #check output directory
    output_dir = config.get('OUTPUT', 'output_dir', fallback=None)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"输出路径 {output_dir} 不存在，已创建！")
        except Exception as e:
            print(f"创建输出路径 {output_dir} 时出错: {e}")
            #logg.error(f"创建输出路径 {path} 时出错: {e}")
    
    # 检查软件路径并更新配置
    valid_paths = {}
    for software, path in software_list.items():
        software_name = software.split(".")[-1]
        print(software_name)
        path = config.get('TOOLS', software_name, fallback=None)
        valid_paths[software] = check_software_availability(software_name, path)

    
    update_config(config_file, valid_paths)

    #logg.info(f"配置检查和更新完成，日志已写入 {log_file}。")
    
    return

if __name__ == "__main__":
    main()

    