import os
import argparse
from pathlib import Path
import yaml
import re
import shutil

def load_yaml(yaml_path):
    """读取 YAML 文件并返回数据"""
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def batch_samples(samples, batch_size=None, regex=None):
        if batch_size:
            for i in range(0, len(samples), batch_size):
                yield samples[i:i + batch_size]
        elif regex:
            pattern = re.compile(regex)
            batches = {}
            for sample in samples:
                match = pattern.match(sample)
                if match:
                    key = match.group(0)
                    if key not in batches:
                        batches[key] = []
                    batches[key].append(sample)
            for batch in batches.values():
                yield batch
        else:
            yield samples
            
def copy_and_modify_snakefile(snake_template, out, bath_id, batch, input_dir, output_dir, yaml_file ):
    if bath_id == 1:
        filename = "Snakefile"
    else:
        filename = f"Snakefile_b{bath_id}"
    batch_snakefile = os.path.join(out,filename)
    shutil.copy(snake_template, batch_snakefile)
    with open(batch_snakefile, 'r') as file:
        content = file.read()
    content = content.replace('configfile: "./config.yaml"', f'configfile: "{yaml_file}"')
    content = content.replace('inputdir = config["inputfile"]["inputdir"]', f'inputdir = "{input_dir}"')
    content = content.replace('output_dir = config["outputdir"]', f'output_dir = "{output_dir}"') #TO DO: check if this is correct
    samples = ', '.join([f'"{sample}"' for sample in batch])
    content = content.replace('SAMPLES = glob_wildcards(f"{inputdir}/{{sample}}.sra").sample', f'SAMPLES = {batch}')
    with open(batch_snakefile, 'w') as file:
        file.write(content)
    print(f"Snakefile for batch {bath_id} created at {batch_snakefile}")
   
  
def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch file processing for Snakefile generation")
    parser.add_argument('-yaml', required=True, help="Path to the YAML file containing inputdir")
    parser.add_argument('-out', default='./' ,required=False, help="Output directory for Snakefiles")
    parser.add_argument('-suffix',default='.sra', required=False, help="Suffix of the methylation files to be batched")
    parser.add_argument('-batch', type=str, help="Batch size when using 'number' batching method")
    parser.add_argument('-regex', help="Regex pattern for file name batching when using 'filename' batching method")
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    snake_template =  "./Snakefile" 
    config = load_yaml(args.yaml)
    input_dir = config["input"]["inputdir"] #raw file path in config.yaml
    output_dir = config["output"]["outputdir"] #output file path in config.yaml
    out = args.out if args.out else os.path.dirname(args.yaml) #out snakefile path
    if not os.path.exists(out):
        os.makedirs(out)
    samples = [os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith(args.suffix)]
    #samples = [f for f in os.listdir(input_dir) if f.endswith(args.suffix)]
    batch_size = int(args.batch) if args.batch.isdigit() else None
    regex = args.batch if not args.batch.isdigit() else None

    for bath_id, batch in enumerate(batch_samples(samples, batch_size, regex)):
        copy_and_modify_snakefile(snake_template, out, bath_id+1, batch, input_dir, output_dir, os.path.abspath(args.yaml))
    
    