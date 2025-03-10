# scMethQ -- single-cell DNA Methylation Tool

scMethQ v0.1.0

## Description
This study develops the Python-based tool scMethtools for single-cell DNA methylation data analysis. This tool is a one-stop analytical framework designed to handle the complex characteristics of single-cell DNA methylation data, providing data processing, computation, and visualization functionalities. The overall architecture of scMethtools is inspired by the Scanpy package, which was designed for scRNA-seq data analysis (Wolf et al., 2018). The core data structure and operations rely on tools such as NumPy (Van Der Walt et al., 2011), Scipy (Virtanen et al., 2020), and Pandas. The computational methods are based on Scikit-learn (Pedregosa et al., 2011) and Statsmodels, while the visualization methods depend on Matplotlib (Hunter, 2007) and Seaborn (Waskom, 2021) etc.

## Dependencies

```
Pytorch >= 1.5.0

numpy >= 1.18.2

scipy >= 1.4.1

pandas >= 1.0.3

sklearn >= 0.22.1

seaborn >= 0.10.0
```

## Installation
Install the develop version from GitHub source code with

```
git clone https://github.com/Wentting/scMethTools.git 
```

And run

``` 
pip install .
```

Uninstall using

```
pip uninstall scmethq
```

## Usage

See Documentation at  https://wentting.github.io/scMethQ

## Content
- `scMethtools/` contains the python code for the package
- `data`

## Cite