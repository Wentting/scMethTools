#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年08月22日

"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot_summary(summary_file):
    stat_df = pd.read_csv(summary_file, sep='\t')
    plt.title("Mean Methylation level range")
    plt.xlabel('mc level')
    plt.ylabel('cell count')
    plt.hist(stat_df['CmMeanLevel_CG'], range=(0, 1))
    plt.axvline(x=0.2, color='r', linestyle='--')