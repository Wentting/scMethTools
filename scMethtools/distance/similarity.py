#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年09月22日

"""

import pandas as pd
import numpy as np
from functools import partial
from multiprocessing import Pool


def mat_F(df):
    chr_P = df.iloc[:,0].astype(str) + '_' + df.iloc[:,1].astype(str)
    df.insert(0, 'chr_P', chr_P)
    df = df.iloc[:,[0,3]]
    return df

def cor_cp(i,j,list_df):
    df1 = mat_F(list_df[j])
    df2 = mat_F(list_df[i])
    df = pd.merge(df1, df2, on='chr_P', how='inner')
    pv = np.corrcoef(df.iloc[:,1], df.iloc[:,2])[0][1]
    return pv

def Get_Pearson(j, data):
    cor_partial = partial(cor_cp, j=j, list_df=data)
    with Pool() as p:
        res_cor = p.map(cor_partial, range(len(data)))
    res_cor = pd.DataFrame(res_cor)
    return res_cor

def cor_cos(i,j,list_df):
    df1 = mat_F(list_df[j])
    df2 = mat_F(list_df[i])
    df = pd.merge(df1, df2, on='chr_P', how='inner')
    cos = sum(df.iloc[:,1]*df.iloc[:,2])/(np.sqrt(sum(df.iloc[:,1]**2))*np.sqrt(sum(df.iloc[:,2]**2)))
    return cos

def Get_Cosine(j, data):
    cor_partial = partial(cor_cos, j=j, list_df=data)
    with Pool() as p:
        res_cor = p.map(cor_partial, range(len(data)))
    res_cor = pd.DataFrame(res_cor)
    return res_cor

def cor_ham(i,j,list_df):
    df1 = mat_F(list_df[j])
    df2 = mat_F(list_df[i])
    df = pd.merge(df1, df2, on='chr_P', how='inner')
    df['cot'] = np.where(df.iloc[:,1] == df.iloc[:,2], 1, 0)
    dua = sum(df['cot'])/len(df['cot'])
    return dua

def Get_Hamming(j, data):
    cor_partial = partial(cor_ham, j=j, list_df=data)
    with Pool() as p:
        res_cor = p.map(cor_partial, range(len(data)))
    res_cor = pd.DataFrame(res_cor)
    return res_cor
