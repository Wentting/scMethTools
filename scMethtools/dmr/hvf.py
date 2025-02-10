#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年09月15日

"""
import numpy as np
import pandas as pd

def calculate_hvf_svr(self, var_dim, mc_type, obs_dim='cell', n_top_feature=5000, da_suffix='frac', plot=True):
    from sklearn.svm import SVR
    import plotly.graph_objects as go

    feature_mc_frac_mean = self[f'{var_dim}_da_{da_suffix}'].sel(mc_type=mc_type).mean(
        dim=obs_dim).to_pandas()
    feature_std = self[f'{var_dim}_da_{da_suffix}'].sel(mc_type=mc_type).std(
        dim=obs_dim).to_pandas()
    feature_cov_mean = self[f'{var_dim}_da'].sel(
        mc_type=mc_type, count_type='cov').mean(dim=obs_dim).to_pandas()

    # remove bad features
    judge = (feature_mc_frac_mean > 0) & (feature_std > 0) & (feature_cov_mean > 0)
    if n_top_feature >= judge.size:
        n_top_feature = judge.size
        print('n_top_feature is than total number of features, will use all features')
    feature_mc_frac_mean = feature_mc_frac_mean[judge]
    feature_var = feature_std[judge] ** 2  # to be consistent with previous bin-based method, use var here
    feature_cov_mean = feature_cov_mean[judge]

    # prepare data
    dispersion = feature_var / feature_mc_frac_mean
    log2_disp = np.log2(dispersion)
    log2_mc_frac_mean = np.log2(feature_mc_frac_mean)
    log2_cov_mean = np.log2(feature_cov_mean)
    x = np.vstack((log2_mc_frac_mean, log2_cov_mean)).T

    # non-linear regression predicting dispersion using mc_frac_mean and cov_mean.
    svr_gamma = 1000 / judge.sum()
    print(f'Fitting SVR with gamma {svr_gamma:.4f}, '
          f'predicting feature dispersion using mc_frac_mean and cov_mean.')
    clf = SVR(gamma=svr_gamma)
    clf.fit(x, log2_disp)
    # Score is the relative position with respect of the fitted curve
    score = log2_disp - clf.predict(x)
    selected_feature_index = score.sort_values()[-n_top_feature:].index
    # make results table
    selected_feature_index = score.sort_values()[-n_top_feature:].index
    hvf_df = pd.DataFrame(
        {
            'mean': feature_mc_frac_mean.reindex(judge.index).fillna(0),
            'dispersion': dispersion.reindex(judge.index).fillna(0),
            'cov': feature_cov_mean.reindex(judge.index).fillna(0),
            'score': score.reindex(judge.index).fillna(-100)
        }
    )
    hvf_df['feature_select'] = hvf_df.index.isin(selected_feature_index)

    print(f'Total Feature Number:     {judge.size}')
    print(f'Highly Variable Feature:  {selected_feature_index.size} '
          f'({(selected_feature_index.size / judge.size * 100):.1f}%)')

    if plot:
        if hvf_df.shape[0] > 5000:
            plot_data = hvf_df.sample(5000)
        else:
            plot_data = hvf_df
        fig = go.Figure(data=[
            go.Scatter3d(
                x=plot_data['mean'],
                y=plot_data['cov'],
                z=np.log2(plot_data['dispersion']),
                mode='markers',
                hoverinfo='none',
                marker=dict(
                    size=2,
                    color=plot_data['feature_select'].map({
                        True: 'red',
                        False: 'gray'
                    }).tolist(),  # set color to an array/list of desired values
                    opacity=0.8), )
        ])
        fig.update_layout(scene=dict(xaxis_title='mC Frac. Mean',
                                     yaxis_title='Coverage Mean',
                                     zaxis_title='log2(Dispersion)'),
                          margin=dict(r=0, b=0, l=0, t=0))
        fig.show()

    for name, column in hvf_df.items():
        self.coords[f'{var_dim}_{mc_type}_{name}'] = column
    return hvf_df