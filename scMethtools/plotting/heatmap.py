import scanpy as sc
from scanpy import _utils
from scanpy.plotting._utils import check_colornorm
from scanpy.plotting._anndata import _prepare_dataframe
from scanpy.plotting import _utils 
import scMethtools.logging as logg
from .basic import get_colors_from_adata, set_colors

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype, is_numeric_dtype

import collections.abc as cabc
from collections.abc import Collection, Iterable, Sequence

from anndata import AnnData

from matplotlib import gridspec, pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from typing import TYPE_CHECKING, Literal, Union

    
def dmr_heatmap(
    adata: AnnData,
    dmr_key: str = "rank_genes_groups",
    groupby: str = None, 
    annotation: str = None,
    use_raw: bool = None,
    dendrogram: bool= True,
    swap_axes: bool = True,
    **kwds)-> dict[str, Axes]:
    """\  
    Generate a heatmap of the differentially methylated regions/genes/features.
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    dmr_key : str, optional (default: "rank_genes_groups")
        Key to access the differentially methylated regions (DMR) in `adata.uns`.
    groupby : str, optional
        The key of the observation grouping to consider.
    annotation : str, optional
        Key for annotations.
    use_raw : bool, optional
        Whether to use `raw` attribute of `adata` if present.
    dendrogram : bool, optional (default: True)
        If True, a dendrogram based on the hierarchical clustering is added.
    swap_axes : bool, optional (default: True)
        If True, swap the axes of the heatmap.
    **kwds : keyword arguments
        Additional arguments passed to the heatmap function.
    Returns
    -------
    dict[str, Axes]
        A dictionary of matplotlib Axes objects.
    Example
    -------
    >>> import scmethtools as scm
    >>> scm.pl.dmr_heatmap(adata, dmr_key='treatment',swap_axes=True,annotation=['Cell_type','Treatment','Label'],groupby='leiden',cmap="Reds",dendrogram=True)
    
    """
    var_names = adata.uns[dmr_key]['names']['0'].tolist()
    heatmap(adata, var_names, groupby, annotation, dendrogram=dendrogram, use_raw=use_raw,swap_axes=swap_axes, **kwds)

def heatmap(
    adata: AnnData,
    var_names: Union[str, Sequence[str]],
    groupby: str,
    annotation: Union[str, Sequence[str]] = None,
    *,
    use_raw: bool = None,
    log: bool = False,
    num_categories: int = 7,
    dendrogram: bool = False,
    gene_symbols: str = None,
    var_group_positions: Sequence[tuple[int, int]] = None,
    var_group_labels: Sequence[str] = None,
    var_group_rotation: float = None,
    layer: str = None,
    standard_scale: Literal["var", "obs"] = None,
    swap_axes: bool = False,
    show_gene_labels: bool = None,
    show: bool = None,
    save: str = None,
    figsize: tuple[float, float] = None,
    vmin: float = None,
    vmax: float = None,
    vcenter: float = None,
    norm: Normalize = None,
    **kwds,
) -> dict[str, Axes]:
    """\
    Heatmap of the expression values of genes.

    If `groupby` is given, the heatmap is ordered by the respective group. For
    example, a list of marker genes can be plotted, ordered by clustering. If
    the `groupby` observation annotation is not categorical the observation
    annotation is turned into a categorical by binning the data into the number
    specified in `num_categories`.

    Parameters
    ----------
    {common_plot_args}
    standard_scale
        Whether or not to standardize that dimension between 0 and 1, meaning for each variable or observation,
        subtract the minimum and divide each by its maximum.
    swap_axes
         By default, the x axis contains `var_names` (e.g. genes) and the y axis the `groupby`
         categories (if any). By setting `swap_axes` then x are the `groupby` categories and y the `var_names`.
    show_gene_labels
         By default gene labels are shown when there are 50 or less genes. Otherwise the labels are removed.
    {show_save_ax}
    {vminmax}
    **kwds
        Are passed to :func:`matplotlib.pyplot.imshow`.

    Returns
    -------
    Dict of :class:`~matplotlib.axes.Axes`

    Examples
    -------
    .. plot::
        :context: close-figs

        import scanpy as sc
        adata = sc.datasets.pbmc68k_reduced()
        markers = ['C1QA', 'PSAP', 'CD79A', 'CD79B', 'CST3', 'LYZ']
        sc.pl.heatmap(adata, markers, groupby='bulk_labels', swap_axes=True)

    .. currentmodule:: scanpy

    See also
    --------
    pl.rank_genes_groups_heatmap
    tl.rank_genes_groups
    """
    var_names, var_group_labels, var_group_positions = _check_var_names_type(
        var_names, var_group_labels, var_group_positions
    )

    categories, obs_tidy = _prepare_dataframe(
        adata,
        var_names,
        groupby,
        use_raw=use_raw,
        log=log,
        num_categories=num_categories,
        gene_symbols=gene_symbols,
        layer=layer,
    )
    anno_df = pd.DataFrame()  # 确保它始终有定义

    if annotation is not None:
        annotation_new = annotation if isinstance(annotation, list) else [annotation]
    else:
        annotation_new=[]
        #annotation_new = ['Cell_type','Treatment','Label'] # TO DO: 这里需要改接收参数
    anno_df = adata.obs[annotation_new]
    anno_df.index=obs_tidy.index

    # check if var_group_labels are a subset of categories:
    if var_group_labels is not None:
        if set(var_group_labels).issubset(categories):
            var_groups_subset_of_groupby = True
        else:
            var_groups_subset_of_groupby = False

    if standard_scale == "obs":
        obs_tidy = obs_tidy.sub(obs_tidy.min(1), axis=0)
        obs_tidy = obs_tidy.div(obs_tidy.max(1), axis=0).fillna(0)
    elif standard_scale == "var":
        obs_tidy -= obs_tidy.min(0)
        obs_tidy = (obs_tidy / obs_tidy.max(0)).fillna(0)
    elif standard_scale is None:
        pass
    else:
        logg.warning("Unknown type for standard_scale, ignored")

    if groupby is None or len(categories) <= 1:
        categorical = False
        # dendrogram can only be computed  between groupby categories
        dendrogram = False
    else:
        categorical = True
        # get categories colors
        if isinstance(groupby, str) and isinstance(
            adata.obs[groupby].dtype, CategoricalDtype
        ):
            # saved category colors only work when groupby is valid adata.obs
            # categorical column. When groupby is a numerical column
            # or when groupby is a list of columns the colors are assigned on the fly,
            # which may create inconsistencies in multiple runs that require sorting
            # of the categories (eg. when dendrogram is plotted).
            if groupby + "_colors" not in adata.uns:
                # if colors are not found, assign a new palette
                # and save it using the same code for embeddings
                #from scanpy._tools.scatterplots import _get_palette

                _get_palette(adata, groupby)
            groupby_colors = adata.uns[groupby + "_colors"]
        else:
            # this case happen when adata.obs[groupby] is numeric
            # the values are converted into a category on the fly
            groupby_colors = None

    if dendrogram:
        dendro_data = _reorder_categories_after_dendrogram(
            adata,
            groupby,
            dendrogram,
            var_names=var_names,
            var_group_labels=var_group_labels,
            var_group_positions=var_group_positions,
            categories=categories,
        )

        var_group_labels = dendro_data["var_group_labels"]
        var_group_positions = dendro_data["var_group_positions"]

        # reorder obs_tidy
        if dendro_data["var_names_idx_ordered"] is not None:
            obs_tidy = obs_tidy.iloc[:, dendro_data["var_names_idx_ordered"]]
            var_names = [var_names[x] for x in dendro_data["var_names_idx_ordered"]]

        obs_tidy.index = obs_tidy.index.reorder_categories(
            [categories[x] for x in dendro_data["categories_idx_ordered"]],
            ordered=True,
        )

        # reorder groupby colors
        if groupby_colors is not None:
            groupby_colors = [
                groupby_colors[x] for x in dendro_data["categories_idx_ordered"]
            ]

    if show_gene_labels is None:
        if len(var_names) <= 50:
            show_gene_labels = True
        else:
            show_gene_labels = False
            logg.warning(
                "Gene labels are not shown when more than 50 genes are visualized. "
                "To show gene labels set `show_gene_labels=True`"
            )
    if categorical:
        obs_tidy = obs_tidy.sort_index()

        anno_df = anno_df.sort_index()

    colorbar_width = 0.2
    norm = check_colornorm(vmin, vmax, vcenter, norm)

    if not swap_axes:

        #labels are on the left or bottom

        dendro_width = 1 if dendrogram else 0
        groupby_width = 0.2 if categorical else 0
         
        if figsize is None:
            height = 6
            if show_gene_labels:
                heatmap_width = len(var_names) * 0.3
            else:
                heatmap_width = 8
            width = heatmap_width + dendro_width + groupby_width
        else:
            width, height = figsize
            heatmap_width = width - (dendro_width + groupby_width)

        if var_group_positions is not None and len(var_group_positions) > 0:
            # add some space in case 'brackets' want to be plotted on top of the image
            height_ratios = [0.15, height]
        else:
            height_ratios = [0, height]
        
                # 确定 anno_df 中分组列的数量
        num_groupby_columns = len(anno_df.columns) if not anno_df.empty else 1
        
        # 生成相应数量的 groupby_width
        groupby_widths = [groupby_width] * num_groupby_columns
        
        # 构造 width_ratios 列表
        width_ratios = [dendro_width] + groupby_widths + [groupby_width, heatmap_width, colorbar_width]
        
        fig = plt.figure(figsize=(width, height))

        axs = gridspec.GridSpec(
            nrows=2,
            ncols=4+num_groupby_columns,
            width_ratios=width_ratios,
            wspace=0.25 / width,
            hspace=0.15 / height,
            height_ratios=height_ratios,
        )

        heatmap_ax = fig.add_subplot(axs[1, num_groupby_columns+2])
        kwds.setdefault("interpolation", "nearest")
        im = heatmap_ax.imshow(obs_tidy.values, aspect="auto", norm=norm, **kwds)

        heatmap_ax.set_ylim(obs_tidy.shape[0] - 0.5, -0.5)
        heatmap_ax.set_xlim(-0.5, obs_tidy.shape[1] - 0.5)
        heatmap_ax.tick_params(axis="y", left=False, labelleft=False)
        heatmap_ax.set_ylabel("")
        heatmap_ax.grid(visible=False)

        if show_gene_labels:
            heatmap_ax.tick_params(axis="x", labelsize="small")
            heatmap_ax.set_xticks(np.arange(len(var_names)))
            heatmap_ax.set_xticklabels(var_names, rotation=90)
        else:
            heatmap_ax.tick_params(axis="x", labelbottom=False, bottom=False)
        # plot colorbar
        _plot_colorbar(im, fig, axs[1, 3+num_groupby_columns])
        
        if categorical:
            # 定义图例的初始位置
            # 计算 figure 高度
            fig_height = fig.get_size_inches()[1]  # 获取当前 figure 的高度
            max_legends = int(fig_height / 1.5)  # 根据高度估算最多能容纳多少个图例1.5 是经验参数 TODO：这里应该修改为动态计算
    
            # 初始图例位置（右上角）
            x_position = 1.01
            y_start = 0.9  # 统一的 y 轴起点，保证每列对齐
            y_position = y_start  # 当前 y 位置
            legends_in_current_col = 0  # 记录当前列已放置的图例数

              #----- group的annotation
            groupby_ax = fig.add_subplot(axs[1, 1])
            (
                label2code,
                ticks,
                labels,
                groupby_cmap,
                norm,
            ) = _plot_categories_as_colorblocks(
                groupby_ax, obs_tidy, colors=get_colors_from_adata(adata, groupby),  orientation="left", label=groupby
            )
            legend_elements = [
                Patch(facecolor=groupby_cmap(color), edgecolor='none', label=label)
                for label, color in label2code.items()
            ]
            # 统一放置 legend
            legend = fig.legend(
                handles=legend_elements,
                title=groupby,  # 设置标题
                loc='upper left',
                bbox_to_anchor=(x_position, y_position),  # 右侧纵向排列
                frameon=True,
                # bbox_transform=fig.transFigure,  # 确保 y_position 参考的是 fig，而不是 ax
                # borderpad=0.5,
                title_fontsize='small',
                fontsize='small',
                markerscale=0.6
            )
            # 获取当前图例的高度
            # legend_height = legend.get_window_extent().height / (fig.dpi * 10)  # 转换为英寸
            # 正确计算图例高度（使用figure坐标）
            fig.canvas.draw()  # 确保渲染器更新
            renderer = fig.canvas.get_renderer()
            legend_bbox = legend.get_window_extent(renderer)
            legend_bbox_fig = legend_bbox.transformed(fig.transFigure.inverted())
            legend_height = legend_bbox_fig.height
            #print("legend_height", legend_height)
            # 更新 y_position，确保下一个图例不会重叠
            y_position -= legend_height + 0.05  # 图例的高度加上一些间隔
            #print("y_position", y_position)
            legends_in_current_col += 1

            # 如果当前列的图例数量超过 `max_legends_per_col`，换列
            if legends_in_current_col >= max_legends:
                x_position += 0.1  # 水平间距，数值越大间距越大
                y_position = y_start  # 重新从顶部开始
                legends_in_current_col = 0  # 重置当前列计数
            
            # plot labels annotation bar
            if annotation_new:
                for idx,ann in enumerate(annotation_new):
                    # 获取当前注释的颜色块信息
                    #print(idx,ann)
                    anno_ax = fig.add_subplot(axs[1, idx + 2])
                    anno_df.index = anno_df[ann]
                    (
                    label2code,
                    ticks,
                    labels,
                    groupby_cmap,
                    norm,
                    ) =_plot_categories_as_colorblocks(
                        anno_ax, anno_df, colors=get_colors_from_adata(adata, ann),  orientation="left", label=ann
                    ) #should defined color by labels
                     #创建每个 ann 的图例元素列表
                    legend_elements = [
                        Patch(facecolor=groupby_cmap(color), edgecolor='none', label=label)
                        for label, color in label2code.items()
                    ]
                    
                    # 统一放置 legend
                    legend = fig.legend(
                        handles=legend_elements,
                        title=groupby,  # 设置标题
                        loc='upper left',
                        bbox_to_anchor=(x_position, y_position),  # 右侧纵向排列
                        frameon=True,
                        # bbox_transform=fig.transFigure,  # 确保 y_position 参考的是 fig，而不是 ax
                        # borderpad=0.5,
                        title_fontsize='small',
                        fontsize='small',
                        markerscale=0.6
                    )
                    # 获取当前图例的高度
                    # legend_height = legend.get_window_extent().height / (fig.dpi * 10)  # 转换为英寸
                    # 正确计算图例高度（使用figure坐标）
                    fig.canvas.draw()  # 确保渲染器更新
                    renderer = fig.canvas.get_renderer()
                    legend_bbox = legend.get_window_extent(renderer)
                    legend_bbox_fig = legend_bbox.transformed(fig.transFigure.inverted())
                    legend_height = legend_bbox_fig.height
                    #print("legend_height", legend_height)
                    # 更新 y_position，确保下一个图例不会重叠
                    y_position -= legend_height + 0.05  # 图例的高度加上一些间隔
                    #print("y_position", y_position)
                    legends_in_current_col += 1
        
                    # 如果当前列的图例数量超过 `max_legends_per_col`，换列
                    if legends_in_current_col >= max_legends:
                        x_position += 0.1  # 水平间距，数值越大间距越大
                        y_position = y_start  # 重新从顶部开始
                        legends_in_current_col = 0  # 重置当前列计数

            # add lines to main heatmap
            line_positions = (
                np.cumsum(obs_tidy.index.value_counts(sort=False))[:-1] - 0.5
            )
            heatmap_ax.hlines(
                line_positions,
                -0.5,
                len(var_names) - 0.5,
                lw=1,
                color="grey",
                zorder=10,
                clip_on=False,
            )

        if dendrogram:
            dendro_ax = fig.add_subplot(axs[1, 0], sharey=heatmap_ax)
            _plot_dendrogram(
                dendro_ax, adata, groupby, ticks=ticks, dendrogram_key=dendrogram,orientation="left"
            )

        # plot group legends on top of heatmap_ax (if given)
        if var_group_positions is not None and len(var_group_positions) > 0:
            gene_groups_ax = fig.add_subplot(axs[0, num_groupby_columns+1], sharex=heatmap_ax)
            _plot_gene_groups_brackets(
                gene_groups_ax,
                group_positions=var_group_positions,
                group_labels=var_group_labels,
                rotation=var_group_rotation,
                left_adjustment=-0.3,
                right_adjustment=0.3,
            )

    # swap axes case
    else:
        # define a layout of 3 rows x 3 columns
        # The first row is for the dendrogram (if not dendrogram height is zero)
        # second row is for main content. This col is divided into three axes:
        #   first ax is for the heatmap
        #   second ax is for 'brackets' if any (othwerise width is zero)
        #   third ax is for colorbar

        dendro_height = 0.8 if dendrogram else 0
        groupby_height = 0.18 if categorical else 0
        if figsize is None:
            if show_gene_labels:
                heatmap_height = len(var_names) * 0.18
            else:
                heatmap_height = 4
            width = 10
            height = heatmap_height + dendro_height + groupby_height
        else:
            width, height = figsize
            heatmap_height = height - (dendro_height + groupby_height)


        if var_group_positions is not None and len(var_group_positions) > 0:
            # add some space in case 'brackets' want to be plotted on top of the image
            width_ratios = [width, 0.14, colorbar_width]
        else:
            width_ratios = [width, 0, colorbar_width]

        # num_groupby_columns = len(anno_df.columns)
        num_groupby_rows = len(anno_df.columns) if not anno_df.empty else 1

        # 生成相应数量的 groupby_width
        groupby_heights = [groupby_height] * num_groupby_rows
        # 构造 width_ratios 列表
        height_ratios = [dendro_height] + groupby_heights+ [groupby_height , heatmap_height]
        fig = plt.figure(figsize=(width, height))
        axs = gridspec.GridSpec(
            nrows=3+num_groupby_rows,
            ncols=3,
            wspace=0.25 / width,
            hspace=0.4 / height,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )

        # plot heatmap
        heatmap_ax = fig.add_subplot(axs[num_groupby_rows+2, 0])

        kwds.setdefault("interpolation", "nearest")
        im = heatmap_ax.imshow(obs_tidy.T.values, aspect="auto", norm=norm, **kwds)
        heatmap_ax.set_xlim(0 - 0.5, obs_tidy.shape[0] - 0.5)
        heatmap_ax.set_ylim(obs_tidy.shape[1] - 0.5, -0.5)
        heatmap_ax.tick_params(axis="x", bottom=False, labelbottom=False)
        heatmap_ax.set_xlabel("")
        heatmap_ax.grid(visible=False)
        if show_gene_labels:
            heatmap_ax.tick_params(axis="y", labelsize="small", length=1)
            heatmap_ax.set_yticks(np.arange(len(var_names)))
            heatmap_ax.set_yticklabels(var_names, rotation=0)
        else:
            heatmap_ax.tick_params(axis="y", labelleft=False, left=False)

        # 计算 figure 高度
        fig_height = fig.get_size_inches()[1]  # 获取当前 figure 的高度
        max_legends = int(fig_height / 1.5)  # 根据高度估算最多能容纳多少个图例1.5 是经验参数 TODO：这里应该修改为动态计算
        
        # 初始图例位置（右上角）
        x_position = 0.9
        y_start = 0.9  # 统一的 y 轴起点，保证每列对齐
        y_position = y_start  # 当前 y 位置
        legends_in_current_col = 0 # 记录当前列已放置的图例数

        if categorical:
            #----- group的annotation
            groupby_ax = fig.add_subplot(axs[1, 0])
            (
                label2code,
                ticks,
                labels,
                groupby_cmap,
                norm,
            ) = _plot_categories_as_colorblocks(
                groupby_ax, obs_tidy, colors=get_colors_from_adata(adata, groupby_colors), annotation=True, orientation="bottom",label=groupby
            )

            legend_elements = [
                        Patch(facecolor=groupby_cmap(color), edgecolor='none', label=label)
                        for label, color in label2code.items()
                    ]                  
            # 统一放置 legend
            legend = fig.legend(
                handles=legend_elements,
                title=groupby,  # 设置标题
                loc='upper left',
                bbox_to_anchor=(x_position, y_position),  # 右侧纵向排列
                frameon=True,
                #bbox_transform=fig.transFigure,  # 确保 y_position 参考的是 fig，而不是 ax
                #borderpad=0.5,
                title_fontsize='small',
                fontsize='small',
                markerscale=0.6
            )
            # 获取当前图例的高度
            #legend_height = legend.get_window_extent().height / (fig.dpi * 10)  # 转换为英寸
            # 正确计算图例高度（使用figure坐标）
            fig.canvas.draw()  # 确保渲染器更新
            renderer = fig.canvas.get_renderer()
            legend_bbox = legend.get_window_extent(renderer)
            legend_bbox_fig = legend_bbox.transformed(fig.transFigure.inverted())
            legend_height = legend_bbox_fig.height
            #print("legend_height",legend_height)
            # 更新 y_position，确保下一个图例不会重叠
            y_position -= legend_height + 0.05 # 图例的高度加上一些间隔
            #print("y_position",y_position)
            legends_in_current_col += 1
        
            # 如果当前列的图例数量超过 `max_legends_per_col`，换列
            if legends_in_current_col >= max_legends:
                x_position += 0.1  # 水平间距，数值越大间距越大
                y_position = y_start  # 重新从顶部开始
                legends_in_current_col = 0  # 重置当前列计数
            #print(x_position,y_position,legends_in_current_col)

            
            
            if annotation_new:
                 # plot labels annotation bar
                for idx,ann in enumerate(annotation_new):
                    #print(idx,ann)
                    anno_ax = fig.add_subplot(axs[idx+2, 0]) 
                    anno_df.index = anno_df[ann]
                    (
                        label2code,
                        ticks,
                        labels,
                        groupby_cmap,
                        norm,
                    ) = _plot_categories_as_colorblocks(
                        anno_ax, anno_df, colors=get_colors_from_adata(adata, ann),annotation=True, orientation="bottom",label=ann
                    )
                    legend_elements = [
                        Patch(facecolor=groupby_cmap(color), edgecolor='none', label=label)
                        for label, color in label2code.items()
                    ]                  
                    # 统一放置 legend
                    legend = fig.legend(
                        handles=legend_elements,
                        title=ann,  # 设置标题
                        loc='upper left',
                        bbox_to_anchor=(x_position, y_position),  # 右侧纵向排列
                        frameon=True,
                        #bbox_transform=fig.transFigure,  # 确保 y_position 参考的是 fig，而不是 ax
                        #borderpad=0.5,
                        title_fontsize='small',
                        fontsize='small',
                        markerscale=0.6
                    )
                    # 获取当前图例的高度
                    #legend_height = legend.get_window_extent().height / (fig.dpi * 10)  # 转换为英寸
                    # 正确计算图例高度（使用figure坐标）
                    fig.canvas.draw()  # 确保渲染器更新
                    renderer = fig.canvas.get_renderer()
                    legend_bbox = legend.get_window_extent(renderer)
                    legend_bbox_fig = legend_bbox.transformed(fig.transFigure.inverted())
                    legend_height = legend_bbox_fig.height
                    # 更新 y_position，确保下一个图例不会重叠
                    y_position -= legend_height + 0.05 # 图例的高度加上一些间隔
                    legends_in_current_col += 1
                
                    # 如果当前列的图例数量超过 `max_legends_per_col`，换列
                    if legends_in_current_col >= max_legends:
                        x_position += 0.1  # 水平间距，数值越大间距越大
                        y_position = y_start  # 重新从顶部开始
                        legends_in_current_col = 0  # 重置当前列计数
            # 适当调整 fig 布局，避免图例重叠
            fig.subplots_adjust(right=0.8)  # 调整布局，让图例有足够空间
            # add lines to main heatmap
            line_positions = (
                np.cumsum(obs_tidy.index.value_counts(sort=False))[:-1] - 0.5
            )
            heatmap_ax.vlines(
                line_positions,
                -0.5,
                len(var_names) - 0.5,
                lw=1,
                color="black",
                zorder=10,
                clip_on=False,
            )
            

        if dendrogram:
            dendro_ax = fig.add_subplot(axs[0, 0], sharex=heatmap_ax) #确定位置不用管
            _plot_dendrogram(
                dendro_ax,
                adata,
                groupby,
                dendrogram_key=dendrogram,
                ticks=ticks,
                orientation="top",
            )

        # plot group legends next to the heatmap_ax (if given)
        if var_group_positions is not None and len(var_group_positions) > 0:
            gene_groups_ax = fig.add_subplot(axs[num_groupby_rows+2, 1])
            arr = []
            for idx, (label, pos) in enumerate(
                zip(var_group_labels, var_group_positions)
            ):
                if var_groups_subset_of_groupby:
                    label_code = label2code[label]
                else:
                    label_code = idx
                arr += [label_code] * (pos[1] + 1 - pos[0])
            gene_groups_ax.imshow(
                np.array([arr]).T, aspect="auto", cmap=groupby_cmap, norm=norm
            )
            gene_groups_ax.axis("off")

        # plot colorbar
        _plot_colorbar(im, fig, axs[num_groupby_rows+2, 2])

    return_ax_dict = {"heatmap_ax": heatmap_ax}
    if categorical:
        return_ax_dict["groupby_ax"] = groupby_ax
    if dendrogram:
        return_ax_dict["dendrogram_ax"] = dendro_ax
    if var_group_positions is not None and len(var_group_positions) > 0:
        return_ax_dict["gene_groups_ax"] = gene_groups_ax

    _utils.savefig_or_show("heatmap", show=show, save=save)
    show = settings.autoshow if show is None else show
    if show:
        return None
    return return_ax_dict

def _prepare_dataframe(
    adata: AnnData,
    var_names: Union[str, Sequence[str]],
    groupby: str= None,
    *,
    use_raw: bool = False,
    log: bool = False,
    num_categories: int = 7,
    layer: str = None,
    gene_symbols: str = None,
) -> tuple[Sequence[str], pd.DataFrame]:
    """
    Given the anndata object, prepares a data frame in which the row index are the categories
    defined by group by and the columns correspond to var_names.

    Parameters
    ----------
    adata
        Annotated data matrix.
    var_names
        `var_names` should be a valid subset of  `adata.var_names`.
    groupby
        The key of the observation grouping to consider. It is expected that
        groupby is a categorical. If groupby is not a categorical observation,
        it would be subdivided into `num_categories`.
    use_raw
        Whether to use `raw` attribute of `adata`. Defaults to `True` if `.raw` is present.
    log
        Use the log of the values.
    layer
        AnnData layer to use. Takes precedence over `use_raw`
    num_categories
        Only used if groupby observation is not categorical. This value
        determines the number of groups into which the groupby observation
        should be subdivided.
    gene_symbols
        Key for field in .var that stores gene symbols.

    Returns
    -------
    Tuple of `pandas.DataFrame` and list of categories.
    """

    adata._sanitize()
    # use_raw = _utils._check_use_raw(adata, use_raw)
    use_raw = adata.raw is not None and (use_raw is True)
    if layer is not None:
        use_raw = False
    if isinstance(var_names, str):
        var_names = [var_names]

    groupby_index = None
    if groupby is not None:
        if isinstance(groupby, str):
            # if not a list, turn into a list
            groupby = [groupby]
        for group in groupby:
            if group not in list(adata.obs_keys()) + [adata.obs.index.name]:
                if adata.obs.index.name is not None:
                    msg = f' or index name "{adata.obs.index.name}"'
                else:
                    msg = ""
                raise ValueError(
                    "groupby has to be a valid observation. "
                    f"Given {group}, is not in observations: {adata.obs_keys()}" + msg
                )
            if group in adata.obs.keys() and group == adata.obs.index.name:
                raise ValueError(
                    f"Given group {group} is both and index and a column level, "
                    "which is ambiguous."
                )
            if group == adata.obs.index.name:
                groupby_index = group
    if groupby_index is not None:
        # obs_tidy contains adata.obs.index
        # and does not need to be given
        groupby = groupby.copy()  # copy to not modify user passed parameter
        groupby.remove(groupby_index)
    keys = list(groupby) + list(np.unique(var_names))
    obs_tidy = sc.get.obs_df(
        adata, keys=keys, layer=layer, use_raw=use_raw, gene_symbols=gene_symbols
    )
    assert np.all(np.array(keys) == np.array(obs_tidy.columns))

    if groupby_index is not None:
        # reset index to treat all columns the same way.
        obs_tidy.reset_index(inplace=True)
        groupby.append(groupby_index)

    if groupby is None:
        categorical = pd.Series(np.repeat("", len(obs_tidy))).astype("category")
    elif len(groupby) == 1 and is_numeric_dtype(obs_tidy[groupby[0]]):
        # if the groupby column is not categorical, turn it into one
        # by subdividing into  `num_categories` categories
        categorical = pd.cut(obs_tidy[groupby[0]], num_categories)
    elif len(groupby) == 1:
        categorical = obs_tidy[groupby[0]].astype("category")
        categorical.name = groupby[0]
    else:
        # join the groupby values  using "_" to make a new 'category'
        categorical = obs_tidy[groupby].apply("_".join, axis=1).astype("category")
        categorical.name = "_".join(groupby)

        # preserve category order
        from itertools import product

        order = {
            "_".join(k): idx
            for idx, k in enumerate(
                product(*(obs_tidy[g].cat.categories for g in groupby))
            )
        }
        categorical = categorical.cat.reorder_categories(
            sorted(categorical.cat.categories, key=lambda x: order[x])
        )
    obs_tidy = obs_tidy[var_names].set_index(categorical)
    categories = obs_tidy.index.categories

    if log:
        obs_tidy = np.log1p(obs_tidy)

    return categories, obs_tidy
def _reorder_categories_after_dendrogram(
    adata: AnnData,
    groupby,
    dendrogram,
    *,
    var_names=None,
    var_group_labels=None,
    var_group_positions=None,
    categories=None,
):
    """\
    Function used by plotting functions that need to reorder the the groupby
    observations based on the dendrogram results.

    The function checks if a dendrogram has already been precomputed.
    If not, `sc.tl.dendrogram` is run with default parameters.

    The results found in `.uns[dendrogram_key]` are used to reorder
    `var_group_labels` and `var_group_positions`.


    Returns
    -------
    dictionary with keys:
    'categories_idx_ordered', 'var_group_names_idx_ordered',
    'var_group_labels', and 'var_group_positions'
    """

    key = _get_dendrogram_key(adata, dendrogram, groupby)

    print("dendrogram_key by ",key)
    if isinstance(groupby, str):
        groupby = [groupby]

    dendro_info = adata.uns[key]
    if groupby != dendro_info["groupby"]:
        raise ValueError(
            "Incompatible observations. The precomputed dendrogram contains "
            f"information for the observation: '{groupby}' while the plot is "
            f"made for the observation: '{dendro_info['groupby']}. "
            "Please run `sc.tl.dendrogram` using the right observation.'"
        )
        
    if categories is None:
        categories = adata.obs[dendro_info["groupby"]].categories

    # order of groupby categories
    categories_idx_ordered = dendro_info["categories_idx_ordered"]
    categories_ordered = dendro_info["categories_ordered"]

    if len(categories) != len(categories_idx_ordered):
        raise ValueError(
            "Incompatible observations. Dendrogram data has "
            f"{len(categories_idx_ordered)} categories but current groupby "
            f"observation {groupby!r} contains {len(categories)} categories. "
            "Most likely the underlying groupby observation changed after the "
            "initial computation of `sc.tl.dendrogram`. "
            "Please run `sc.tl.dendrogram` again.'"
        )

    # reorder var_groups (if any)
    if var_names is not None:
        var_names_idx_ordered = list(range(len(var_names)))

    if var_group_positions:
        if set(var_group_labels) == set(categories):
            positions_ordered = []
            labels_ordered = []
            position_start = 0
            var_names_idx_ordered = []
            for cat_name in categories_ordered:
                idx = var_group_labels.index(cat_name)
                position = var_group_positions[idx]
                _var_names = var_names[position[0] : position[1] + 1]
                var_names_idx_ordered.extend(range(position[0], position[1] + 1))
                positions_ordered.append(
                    (position_start, position_start + len(_var_names) - 1)
                )
                position_start += len(_var_names)
                labels_ordered.append(var_group_labels[idx])
            var_group_labels = labels_ordered
            var_group_positions = positions_ordered
    else:
        var_names_idx_ordered = None

    if var_names_idx_ordered is not None:
        var_names_ordered = [var_names[x] for x in var_names_idx_ordered]
    else:
        var_names_ordered = None

    return dict(
        categories_idx_ordered=categories_idx_ordered,
        categories_ordered=dendro_info["categories_ordered"],
        var_names_idx_ordered=var_names_idx_ordered,
        var_names_ordered=var_names_ordered,
        var_group_labels=var_group_labels,
        var_group_positions=var_group_positions,
    )
def _get_dendrogram_key(adata, dendrogram_key, groupby):
    # the `dendrogram_key` can be a bool an NoneType or the name of the
    # dendrogram key. By default the name of the dendrogram key is 'dendrogram'
    if not isinstance(dendrogram_key, str):
        if isinstance(groupby, str):
            dendrogram_key = f"dendrogram_{groupby}"
        elif isinstance(groupby, list):
            dendrogram_key = f'dendrogram_{"_".join(groupby)}'

    if dendrogram_key not in adata.uns:
        sc.tl.dendrogram(adata, groupby, key_added=dendrogram_key)

    if "dendrogram_info" not in adata.uns[dendrogram_key]:
        raise ValueError(
            f"The given dendrogram key ({dendrogram_key!r}) does not contain "
            "valid dendrogram information."
        )

    return dendrogram_key

def _plot_categories_as_colorblocks(
    groupby_ax: Axes,
    obs_tidy: pd.DataFrame,
    annotation=True,
    colors=None,
    orientation: Literal["top", "bottom", "left", "right"] = "left",
    cmap_name: str = "tab20",
    label=None,
):
    """\
    Plots categories as colored blocks. If orientation is 'left', the categories
    are plotted vertically, otherwise they are plotted horizontally.

    Parameters
    ----------
    groupby_ax
    obs_tidy
    colors
        Sequence of valid color names to use for each category.
    orientation
    cmap_name
        Name of colormap to use, in case colors is None

    Returns
    -------
    ticks position, labels, colormap
    """
    var_names= [label]
    groupby = obs_tidy.index.name #获取需要绘制的labels列表
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if colors is None:
        groupby_cmap = plt.get_cmap(cmap_name)
    else:
        groupby_cmap = ListedColormap(colors, groupby + "_cmap")
    norm = BoundaryNorm(np.arange(groupby_cmap.N + 1) - 0.5, groupby_cmap.N)

    # determine groupby label positions such that they appear
    # centered next/below to the color code rectangle assigned to the category
    value_sum = 0
    ticks = []  # list of centered position of the labels
    labels = []
    label2code = {}  # dictionary of numerical values asigned to each label
    for code, (label, value) in enumerate(
        obs_tidy.index.value_counts(sort=False).items()
    ):
        ticks.append(value_sum + (value / 2))
        labels.append(label)
        value_sum += value
        label2code[label] = code

    groupby_ax.grid(visible=False)

    if orientation == "left":
        groupby_ax.imshow(
            np.array([[label2code[lab] for lab in obs_tidy.index]]).T,
            aspect="auto",
            cmap=groupby_cmap,
            norm=norm,
        )
        if len(labels) > 1:
            groupby_ax.set_yticks(ticks)
            groupby_ax.set_yticklabels(labels)

        # remove y ticks
        groupby_ax.tick_params(axis="y", left=False, labelsize="small")
        # remove x ticks and labels
        groupby_ax.tick_params(axis="x", bottom=True, labelbottom=True)

        # remove surrounding lines
        groupby_ax.spines["right"].set_visible(False)
        groupby_ax.spines["top"].set_visible(False)
        groupby_ax.spines["left"].set_visible(False)
        groupby_ax.spines["bottom"].set_visible(False)
        if annotation is not True:
            groupby_ax.set_ylabel(groupby)
        else:
            # 把刻度和标签放到右侧
            groupby_ax.tick_params(axis="x", labelsize="small", length=2)  # 右侧显示标签，左侧隐藏
            groupby_ax.set_xticks(np.arange(len(var_names)))  # 设置 Y 轴刻度位置
            groupby_ax.set_xticklabels(var_names, rotation=90,fontsize=10)  # 设置 Y 轴标签，水平显示
            groupby_ax.tick_params(axis="y", labelbottom=False, labelleft=False)
            
    else:
        groupby_ax.imshow(
            np.array([[label2code[lab] for lab in obs_tidy.index]]),
            aspect="auto",
            cmap=groupby_cmap,
            norm=norm,
        )
        if len(labels) > 1:
            groupby_ax.set_xticks(ticks)
            if max([len(str(x)) for x in labels]) < 3:
                # if the labels are small do not rotate them
                rotation = 0
            else:
                rotation = 90
            groupby_ax.set_xticklabels(labels, rotation=rotation)

        # remove x ticks
        groupby_ax.tick_params(axis="x", bottom=False, labelsize="small")
        # remove y ticks and labels
        groupby_ax.tick_params(axis="y", left=False, labelleft=False)

        # remove surrounding lines
        groupby_ax.spines["right"].set_visible(False)
        groupby_ax.spines["top"].set_visible(False)
        groupby_ax.spines["left"].set_visible(False)
        groupby_ax.spines["bottom"].set_visible(False)
        if annotation is not True:
            groupby_ax.set_xlabel(groupby)
        else:
            # 把刻度和标签放到右侧
            groupby_ax.tick_params(axis="y", labelsize="small", length=2,
                                   left=False, right=True,   # 右侧显示刻度，左侧不显示
                                   labelleft=False, labelright=True)  # 右侧显示标签，左侧隐藏
            groupby_ax.set_yticks(np.arange(len(var_names)))  # 设置 Y 轴刻度位置
            groupby_ax.set_yticklabels(var_names, rotation=0,fontsize=10)  # 设置 Y 轴标签，水平显示
            groupby_ax.tick_params(axis="x", labelbottom=False, labelleft=False)


    # this True {'2i': 0, 'Unknown': 1, 'serum/LIF': 2} [6.0, 18.0, 34.5] ['2i', 'Unknown', 'serum/LIF'] <matplotlib.colors.ListedColormap object at 0x7f7886af19a0> <matplotlib.colors.BoundaryNorm object at 0x7f7886af1520>
    # this False {'0': 0, '1': 1, '2': 2} [9.5, 26.0, 39.0] ['0', '1', '2'] <matplotlib.colors.ListedColormap object at 0x7f7886a3a580> <matplotlib.colors.BoundaryNorm object at 0x7f7886a53f10>

    #labels are all the type
    #
    return label2code, ticks, labels, groupby_cmap, norm

def _get_palette(adata, values_key: str, palette=None):
    color_key = f"{values_key}_colors"
    if adata.obs[values_key].dtype == bool:
        values = pd.Categorical(adata.obs[values_key].astype(str))
    else:
        values = pd.Categorical(adata.obs[values_key])
    if palette:
        sc.pl._utils._set_colors_for_categorical_obs(adata, values_key, palette)
    elif color_key not in adata.uns or len(adata.uns[color_key]) < len(
        values.categories
    ):
        #  set a default palette in case that no colors or few colors are found
        sc.pl._utils._set_default_colors_for_categorical_obs(adata, values_key)
    else:
        sc.pl._utils._validate_palette(adata, values_key)
    return dict(zip(values.categories, adata.uns[color_key]))
def _check_var_names_type(var_names, var_group_labels, var_group_positions):
    """
    checks if var_names is a dict. Is this is the cases, then set the
    correct values for var_group_labels and var_group_positions

    Returns
    -------
    var_names, var_group_labels, var_group_positions

    """
    if isinstance(var_names, cabc.Mapping):
        if var_group_labels is not None or var_group_positions is not None:
            logg.warning(
                "`var_names` is a dictionary. This will reset the current "
                "value of `var_group_labels` and `var_group_positions`."
            )
        var_group_labels = []
        _var_names = []
        var_group_positions = []
        start = 0
        for label, vars_list in var_names.items():
            if isinstance(vars_list, str):
                vars_list = [vars_list]
            # use list() in case var_list is a numpy array or pandas series
            _var_names.extend(list(vars_list))
            var_group_labels.append(label)
            var_group_positions.append((start, start + len(vars_list) - 1))
            start += len(vars_list)
        var_names = _var_names

    elif isinstance(var_names, str):
        var_names = [var_names]

    return var_names, var_group_labels, var_group_positions
def _plot_colorbar(mappable, fig, subplot_spec, max_cbar_height: float = 1.0):
    """
    Plots a vertical color bar based on mappable.
    The height of the colorbar is min(figure-height, max_cmap_height)

    Parameters
    ----------
    mappable
        The image to which the colorbar applies.
    fig
        The figure object
    subplot_spec
        The gridspec subplot. Eg. axs[1,2]
    max_cbar_height
        The maximum colorbar height

    Returns
    -------
    color bar ax
    """
    width, height = fig.get_size_inches()
    if height > max_cbar_height:
        # to make the colorbar shorter, the
        # ax is split and the lower portion is used.
        axs2 = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=subplot_spec,
            height_ratios=[height - max_cbar_height, max_cbar_height],
        )
        heatmap_cbar_ax = fig.add_subplot(axs2[1])
    else:
        heatmap_cbar_ax = fig.add_subplot(subplot_spec)
    plt.colorbar(mappable, cax=heatmap_cbar_ax)
    return heatmap_cbar_ax

def _plot_dendrogram(
    dendro_ax: Axes,
    adata: AnnData,
    groupby: str,
    *,
    dendrogram_key: str  = None,
    orientation: Literal["top", "bottom", "left", "right"] = "right",
    remove_labels: bool = True,
    ticks: Collection[float]  = None,
):
    """\
    Plots a dendrogram on the given ax using the precomputed dendrogram
    information stored in `.uns[dendrogram_key]`
    """

    dendrogram_key = _get_dendrogram_key(adata, dendrogram_key, groupby)

    def translate_pos(pos_list, new_ticks, old_ticks):
        """\
        transforms the dendrogram coordinates to a given new position.
        The xlabel_pos and orig_ticks should be of the same
        length.

        This is mostly done for the heatmap case, where the position of the
        dendrogram leaves needs to be adjusted depending on the category size.

        Parameters
        ----------
        pos_list
            list of dendrogram positions that should be translated
        new_ticks
            sorted list of goal tick positions (e.g. [0,1,2,3] )
        old_ticks
            sorted list of original tick positions (e.g. [5, 15, 25, 35]),
            This list is usually the default position used by
            `scipy.cluster.hierarchy.dendrogram`.

        Returns
        -------
        translated list of positions

        Examples
        --------
        >>> translate_pos(
        ...     [5, 15, 20, 21],
        ...     [0,  1,  2, 3 ],
        ...     [5, 15, 25, 35],
        ... )
        [0, 1, 1.5, 1.6]
        """
        # of given coordinates.

        if not isinstance(old_ticks, list):
            # assume that the list is a numpy array
            old_ticks = old_ticks.tolist()
        new_xs = []
        for x_val in pos_list:
            if x_val in old_ticks:
                new_x_val = new_ticks[old_ticks.index(x_val)]
            else:
                # find smaller and bigger indices
                idx_next = np.searchsorted(old_ticks, x_val, side="left")
                idx_prev = idx_next - 1
                old_min = old_ticks[idx_prev]
                old_max = old_ticks[idx_next]
                new_min = new_ticks[idx_prev]
                new_max = new_ticks[idx_next]
                new_x_val = ((x_val - old_min) / (old_max - old_min)) * (
                    new_max - new_min
                ) + new_min
            new_xs.append(new_x_val)
        return new_xs

    dendro_info = adata.uns[dendrogram_key]["dendrogram_info"]
    leaves = dendro_info["ivl"]
    icoord = np.array(dendro_info["icoord"])
    dcoord = np.array(dendro_info["dcoord"])

    orig_ticks = np.arange(5, len(leaves) * 10 + 5, 10).astype(float)
    # check that ticks has the same length as orig_ticks
    if ticks is not None and len(orig_ticks) != len(ticks):
        logg.warning(
            "ticks argument does not have the same size as orig_ticks. "
            "The argument will be ignored"
        )
        ticks = None

    for xs, ys in zip(icoord, dcoord):
        if ticks is not None:
            xs = translate_pos(xs, ticks, orig_ticks)
        if orientation in ["right", "left"]:
            xs, ys = ys, xs
        dendro_ax.plot(xs, ys, color="#555555")

    dendro_ax.tick_params(bottom=False, top=False, left=False, right=False)
    ticks = ticks if ticks is not None else orig_ticks
    if orientation in ["right", "left"]:
        dendro_ax.set_yticks(ticks)
        dendro_ax.set_yticklabels(leaves, fontsize="small", rotation=0)
        dendro_ax.tick_params(labelbottom=False, labeltop=False)
        if orientation == "left":
            xmin, xmax = dendro_ax.get_xlim()
            dendro_ax.set_xlim(xmax, xmin)
            dendro_ax.tick_params(labelleft=False, labelright=True)
    else:
        dendro_ax.set_xticks(ticks)
        dendro_ax.set_xticklabels(leaves, fontsize="small", rotation=90)
        dendro_ax.tick_params(labelleft=False, labelright=False)
        if orientation == "bottom":
            ymin, ymax = dendro_ax.get_ylim()
            dendro_ax.set_ylim(ymax, ymin)
            dendro_ax.tick_params(labeltop=True, labelbottom=False)

    if remove_labels:
        dendro_ax.tick_params(
            labelbottom=False, labeltop=False, labelleft=False, labelright=False
        )

    dendro_ax.grid(False)

    dendro_ax.spines["right"].set_visible(False)
    dendro_ax.spines["top"].set_visible(False)
    dendro_ax.spines["left"].set_visible(False)
    dendro_ax.spines["bottom"].set_visible(False)
    
def _plot_gene_groups_brackets(
    gene_groups_ax: Axes,
    *,
    group_positions: Iterable[tuple[int, int]],
    group_labels: Sequence[str],
    left_adjustment: float = -0.3,
    right_adjustment: float = 0.3,
    rotation: float,
    orientation: Literal["top", "right"] = "top",
):
    """Draw brackets that represent groups of genes on the give axis.

    For best results, this axis is located on top of an image whose
    x axis contains gene names.

    The gene_groups_ax should share the x axis with the main ax.

    Eg: gene_groups_ax = fig.add_subplot(axs[0, 0], sharex=dot_ax)

    This function is used by dotplot, heatmap etc.

    Parameters
    ----------
    gene_groups_ax
        In this axis the gene marks are drawn
    group_positions
        Each item in the list, should contain the start and end position that the
        bracket should cover.
        Eg. [(0, 4), (5, 8)] means that there are two brackets, one for the var_names (eg genes)
        in positions 0-4 and other for positions 5-8
    group_labels
        List of group labels
    left_adjustment
        adjustment to plot the bracket start slightly before or after the first gene position.
        If the value is negative the start is moved before.
    right_adjustment
        adjustment to plot the bracket end slightly before or after the last gene position
        If the value is negative the start is moved before.
    rotation
        rotation degrees for the labels. If not given, small labels (<4 characters) are not
        rotated, otherwise, they are rotated 90 degrees
    orientation
        location of the brackets. Either `top` or `right`

    Returns
    -------
    None

    """
    import matplotlib.patches as patches
    from matplotlib.path import Path

    # get the 'brackets' coordinates as lists of start and end positions

    left = [x[0] + left_adjustment for x in group_positions]
    right = [x[1] + right_adjustment for x in group_positions]

    # verts and codes are used by PathPatch to make the brackets
    verts = []
    codes = []
    if orientation == "top":
        # rotate labels if any of them is longer than 4 characters
        if rotation is None and group_labels:
            rotation = 90 if max([len(x) for x in group_labels]) > 4 else 0
        for idx in range(len(left)):
            verts.append((left[idx], 0))  # lower-left
            verts.append((left[idx], 0.6))  # upper-left
            verts.append((right[idx], 0.6))  # upper-right
            verts.append((right[idx], 0))  # lower-right

            codes.append(Path.MOVETO)
            codes.append(Path.LINETO)
            codes.append(Path.LINETO)
            codes.append(Path.LINETO)

            try:
                group_x_center = left[idx] + float(right[idx] - left[idx]) / 2
                gene_groups_ax.text(
                    group_x_center,
                    1.1,
                    group_labels[idx],
                    ha="center",
                    va="bottom",
                    rotation=rotation,
                )
            except Exception:  # TODO catch the correct exception
                pass
    else:
        top = left
        bottom = right
        for idx in range(len(top)):
            verts.append((0, top[idx]))  # upper-left
            verts.append((0.15, top[idx]))  # upper-right
            verts.append((0.15, bottom[idx]))  # lower-right
            verts.append((0, bottom[idx]))  # lower-left

            codes.append(Path.MOVETO)
            codes.append(Path.LINETO)
            codes.append(Path.LINETO)
            codes.append(Path.LINETO)

            try:
                diff = bottom[idx] - top[idx]
                group_y_center = top[idx] + float(diff) / 2
                if diff * 2 < len(group_labels[idx]):
                    # cut label to fit available space
                    group_labels[idx] = group_labels[idx][: int(diff * 2)] + "."
                gene_groups_ax.text(
                    0.6,
                    group_y_center,
                    group_labels[idx],
                    ha="right",
                    va="center",
                    rotation=270,
                    fontsize="small",
                )
            except Exception as e:
                print(f"problems {e}")
                pass

    path = Path(verts, codes)

    patch = patches.PathPatch(path, facecolor="none", lw=1.5)

    gene_groups_ax.add_patch(patch)
    gene_groups_ax.grid(visible=False)
    gene_groups_ax.axis("off")
    # remove y ticks
    gene_groups_ax.tick_params(axis="y", left=False, labelleft=False)
    # remove x ticks and labels
    gene_groups_ax.tick_params(
        axis="x", bottom=False, labelbottom=False, labeltop=False
    )
