#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
> Author: zongwt 
> Created Time: 2023年09月15日

"""
import numpy as np
import pylab as pl
import matplotlib.transforms as mtransforms


################################################################################
# Display correlation matrices

def fit_axes(ax):
    """ Redimension the given axes to have labels fitting.
    """
    # Horizontal
    bboxes = []
    for label in ax.get_yticklabels():
        bbox = label.get_window_extent()
        # the figure transform goes from relative coords->pixels and we
        # want the inverse of that
        bboxi = bbox.inverse_transformed(ax.figure.transFigure)
        bboxes.append(bboxi)

    # this is the bbox that bounds all the bboxes, again in relative
    # figure coords
    bbox = mtransforms.Bbox.union(bboxes)
    if ax.get_position().xmin < 1.1 * bbox.width:
        # we need to move it over
        new_position = ax.get_position()
        new_position.x0 = 1.1 * bbox.width  # pad a little
        ax.set_position(new_position)

    # Vertical
    bboxes = []
    for label in ax.get_xticklabels():
        bbox = label.get_window_extent()
        # the figure transform goes from relative coords->pixels and we
        # want the inverse of that
        bboxi = bbox.inverse_transformed(ax.figure.transFigure)
        bboxes.append(bboxi)

    # this is the bbox that bounds all the bboxes, again in relative
    # figure coords
    bbox = mtransforms.Bbox.union(bboxes)
    if ax.get_position().ymin < 1.1 * bbox.height:
        # we need to move it over
        new_position = ax.get_position()
        new_position.y0 = 1.1 * bbox.height  # pad a little
        ax.set_position(new_position)


def plot_correlation(mat, tri='lower', text=None, labels=None,
                     auto_fit=True, grid=(.8, .8, .8), colorbar=False,
                     **kwargs):
    """ Plot the given correlation matrix.
        Parameters
        ==========
        tri: {'lower', 'diag', 'full'}
            Which triangular part of the correlation matrix to plot:
            'lower' is the lower part, 'diag' is the lower including
            diagonal, and 'full' is the full matrix.
        text: string or None
            A text to add in the upper left corner.
        labels: list of strings
            The label of each row and column
        auto_fit: boolean, optional
            If auto_fit is True, the axes are dimensioned to give room
            for the labels. This assumes that the labels are resting
            against the bottom and left edges of the figure.
        grid: color or False
            If not, a gray grid is plotted to separate rows and columns
            using the given color.
        colorbar: boolean
            If True, an integrated colorbar is added.
        kwargs: extra keyword arguments
            Extra keyword arguments are sent to pylab.imshow
    """
    if tri == 'lower':
        mask = np.tri(mat.shape[0], k=-1, dtype=np.bool) - True
        mat = np.ma.masked_array(mat, mask)
    elif tri == 'diag':
        mask = np.tri(mat.shape[0], dtype=np.bool) - True
        mat = np.ma.masked_array(mat, mask)
    obj = pl.imshow(mat, aspect='equal',
                    interpolation='nearest',
                    **kwargs)
    ax = pl.gca()
    ax.set_autoscale_on(False)
    ymin, ymax = pl.ylim()
    if labels is False:
        ax.xaxis.set_major_formatter(pl.NullFormatter())
        ax.yaxis.set_major_formatter(pl.NullFormatter())
    elif labels is not None:
        pl.xticks(np.arange(len(labels)), labels, size='x-small')
        for label in pl.gca().get_xticklabels():
            label.set_ha('right')
            label.set_rotation(50)
        pl.yticks(np.arange(len(labels)), labels, size='x-small')
        for label in pl.gca().get_yticklabels():
            label.set_ha('right')
            label.set_rotation(10)

    if colorbar:
        bb = ax.get_position()
        ax_cbar = pl.axes([bb.x1 - .05 * bb.width,
                           bb.y0 + .2 * bb.height,
                           0.04 * bb.width, 0.72 * bb.height])
        pl.xticks(())
        pl.colorbar(mappable=obj, cax=ax_cbar, orientation='vertical')
        ax_cbar.yaxis.tick_left()
        pl.axes(ax)

    if text is not None:
        pl.text(0.9 - .15 * colorbar, 0.9 + .05 * colorbar, text,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)

    if grid is not False:
        size = len(mat)
        for i in range(size):
            # Correct for weird mis-sizing
            i = 1.001 * i
            pl.plot([i + 0.5, i + 0.5], [size - 0.5, i + 0.5], color=grid)
            pl.plot([i + 0.5, -0.5], [i + 0.5, i + 0.5], color=grid)

    pl.ylim(ymin, ymax)
    if auto_fit and labels is not None and labels is not False:
        pl.draw()
        fit_axes(ax)
    return obj