import seaborn as sns
import matplotlib as mpl
from matplotlib import cm, colors, rcParams
import warnings
from cycler import cycler
from packaging.version import parse

"""
basic var settings

"""
"""Settings
"""

verbosity = 3
"""Verbosity level (0=errors, 1=warnings, 2=info, 3=hints)
"""

plot_prefix = "scmat_"
"""Global prefix that is appended to figure filenames.
"""

plot_suffix = ""
"""Global suffix that is appended to figure filenames.
"""

file_format_data = "h5ad"
"""File format for saving AnnData objects.
Allowed are 'txt', 'csv' (comma separated value file) for exporting and 'h5ad'
(hdf5) for lossless saving.
"""

file_format_figs = "png"
"""File format for saving figures.
For example 'png', 'pdf' or 'svg'. Many other formats work as well (see
`matplotlib.pyplot.savefig`).
"""

autosave = False
"""Save plots/figures as files in directory 'figs'.
Do not show plots/figures interactively.
"""

autoshow = True
"""Show all plots/figures automatically if autosave == False.
There is no need to call the matplotlib pl.show() in this case.
"""

writedir = "./write/"
"""Directory where adata is stored (default './write/').
"""

cachedir = "./cache/"
"""Default cache directory.
"""

figdir = "./figures/"
"""Directory where plots are saved (default './figures/').
"""

max_memory = 15
"""Maximal memory usage in Gigabyte.
Is currently not well respected....
"""

n_jobs = 1
"""Default number of jobs/ CPUs to use for parallel computing.
"""
log_dir = 'logs'  # 日志文件目录

logfile = "scm.log"
"""Name of logfile. By default is set to '' and writes to standard output."""

categories_to_ignore = ["N/A", "dontknow", "no_gate", "?"]
"""Categories that are omitted in plotting etc.
"""

presenter_view = None
"""Set True for maximum width of 12.
"""

_frameon = False
"""See set_figure_params.
"""

_rcParams_style = None
"""See set_figure_params.
"""

vega_10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------

warnings.filterwarnings("ignore", category=mpl.MatplotlibDeprecationWarning)


# def set_figure_params(context='paper',style='white',palette='deep',font='sans-serif',font_scale=1.1,color_codes=True,
#                       dpi=80,dpi_save=150,figsize=[5.4, 4.8],rc=None):
#     """ Set global parameters for figures. Modified from sns.set()
#     Parameters
#     ----------
#     context : string or dict
#         Plotting context parameters, see seaborn :func:`plotting_context
#     style: `string`,optional (default: 'white')
#         Axes style parameters, see seaborn :func:`axes_style`
#     palette : string or sequence
#         Color palette, see seaborn :func:`color_palette`
#     font_scale: `float`, optional (default: 1.3)
#         Separate scaling factor to independently scale the size of the font elements.        
#     color_codes : `bool`, optional (default: True)
#         If ``True`` and ``palette`` is a seaborn palette, remap the shorthand
#         color codes (e.g. "b", "g", "r", etc.) to the colors from this palette.
#     dpi: `int`,optional (default: 80)
#         Resolution of rendered figures.
#     dpi_save: `int`,optional (default: 150)
#         Resolution of saved figures.
#     rc: `dict`,optional (default: None)
#         rc settings properties. Parameter mappings to override the values in the preset style.
#         Please see https://matplotlib.org/tutorials/introductory/customizing.html#a-sample-matplotlibrc-file
#     """
# #     mpl.rcParams.update(mpl.rcParamsDefault)
    
#     sns.set_theme(context=context,style=style,palette=palette,font=font,font_scale=font_scale,color_codes=color_codes,
#             rc={'figure.dpi':dpi,
#                 'savefig.dpi':dpi_save,
#                 'figure.figsize':figsize,
#                 'image.cmap': 'viridis',
#                 'lines.markersize':6,
#                 'legend.columnspacing':0.1,
#                 'legend.borderaxespad':0.1,
#                 'legend.handletextpad':0.1,
#                 'pdf.fonttype':42,})
#     if(rc is not None):
#         assert isinstance(rc,dict),"rc must be dict"  
#         for key, value in rc.items():
#             if key in mpl.rcParams.keys():
#                 mpl.rcParams[key] = value
#             else:
#                 raise Exception("unrecognized property '%s'" % key)

def set_rcParams_scm(fontsize=12, color_map=None, frameon='small'):
    """Set matplotlib.rcParams to scvelo defaults."""
    # dpi options (mpl default: 100, 100)
    rcParams["figure.dpi"] = 100
    rcParams["savefig.dpi"] = 150

    # figure (mpl default: 0.125, 0.96, 0.15, 0.91)
    rcParams["figure.figsize"] = (5.4, 4.8)
    rcParams["figure.subplot.left"] = 0.18
    rcParams["figure.subplot.right"] = 0.96
    rcParams["figure.subplot.bottom"] = 0.15
    rcParams["figure.subplot.top"] = 0.91

    # lines (defaults:  1.5, 6, 1)
    rcParams["lines.linewidth"] = 1.5  # the line width of the frame
    rcParams["lines.markersize"] = 6
    rcParams["lines.markeredgewidth"] = 1

    # font
    rcParams["font.sans-serif"] = [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "sans-serif",
    ]
    rcParams["font.serif"] = [
        'Times New Roman', 
        'Times',
        'DejaVu Serif', 
        'Palatino', 'Charter', 'serif',
    ]

    fontsize = fontsize
    labelsize = 0.92 * fontsize

    # fonsizes (mpl default: 10, medium, large, medium)
    rcParams["font.size"] = fontsize
    rcParams["legend.fontsize"] = labelsize
    rcParams["axes.titlesize"] = fontsize
    rcParams["axes.labelsize"] = labelsize

    # legend (mpl default: 1, 1, 2, 0.8)
    rcParams["legend.numpoints"] = 1
    rcParams["legend.scatterpoints"] = 1
    rcParams["legend.handlelength"] = 0.5
    rcParams["legend.handletextpad"] = 0.4

    # color cycle
    rcParams["axes.prop_cycle"] = cycler(color=vega_10)

    # axes
    rcParams["axes.linewidth"] = 0.8
    rcParams["axes.edgecolor"] = "black"
    rcParams["axes.facecolor"] = "white"

    # ticks (mpl default: k, k, medium, medium)
    rcParams["xtick.color"] = "k"
    rcParams["ytick.color"] = "k"
    rcParams["xtick.labelsize"] = labelsize
    rcParams["ytick.labelsize"] = labelsize

    # axes grid (mpl default: False, #b0b0b0)
    rcParams["axes.grid"] = False
    rcParams["grid.color"] = ".8"

    # color map
    rcParams["image.cmap"] = "RdBu_r" if color_map is None else color_map

    # frame (mpl default: True)
    frameon = False if frameon is None else frameon
    global _frameon
    _frameon = frameon


def set_rcParams_scanpy(fontsize=12, color_map=None, frameon=None):
    """Set matplotlib.rcParams to Scanpy defaults."""
    # dpi options
    rcParams["figure.dpi"] = 100
    rcParams["savefig.dpi"] = 150

    # figure
    rcParams["figure.figsize"] = (4, 4)
    rcParams["figure.subplot.left"] = 0.18
    rcParams["figure.subplot.right"] = 0.96
    rcParams["figure.subplot.bottom"] = 0.15
    rcParams["figure.subplot.top"] = 0.91

    rcParams["lines.linewidth"] = 1.5  # the line width of the frame
    rcParams["lines.markersize"] = 6
    rcParams["lines.markeredgewidth"] = 1

    # font
    rcParams["font.sans-serif"] = [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "sans-serif",
    ]

    fontsize = fontsize
    labelsize = 0.92 * fontsize

    rcParams["font.size"] = fontsize
    rcParams["legend.fontsize"] = labelsize
    rcParams["axes.titlesize"] = fontsize
    rcParams["axes.labelsize"] = fontsize

    # legend
    rcParams["legend.numpoints"] = 1
    rcParams["legend.scatterpoints"] = 1
    rcParams["legend.handlelength"] = 0.5
    rcParams["legend.handletextpad"] = 0.4

    # lines
    rcParams["axes.linewidth"] = 0.8
    rcParams["axes.edgecolor"] = "black"
    rcParams["axes.facecolor"] = "white"

    # ticks
    rcParams["xtick.color"] = "k"
    rcParams["ytick.color"] = "k"
    rcParams["xtick.labelsize"] = fontsize
    rcParams["ytick.labelsize"] = fontsize

    # axes grid
    rcParams["axes.grid"] = True
    rcParams["grid.color"] = ".8"

    # color map
    rcParams["image.cmap"] = rcParams["image.cmap"] if color_map is None else color_map

    # frame
    frameon = True if frameon is None else frameon
    global _frameon
    _frameon = frameon


            
def set_rcParams_defaults():
    """Reset `matplotlib.rcParams` to defaults."""
    from matplotlib import rcParamsDefault

    rcParams.update(rcParamsDefault)

def set_figure_params(
    style="scm",
    dpi=100,
    dpi_save=150,
    font="sans-serif",
    frameon=None,
    vector_friendly=True,
    transparent=True,
    fontsize=12,
    figsize=None,
    color_map=None,
    facecolor=None,
    format="pdf",
    ipython_format="png2x",
):
    """Set resolution/size, styling and format of figures.

    Arguments:
    ---------
    style : `str` (default: `None`)
        Init default values for ``matplotlib.rcParams`` suited for `scvelo` or `scanpy`.
        Use `None` for the default matplotlib values.

    dpi : `int` (default: `None`)
        Resolution of rendered figures - affects the size of figures in notebooks.
    dpi_save : `int` (default: `None`)
        Resolution of saved figures. This should typically be higher to achieve
        publication quality.
    frameon : `bool` (default: `None`)
        Add frames and axes labels to scatter plots.
    vector_friendly : `bool` (default: `True`)
        Plot scatter plots using `png` backend even when exporting as `pdf` or `svg`.
    transparent : `bool` (default: `True`)
        Save figures with transparent back ground. Sets
        `rcParams['savefig.transparent']`.
    fontsize : `int` (default: 14)
        Set the fontsize for several `rcParams` entries.
    figsize: `[float, float]` (default: `None`)
        Width and height for default figure size.
    color_map : `str` (default: `None`)
        Convenience method for setting the default color map.
    facecolor : `str` (default: `None`)
        Sets backgrounds `rcParams['figure.facecolor']`
        and `rcParams['axes.facecolor']` to `facecolor`.
    format : {'png', 'pdf', 'svg', etc.} (default: 'pdf')
        This sets the default format for saving figures: `file_format_figs`.
    ipython_format : list of `str` (default: 'png2x')
        Only concerns the notebook/IPython environment; see
        `IPython.core.display.set_matplotlib_formats` for more details.
    """
    global _rcParams_style
    _rcParams_style = style
    global _vector_friendly
    _vector_friendly = vector_friendly
    global file_format_figs
    file_format_figs = format
    rcParams["pdf.fonttype"] = 42
    if transparent is not None:
        rcParams["savefig.transparent"] = transparent
    if facecolor is not None:
        rcParams["figure.facecolor"] = facecolor
        rcParams["axes.facecolor"] = facecolor
    if font is not None:
        rcParams["font.family"] = font
    if style == "scm":
        set_rcParams_scm(fontsize=fontsize, color_map=color_map, frameon=frameon)
    elif style == "scanpy":
        set_rcParams_scanpy(fontsize=fontsize, color_map=color_map, frameon=frameon)
    # Overwrite style options if given
    if figsize is not None:
        rcParams["figure.figsize"] = figsize
    if dpi is not None:
        rcParams["figure.dpi"] = dpi
    if dpi_save is not None:
        rcParams["savefig.dpi"] = dpi_save

def _set_start_time():
    from time import time

    return time()


# ------------------------------------------------------------------------------
# Private global variables & functions
# ------------------------------------------------------------------------------

_vector_friendly = False
"""Set to true if you want to include pngs in svgs and pdfs.
"""

_low_resolution_warning = True
"""Print warning when saving a figure with low resolution."""

_start = _set_start_time()
"""Time when the settings module is first imported."""

_previous_time = _start
"""Variable for timing program parts."""

_previous_memory_usage = -1
"""Stores the previous memory usage."""
