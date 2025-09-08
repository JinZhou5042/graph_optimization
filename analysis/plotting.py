"""
Created on Fri Mar 24 10:18:22 2023

@author: atownse2
"""
import numpy as np
# import hist

import matplotlib as mpl
import matplotlib.pyplot as plt

# import arviz as az
import mplhep as hep

# import seaborn as sns
# sns.set_palette('colorblind')

import random
random_string = lambda: ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10))

hep.style.use(hep.style.CMS)

colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
marker_styles = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'P', '*', 'X', 'd', 'H', 'h', '+', 'x', '|', '_']
line_styles = ['solid', 'dotted', 'dashed', 'dashdot', 'solid', 'dotted', 'dashed', 'dashdot']

defaults = {
    'figsize': (6, 6),
    'fontsize': 8,

    'yerr': None,

    'fig': None,
    'ax': None,

    'xlabel': None,
    'ylabel': "Events",

    'subtitle': None,
    'subtitle_coord': (0.5, 0.9),

    'hist_flow': False, # True or False
    'mpl_flow': 'none', # show, sum, hint, none

    'is_cms_data': False,
    'do_cms_label': True,
    'cms_label': 'Work In Progress',
    'cms_font_size' : 12,
    'year': None,

    'setlogy': False,
    'setlogx': False,
    'setlogz': False,

    'cmap': 'binary',
    'colors': list(mpl.cm.get_cmap('tab10').colors),

    'legend_loc': 'best',
    'xlim': None,
    'xticks': None,
    'density': False,
    'zero_supress': False,

    'ratio_histtype': 'errorbar',

    'save_as': None
}

hist_attrs = ['hists', 'labels', 'colors', 'histtypes', 'yerrs']

def is_hist(h):
    if isinstance(h, hgm.Histogram):
        return True
    elif isinstance(h, bh.Hist):
        return True
    elif isinstance(h, bh.Stack):
        return True
    elif isinstance(h, tuple) and len(h) == 2:
        return True
    return False


class PlotConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(**defaults)

        if 'config' in kwargs:
            self.update(kwargs['config'])
            return

        self.parse_args(args)
        self.parse_kwargs(kwargs)

    def parse_args(self, args):
        if len(args) == 0:
            return

        if len(args) == 1:
            if isinstance(args[0], list):
                self['hists'] = args[0]
            else:
                self['hists'] = [args[0]]
        else:
            if all(is_hist(h) for h in args):
                self['hists'] = args
            elif all(isinstance(_, list) for _ in args):
                self.update({plural_hist_attrs[i]: args[i] for i in range(len(args))})

    def parse_kwargs(self, kwargs):
        # Can pass hists as a list or as individual arguments
        for attr in hist_attrs:
            if attr in kwargs:
                self[attr] = kwargs[attr]
                del kwargs[attr]
            if attr[:-1] in kwargs:
                self[attr] = [kwargs[attr[:-1]]]
                del kwargs[attr[:-1]]
        
        self.update(kwargs)

    def plot_config(self, i):
        _c = {'ax': self.ax, 'density': self.density}

        for attr in hist_attrs:
            if attr == 'hists': continue
            if self[attr] is not None and len(self[attr]) > i:
                _c[attr[:-1]] = self[attr][i]
 
        if self.is_cms_data and self.labels is None:
            _c['color'] = 'black'
            _c['histtype'] = 'errorbar'

        if self.stack is not None:
            _c['stack'] = self.stack
            _c['histtype'] = 'fill'
            if 'color' in _c:
                del _c['color']

        if self.flow is not None:
            _c['flow'] = self.flow

        return _c

    def __getitem__(self, key):
        if "all" in key and key != "all_axs":
            _key = key.replace("all_", "")
            _item = self.get(_key, []) + self.get(f"ratio_{_key}", [])
            if len(_item) == 0:
                return None
            return _item
        
        return self.get(key, None)

    def __getattr__(self, key: str):
        return self[key]
    
    def __setattr__(self, key: str, value):
        self[key] = value
    
    def copy(self, **kwargs):
        raise NotImplementedError("Use update instead")
    
def split_axes(ax, divider=0.25, spacing=0):
    """
    Splits an existing axis into a main and a ratio axis.
    
    Parameters:
    - ax: The original matplotlib axis to be split.
    - divider: Fraction of the figure height allocated to the bottom (ratio) plot.
    - spacing: Spacing between the main and ratio plots as a fraction of the figure height.
    
    Returns:
    - main_ax: The axis for the main plot.
    - ratio_ax: The axis for the ratio plot.
    """
    fig = ax.figure
    box = ax.get_position()
    height_ratio = [1 - divider - spacing, divider]
    gs = mpl.gridspec.GridSpec(2, 1, figure=fig, left=box.x0, right=box.x1, bottom=box.y0, top=box.y1,
                  height_ratios=height_ratio, hspace=spacing)
    
    # Remove the original axis
    fig.delaxes(ax)
    
    # Create new axes
    main_ax = fig.add_subplot(gs[0])
    ratio_ax = fig.add_subplot(gs[1], sharex=main_ax)
    
    # Hide x-ticks for main_ax to avoid overlap
    plt.setp(main_ax.get_xticklabels(), visible=False)
    
    return main_ax, ratio_ax

def cms_label(config: PlotConfig):
    if config['do_cms_label'] is True:
        hep.cms.label(
            ax=config.ax if config.ax is not None else config.main_ax,
            year=config.era,
            label=config['cms_label'],
            data=config["is_cms_data"],
            lumi=config["lumi"],
            fontsize=config['cms_font_size']
            )

def setup_ax(config: PlotConfig):

    if config.ratio_hists and config.ratio_ax is None:
        if config.all_axs is not None:
            config.ax, config.ratio_ax = config.all_axs
        elif config.ax is not None:
            config.ax, config.ratio_ax = split_axes(config.ax)
        else:
            config.fig, (config.ax, config.ratio_ax) = plt.subplots(
                2, 1,
                figsize=config.figsize,
                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0},
                sharex=True
                )

    if config.ax is None:
        config.fig, config.ax = plt.subplots(figsize=config.figsize)

    if config.simple is True:
        return

    ax = config.ax

    if config.setlogy is True: ax.set_yscale('log')
    if config.setlogx is True: ax.set_xscale('log')

    ylabel = config.ylabel
    if ylabel is not None and config.plot2D is None:
        ax.set_ylabel(ylabel)

    fontsize = config['fontsize']
    subtitle = config['subtitle']
    if subtitle is not None:
        ax.text(
            *config['subtitle_coord'],
            subtitle,
            ha='center', va='bottom', transform=ax.transAxes, fontsize=fontsize
            )

    cms_label(config)

def finalize_plot(config: PlotConfig):
    if config.simple:        
        return

    if config.labels is not None:
        config.ax.legend(loc=config.legend_loc, fontsize=config.fontsize)
    
    if config.plot_ratio:
        # Remove the lowest y-tick from the main axis
        plt.setp(config.ax.get_yticklabels()[0], visible=False)
        config.ax.legend(loc=config.legend_loc, fontsize=config.fontsize)
    
    # Set the size of the x and y axis labels and ticks
    if config.fontsize is not None:
        config.ax.tick_params(axis='both', which='major', labelsize=config.fontsize)
        config.ax.xaxis.label.set_size(config.fontsize)
        config.ax.yaxis.label.set_size(config.fontsize)

    save_as = config['save_as']
    if save_as is not None:
        plt.savefig(save_as)

def plot(*args, **kwargs):
    config = PlotConfig(*args, **kwargs)

    if len(config.all_hists) == 0:
        raise ValueError("Must provide hists or ratio_hists")

    # Check if we are plotting 1D or 2D
    if len(config.all_hists[0].axes) == 1:
        return plot1d(config=config)
    else:
            return plot2d(config=config)

def plot1d(*args, **kwargs):
    config = PlotConfig(*args, **kwargs)
    setup_ax(config)


    if config.plot_ratio:
        ratio_config = plot_ratio(**config)
        config.min_ratio = ratio_config.min_ratio
        config.max_ratio = ratio_config.max_ratio

    for i, h in enumerate(config.hists):
        if config.stack is not None:
            h.plot(stack=True, ax=config.ax, histtype='fill', alpha=0.6)
        else:
            hep.histplot(h, **config.plot_config(i))

    finalize_plot(config)

    if config.return_config:
        return config

def plot2d(**kwargs):
    config = PlotConfig(**kwargs)
    config.plot2D = True
    setup_ax(config)

    for i, h in enumerate(config.hists):
        hep.hist2dplot(
            h,
            ax=config.ax,
            cmap=config.cmap,
            flow=config.flow,
            norm=mpl.colors.LogNorm() if config.setlogz else None,
            # cmin=1,
            alpha=config.alpha,
            cbar=config.cbar,
            )

        if config.display_bin_values:
            H, xedges, yedges = h.to_numpy(flow=True if config.flow == 'show' else False)
            for i in range(len(xedges)-1):
                for j in range(len(yedges)-1):
                    config.ax.text(
                        (xedges[i]+xedges[i+1])/2,
                        (yedges[j]+yedges[j+1])/2,
                        np.round(H[i,j], decimals=2), 
                        ha='center', va='center', color='black'
                        )
    
    finalize_plot(config)

    if config.return_config is True:
        return config

def ratio_subplot(**kwargs):
    subplot_kwargs = {"figsize": (12, 12), "gridspec_kw": {'height_ratios': [3, 1], 'hspace': 0}}
    subplot_kwargs.update(kwargs)
    fig, axs = plt.subplots(2, 1, **subplot_kwargs)
    return fig, axs

def plot_ratio(**kwargs):

    config = PlotConfig(**kwargs)
    
    ratio_hists = config.hists[:2]
    ratio_labels = config.labels[:2]
    ratio_ax = config.ratio_ax

    h_num, h_den = ratio_hists
    l_num, l_den = ratio_labels

    # Calculate ratio
    flow = True if config.flow == 'show' else False

    num = h_num.values(flow=flow)
    den = h_den.values(flow=flow)
    num_var = h_num.variances(flow=flow)
    den_var = h_den.variances(flow=flow)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = num/den
        if config.ratio_error is None:
            ratio_var = (num_var/den**2 + den_var*(num/den**2)**2)
        elif config.ratio_error == 'fraction':
            ratio_var = num_var/den**2

    # Copy h_num
    h_ratio = h_num.copy().reset()

    h_ratio.values(flow=flow)[...] = ratio
    h_ratio.variances(flow=flow)[...] = ratio_var

    hep.histplot(
        h_ratio,
        flow=config.flow,
        ax=ratio_ax,
        label=f'{l_num}/{l_den}',
        color='black',
        histtype=config.ratio_histtype,
        )

    _min = ratio - np.sqrt(ratio_var)
    _max = ratio + np.sqrt(ratio_var)
    ratio_ax.set_ylim(_min.min(), _max.max())
    config.min_ratio = _min.min()
    config.max_ratio = _max.max()

    if config.fit_ratio:
        from scipy.optimize import curve_fit

        # Fit the ratio to a constant
        def constant(x, c):
            return c*np.ones_like(x)
        
        bin_values, bin_edges = h_ratio.to_numpy(flow=flow)
        bin_centers = (bin_edges[1:] + bin_edges[:-1])/2

        _mask = np.isfinite(ratio) & (ratio_var > 0)

        x = bin_centers[_mask]
        y = ratio[_mask]
        yerr = np.sqrt(ratio_var[_mask])

        if not np.isfinite(x[-1]):
            x[-1] = bin_edges[-2] + (bin_edges[-2] - bin_edges[-3])
        if not np.isfinite(x[0]):
            x[0] = x[1] - (x[2] - x[1])

        popt, pcov = curve_fit(constant, x, y, sigma=yerr)
        perr = np.sqrt(np.diag(pcov))

        # Plot line at 1
        # ratio_ax.axhline(1, color='black', linestyle='--')

        # Plot the straight line
        _x = np.linspace(*ratio_ax.get_xlim(), 100)
        ratio_ax.plot(
            _x, constant(_x, popt[0]),
            label=f'Fit: {popt[0]:.3f} $\\pm$ {perr[0]:.3f}',
            color='black')
        ratio_ax.fill_between(
            _x,
            constant(_x, popt[0]-perr),
            constant(_x, *popt[0]+perr),
            color='black',
            alpha=0.3)

    if config.simple:
        return config
    
    ratio_ax.set_ylabel('Ratio')
    ratio_ax.legend(loc=config['legend_loc'], fontsize=config['fontsize'])

    return config

# def plot_autocorr(fit, var_names):
#     fig, axs = plt.subplots( len(var_names)*4, 1, figsize=(15, 8*len(var_names)), sharex=True, gridspec_kw={"hspace": 0})
#     az.plot_autocorr(fit, var_names=var_names, grid=(3*4,1), max_lag=50, ax=axs)
#     for i, ax in enumerate(axs):
#         ax.set_ylabel(axs[i].get_title(), rotation=0, ha='right')
#         ax.set_title("")
#     plt.tight_layout()

# def plot_chains(fit, var_names, start=0, stop=-1, inc_warmup=False):
#     column_names = fit.column_names
#     draws = fit.draws(inc_warmup=inc_warmup)
#     n_draws, n_chains, n_vars = draws.shape

#     all_var_names = []
#     for var in var_names:
#         some_names = [name for name in column_names if var==name or var+"[" in name]
#         if len(some_names) == 0:
#             raise ValueError(f"No variable matching {var} found in the fit.")
#         all_var_names.extend(some_names)

#     fig, ax = plt.subplots(len(all_var_names), 1, figsize=(20, 4*len(all_var_names)), sharex=True, gridspec_kw={"hspace": 0})

#     for i, var in enumerate(all_var_names):
#         draws_index = column_names.index(var)
#         for chain in range(n_chains):
#             ax[i].plot(draws[:, chain, draws_index][start:stop], alpha=0.5, label=f"Chain {chain}")
#         ax[i].set_ylabel(var)
#     ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

#     plt.show()

# def plot_trace(fit, var_names, start=0, end=None):
#     axs = az.plot_trace(
#         fit, var_names=var_names,
#         compact=False, combined=False,
#         figsize=(20, 6*len(var_names)),
#     )
#     for ax in axs[:,1]:
#         ax.set_xlim(start, end)

# def plot_pair(fit, var_names, size=4):
#     az.plot_pair(fit, var_names=var_names, marginals=True, textsize=20, figsize=(15, 15), )#backend_kwargs={"gridspec_kw": {"hspace": 0, "wspace": 0}})

def plot_ellipse(ax, pars, **kwargs):

    ellipse = mpl.patches.Ellipse(
        (pars['cx'], pars['cy']), 2*pars['a'], 2*pars['b'],
        angle=np.degrees(pars['theta']),
        alpha=0.5,  # Set alpha value to make the ellipse transparent
        **kwargs
        )
    ax.add_patch(ellipse)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_2D_gaus(ax, pars, n_std=1):
    mean = pars["mu"]
    cov = pars["Sigma"]
    # Eigenvalues and eigenvectors of the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    
    # Angle of the ellipse
    vx, vy = eigvecs[:, 0]
    theta = np.degrees(np.arctan2(vy, vx))

    # Width and height of the ellipse
    width, height = 2 * n_std * np.sqrt(eigvals)
    
    # Create the Ellipse patch
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, alpha=0.2)

    # Add the ellipse to the plot
    ax.add_patch(ellipse)

from matplotlib.widgets import Slider
def plot_slices(h2d, fit, reverse=False, setlogy=False):

    max_y = np.max(h2d.values()) * 1.1
    if reverse:
        marginal = h2d[:,sum]
        valinit = np.argmax(marginal.values())
        slider_len = h2d.axes[0].size-1
        slice_edges = h2d.axes[0].edges
        x_edges = h2d.axes[1].edges
        hslice = lambda i: h2d[i,:]
        lambda_slice = lambda i: fit.stan_variable("lambda")[i]

    else:
        marginal = h2d[sum,:]
        valinit = np.argmax(marginal.values())
        slider_len = h2d.axes[1].size-1
        slice_edges = h2d.axes[1].edges
        x_edges = h2d.axes[0].edges
        hslice = lambda i: h2d[:,i]


    # Setup the figure and axis for the plot and the slider
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)  # Make space for the slider

    def draw_plot(slice_index):
        lambdas = fit.stan_variable("lambda")
        ax.clear()
        if reverse:
            h_slice = h2d[slice_index,:]
        else:
            h_slice = h2d[:,slice_index]
        plot(hist=h_slice, labels=["data"], ax=ax, simple=True, color="black")
        if len(lambdas.shape) == 3:
            m, u, l = np.percentile(lambdas, [50, 97.5, 2.5], axis=0)
            if reverse:
                m = np.transpose(m)
                u = np.transpose(u)
                l = np.transpose(l)
            ax.step(x_edges, np.concatenate((m[:,slice_index], [0])), where="post", label="fit", )#yerr=[np.concatenate((m-l, [0])), np.concatenate((u-m, [0]))])
            ax.fill_between(x_edges, np.concatenate((u[:,slice_index], [0])), np.concatenate((l[:, slice_index], [0])), color="gray", alpha=0.5, step="post")
        else:
            if reverse:
                lambdas = np.transpose(lambdas)
            ax.step(x_edges, np.concatenate((lambdas[:, slice_index], [0])), where="post", label="fit")

        if setlogy:
            ax.set_yscale("log")
        ax.set_ylim(0, max_y)
        ax.legend()
        ax.set_title(f"Slice : {slice_edges[slice_index]:.0f} < x1 < {slice_edges[slice_index+1]:.0f}")

    draw_plot(valinit)

    # Create the slider axis and the slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])  # Position: [left, bottom, width, height]
    slider = Slider(ax_slider, 'Slice', 0, slider_len, valinit=valinit, valfmt='%0.0f')

    # Update function for the slider
    def update(val):
        slice_index = int(slider.val)
        draw_plot(slice_index)
        fig.canvas.draw_idle()

    # Call update function when slider value is changed
    slider.on_changed(update)

    plt.show()

def plot_2D_fit(
        h2d, fit,
        real_rates=None,
        ):

    # Start with the marginals
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot the data
    plot(h2d[:,sum], ax=axs[0], label="toy data", color='black', histtype='errorbar')
    plot(h2d[sum,:], ax=axs[1], label="toy data", color='black', histtype='errorbar')

    if real_rates is not None:
        axs[0].step(h2d.axes[0].edges, real_rates.sum(axis=1), where="post", label="real intensity", color="orange")
        axs[1].step(h2d.axes[1].edges, real_rates.sum(axis=0), where="post", label="real intensity", color="orange")

    lambdas = fit.stan_variable("lambda")
    if len(lambdas.shape) == 3:
        m, u, l = np.percentile(lambdas, [50, 97.5, 2.5], axis=0)
        print(m.shape)

        axs[0].step(h2d.axes[0].edges, np.concatenate((m.sum(axis=1), [0])), where="post", label="fit")
        axs[0].fill_between(h2d.axes[0].edges, np.concatenate((u.sum(axis=1), [0])), np.concatenate((l.sum(axis=1), [0])), color="gray", alpha=0.5, step="post")

        axs[1].step(h2d.axes[1].edges, np.concatenate((m.sum(axis=0), [0])), where="post", label="fit")
        axs[1].fill_between(h2d.axes[1].edges, np.concatenate((u.sum(axis=0), [0])), np.concatenate((l.sum(axis=0), [0])), color="gray", alpha=0.5, step="post")
    else:
        axs[0].step(h2d.axes[0].edges, np.concatenate((lambdas.sum(axis=1), [0])), where="post", label="fit")
        axs[1].step(h2d.axes[1].edges, np.concatenate((lambdas.sum(axis=0), [0])), where="post", label="fit")
    
    axs[0].set_ylabel("Counts")
    axs[1].set_ylabel("Counts")
    axs[1].set_xlabel("x1")

    axs[0].legend()
    axs[1].legend()

    axs[0].set_yscale("log")
    axs[1].set_yscale("log")

    fig.tight_layout()



def plot_1D_fit(
        h,
        fit_values,
        axs=None,
        hist_label="toy data",
        fit_label="fit",
        fit_color="blue",
        xlabel="x",
        real_rates=None,
        real_rates_label="real intensity",
        real_rates_color="orange",
        setlogy=False,
        title=None,
        figsize=(15, 8),
        return_plot=False,
        bounds=None,
        do_cms_label=False,
        cms_label="Work In Progress",
        is_cms_data=False,
        **plot_config
        ):

    if axs is None:
        fig, axs = plt.subplots(
            2, 1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={
                'height_ratios': [3, 1],
                'hspace': 0
                },
            )
    elif not isinstance(axs, np.ndarray):
        axs = split_axes(axs)

    # Plot the data
    plot(
        h,
        ax=axs[0],
        label=hist_label,
        color='black', histtype='errorbar',
        do_cms_label=do_cms_label, cms_label=cms_label, is_cms_data=is_cms_data,
    )

    axs[0].set_ylabel("Counts")

    # Plot the MCMC samples
    lambda_ = fit_values
    bin_edges = h.axes[0].edges

    if len(lambda_.shape) == 2:
        m, u, l = np.percentile(lambda_, [50, 97.5, 2.5], axis=0)
        axs[0].step(bin_edges, np.concatenate((m, [0])), where="post", label=fit_label, color=fit_color)
        axs[0].fill_between(bin_edges, np.concatenate((u, [0])), np.concatenate((l, [0])), color="gray", alpha=0.5, step="post", label=f"{fit_label} 95% CI")
    else:
        axs[0].step(bin_edges, np.concatenate((lambda_, [0])), where="post", label=fit_label)

    if real_rates is not None:
        hep.histplot((real_rates, bin_edges), ax=axs[0], label=real_rates_label, color=real_rates_color)
    
    # axs[0].set_xlim(*self.bounds)
    if setlogy:
        axs[0].set_yscale('log')

    # Plot the residuals
    values = h.values()
    err = np.sqrt(h.variances())

    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    with np.errstate(divide='ignore', invalid='ignore'):
        if len(lambda_.shape) == 2:
            MSE = np.mean((values - m)**2)
            y_bias = (values - m)/err
            y_bias_up = (u - m)/err
            y_bias_down = (m - l)/err
            axs[1].errorbar(
                bin_centers, y_bias, yerr=[y_bias_up, y_bias_down],
                fmt='o', color='blue', markersize=4, elinewidth=0.8, label=f"{hist_label} - {fit_label}")
        else:
            MSE = np.mean((values - lambda_)**2)
            y_bias = (values - lambda_)/err
            axs[1].errorbar(
                bin_centers, y_bias, yerr=np.zeros_like(y_bias),
                fmt='o', color='blue', markersize=4, elinewidth=0.8, label=f"{hist_label} - {fit_label}")
    
    if real_rates is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            y_real = (values-real_rates)/err
        axs[1].errorbar(
            bin_centers, y_real, yerr=np.zeros_like(y_real),
            color=real_rates_color, label=f"{hist_label} - {real_rates_label}",
            fmt='o', markersize=4, elinewidth=0.8,
        )
    
    axs[1].axhline(0, color='black', linestyle='--')
    axs[1].set_ylabel("Pull")

    y_bias = y_bias[np.isfinite(y_bias)]
    max_res = np.max(np.abs(y_bias)) * 1.5
    axs[1].set_ylim(-max_res, max_res)
    axs[0].set_title(title) 
    axs[1].set_xlabel(xlabel)
    # axs[1].legend()

    # Add MSE to legend
    axs[0].plot([], [], ' ', label=f"MSE: {MSE:.2f}")
    axs[0].legend()

    if bounds is not None:
        axs[0].set_xlim(*bounds)
        axs[1].set_xlim(*bounds)

    if return_plot:
        return fig, axs


def plot_signal_parameter_fit(
    df,
    reg,
    fitted_pars,
    logy=False,
    legend=False,
    colors=colors,
    marker_styles=marker_styles,
    line_styles=line_styles,
    axs=None,
    x1_name = "M_BKK",
    x2_name = "Mass_Ratio",
    n_x1_points = 6,
    n_x2_points = 6,
    ):

    x1_points = df[x1_name].unique()
    x2_points = df[x2_name].unique()

    x1_grid = np.linspace(x1_points.min(), x1_points.max(), 100)
    x2_grid = np.linspace(x2_points.min(), x2_points.max(), 100)

    # Choose n points to plot
    if len(x1_points) > n_x1_points:
        x1_points = np.random.choice(x1_points, n_x1_points, replace=False)
    if len(x2_points) > n_x2_points:
        x2_points = np.random.choice(x2_points, n_x2_points, replace=False)

    x1_points.sort()
    x2_points.sort()

    if axs is None:
        fig, axs = plt.subplots(len(fitted_pars), 2, figsize=(15, 6*len(fitted_pars)), gridspec_kw={'wspace': 0.3})
    for ipar, par in enumerate(fitted_pars):
        if len(fitted_pars) == 1:
            ax = axs
        else:
            ax = axs[ipar]
        for i, x2 in enumerate(x2_points):
            mask = df[x2_name] == x2
            ax[0].errorbar(
                df[x1_name][mask], df[par][mask],
                yerr=df[f"{par}_error"][mask],
                fmt=marker_styles[i],
                label=f'{x2_name}={x2}',
                color=colors[i%len(colors)],)

            predictions = []
            prediction_errors = []
            for x1 in x1_grid:
                prediction = reg.predict(x1, x2)[par]
                predictions.append(prediction)
                if hasattr(reg, 'predict_error'):
                    prediction_error = reg.predict_error(x1, x2)[par]
                    prediction_errors.append(prediction_error)
            # color = cmap(norm(x2))
            ax[0].plot(
                x1_grid, predictions,
                color=colors[i%len(colors)])
            if hasattr(reg, 'predict_error'):
                ax[0].fill_between(
                    x1_grid,
                    np.array(predictions) - np.array(prediction_errors),
                    np.array(predictions) + np.array(prediction_errors),
                    color=colors[i%len(colors)],
                    alpha=0.2,
                )
        
        ax[0].set_xlabel(x1_name.replace("_", " "))
        ax[0].set_ylabel(par)
        if logy:
            ax[0].set_yscale("log")
        if legend:
            ax[0].legend(fontsize=12)

        for i, x1 in enumerate(x1_points):
            mask = df[x1_name] == x1
            ax[1].errorbar(
                df[x2_name][mask], df[par][mask],
                yerr=df[f"{par}_error"][mask],
                fmt=marker_styles[i],
                label=f'{x1_name}={x1}',
                color=colors[i%len(colors)],)
            
            predictions = []
            prediction_errors = []
            for x2 in x2_grid:
                prediction = reg.predict(x1, x2)[par]
                predictions.append(prediction)
                if hasattr(reg, 'predict_error'):
                    prediction_error = reg.predict_error(x1, x2)[par]
                    prediction_errors.append(prediction_error)
            ax[1].plot(
                x2_grid, predictions,
                color=colors[i%len(colors)])
            
            if hasattr(reg, 'predict_error'):
                ax[1].fill_between(
                    x2_grid,
                    np.array(predictions) - np.array(prediction_errors),
                    np.array(predictions) + np.array(prediction_errors),
                    color=colors[i%len(colors)],
                    alpha=0.2,
                )
        
        ax[1].set_xlabel(x2_name.replace("_", " "))
        ax[1].set_ylabel(par)
        if logy:
            ax[1].set_yscale("log")
        if legend:
            ax[1].legend(fontsize=10)

    plt.show()


def plot_signal_parameter_fit_variation(
    M_BKK, Mass_Ratio, dfs, regs, fitted_pars,
    ):
    
    BKK_Mass_points = dfs['nominal'].M_BKK.unique()
    Mass_Ratio_points = dfs['nominal'].Mass_Ratio.unique()
    BKK_Mass_points.sort()
    Mass_Ratio_points.sort()

    BKK_Masses = np.linspace(BKK_Mass_points.min(), BKK_Mass_points.max(), 100)
    Mass_Ratios = np.linspace(Mass_Ratio_points.min(), Mass_Ratio_points.max(), 100)

    fig, axs = plt.subplots(len(fitted_pars), 2, figsize=(15, 6*len(fitted_pars)), gridspec_kw={'wspace': 0.3})

    for i, par in enumerate(fitted_pars):
        ax = axs[i]
        
        colors = ['black', 'blue', 'orange']
        for j, sys_name in enumerate(dfs.keys()):
            df = dfs[sys_name]
            reg = regs[sys_name]
            mask = df.Mass_Ratio == Mass_Ratio
            ax[0].errorbar( df[mask].M_BKK, df[mask][par], yerr=df[mask][f"{par}_error"], fmt='o', color=colors[j])
            ax[0].plot(BKK_Masses, [reg.predict(b, Mass_Ratio)[par] for b in BKK_Masses], label=sys_name.split("_")[-1], color=colors[j])
            if hasattr(regs[sys_name], 'predict_error'):
                ax[0].fill_between(
                    BKK_Masses,
                    [reg.predict(b, Mass_Ratio)[par] - reg.predict_error(b, Mass_Ratio)[par] for b in BKK_Masses],
                    [reg.predict(b, Mass_Ratio)[par] + reg.predict_error(b, Mass_Ratio)[par] for b in BKK_Masses],
                    color=colors[j],
                    alpha=0.2,
                )
        
            mask = df.M_BKK == M_BKK
            ax[1].errorbar( df[mask].Mass_Ratio, df[mask][par], yerr=df[mask][f"{par}_error"], fmt='o', color=colors[j])
            ax[1].plot(Mass_Ratios, [reg.predict(M_BKK, m)[par] for m in Mass_Ratios], label=sys_name.split("_")[-1], color=colors[j])
            if hasattr(regs[sys_name], 'predict_error'):
                ax[1].fill_between(
                    Mass_Ratios,
                    [reg.predict(M_BKK, m)[par] - reg.predict_error(M_BKK, m)[par] for m in Mass_Ratios],
                    [reg.predict(M_BKK, m)[par] + reg.predict_error(M_BKK, m)[par] for m in Mass_Ratios],
                    color=colors[j],
                    alpha=0.2,
                )

        # ax[0].set_xlabel('M_BKK')
        ax[0].set_ylabel(par)
        ax[0].legend()

        # ax[1].set_xlabel('Mass_Ratio')
        ax[1].set_ylabel(par)
        ax[1].legend()

    axs[-1, 0].set_xlabel('M_BKK')
    axs[-1, 1].set_xlabel('Mass_Ratio')
    plt.show()

def plot_1D_marginals(
    model, data,
    tm_var, a_var,
    nbins=32,
    ):
    import ROOT

    # Define the canvas and divide into two columns
    can = ROOT.TCanvas(random_string(), random_string(), 1200, 800)
    can.Divide(2, 1)

    # Create subpads for each column
    def create_subpads(canvas, column_index, main_height=0.7, pull_height=0.3):
        canvas.cd(column_index)
        main_pad = ROOT.TPad(f"main_pad{column_index}", f"Main Pad {column_index}", 0, pull_height, 1, 1)
        pull_pad = ROOT.TPad(f"pull_pad{column_index}", f"Pull Pad {column_index}", 0, 0, 1, pull_height)
        main_pad.SetBottomMargin(0)
        pull_pad.SetTopMargin(0)
        pull_pad.SetBottomMargin(0.3)
        main_pad.Draw()
        pull_pad.Draw()
        return main_pad, pull_pad

    main_pad1, pull_pad1 = create_subpads(can, 1)
    main_pad2, pull_pad2 = create_subpads(can, 2)

    # Plot the marginal distributions on the main pads
    main_pad1.cd()
    plot1 = tm_var.frame(ROOT.RooFit.Bins(nbins))
    plot1.GetYaxis().SetTitleOffset(1.45)
    plot1.SetTitle("")
    data.plotOn(plot1)
    model.plotOn(plot1, ROOT.RooFit.LineColor(2))
    plot1.Draw()

    main_pad2.cd()
    plot2 = a_var.frame(ROOT.RooFit.Bins(nbins))
    plot2.GetYaxis().SetTitleOffset(1.45)
    plot2.SetTitle("")
    data.plotOn(plot2)
    model.plotOn(plot2, ROOT.RooFit.LineColor(2))
    plot2.Draw()

    # Helper function to draw pull plots
    lines = []
    boxes = []
    def draw_pull_plot(pull_pad, frame, var, xlabel):
        pull_pad.cd()
        pull_hist = frame.pullHist()
        # pull_values = [pull_hist.GetBinContent(i) for i in range(1, pull_hist.GetX().GetSize() + 1)]
        pull_frame = var.frame()
        pull_frame.addPlotable(pull_hist, "P")

        # Style adjustments
        pull_frame.SetTitle("")  # Remove title
        pull_frame.GetYaxis().SetTitle("Pull")
        pull_frame.GetYaxis().SetTitleSize(0.1)
        pull_frame.GetYaxis().SetTitleOffset(0.5)
        pull_frame.GetYaxis().SetLabelSize(0.08)
        pull_frame.GetXaxis().SetTitleSize(0.1)
        pull_frame.GetXaxis().SetTitle(xlabel)
        pull_frame.GetXaxis().SetLabelSize(0.08)
        pull_frame.GetYaxis().SetRangeUser(-5, 5)

        pull_frame.Draw()

        # Draw a dotted line at y=0
        x_min = pull_frame.GetXaxis().GetXmin()
        x_max = pull_frame.GetXaxis().GetXmax()
        zero_line = ROOT.TLine(x_min, 0, x_max, 0)
        zero_line.SetLineStyle(2)  # Dotted line
        zero_line.SetLineColor(ROOT.kBlack)
        zero_line.Draw()
        lines.append(zero_line)  # Store zero line to prevent premature cleanup


    # Draw pull plots on the pull pads
    draw_pull_plot(pull_pad1, plot1, tm_var, 'Triphoton Mass [GeV]')
    draw_pull_plot(pull_pad2, plot2, a_var, 'Alpha')

    # Update and display the canvas
    can.Update()
    can.Draw()

    return can, lines

def plot_2D_pull(model, data, tm_var, a_var, tm_spec, a_spec):
    import ROOT

    h_data = ROOT.TH2F("h_data", "Data Histogram;Triphoton Mass [GeV];Alpha", *tm_spec, *a_spec)
    data.fillHistogram(h_data, ROOT.RooArgList(tm_var, a_var))

    h_model = ROOT.TH2F("h_model", "Model Histogram;Triphoton Mass [GeV];Alpha", *tm_spec, *a_spec)
    model.fillHistogram(h_model, ROOT.RooArgList(tm_var, a_var))
    h_model.Scale(h_data.Integral() / h_model.Integral())

    # Create the 2D pull plot
    h_pull = ROOT.TH2F("h_pull", "2D Pull Plot;Triphoton Mass [GeV];Alpha", *tm_spec, *a_spec)
    for ix in range(1, h_pull.GetNbinsX() + 1):
        for iy in range(1, h_pull.GetNbinsY() + 1):
            data_val = h_data.GetBinContent(ix, iy)
            model_val = h_model.GetBinContent(ix, iy)
            data_err = h_data.GetBinError(ix, iy)

            # Calculate the pull
            if data_err > 0:
                pull = (data_val - model_val) / data_err
            else:
                pull = 0
            h_pull.SetBinContent(ix, iy, pull)

    # Plot the 2D pull plot
    canvas = ROOT.TCanvas(random_string(), random_string(), 800, 600)
    h_pull.Draw("COLZ")

    # Remove stats box
    h_pull.SetStats(0)
    max_z = h_pull.GetMaximum()*2
    h_pull.GetZaxis().SetRangeUser(-max_z, max_z)
    h_pull.GetYaxis().SetTitleOffset(1.4)

    canvas.Update()
    canvas.Draw()

    return canvas, h_pull

def plot_2D_data_minus_model(model, data, tm_var, a_var, tm_spec, a_spec):
    import ROOT

    h_data = ROOT.TH2F("h_data", "Data Histogram;Triphoton Mass [GeV];Alpha", *tm_spec, *a_spec)
    data.fillHistogram(h_data, ROOT.RooArgList(tm_var, a_var))

    h_model = ROOT.TH2F("h_model", "Model Histogram;Triphoton Mass [GeV];Alpha", *tm_spec, *a_spec)
    model.fillHistogram(h_model, ROOT.RooArgList(tm_var, a_var))
    h_model.Scale(h_data.Integral() / h_model.Integral())

    # Create the 2D pull plot
    h_pull = ROOT.TH2F("h_pull", "2D (Data - Model);Triphoton Mass [GeV];Alpha", *tm_spec, *a_spec)
    for ix in range(1, h_pull.GetNbinsX() + 1):
        for iy in range(1, h_pull.GetNbinsY() + 1):
            data_val = h_data.GetBinContent(ix, iy)
            model_val = h_model.GetBinContent(ix, iy)
            data_err = h_data.GetBinError(ix, iy)

            # Calculate the pull
            pull = data_val - model_val
            h_pull.SetBinContent(ix, iy, pull)

    # Plot the 2D pull plot
    canvas = ROOT.TCanvas(random_string(), random_string(), 800, 600)
    h_pull.Draw("COLZ")

    # Remove stats box
    h_pull.SetStats(0)
    # h_pull.GetZaxis().SetRangeUser(-5, 5)
    h_pull.GetYaxis().SetTitleOffset(1.4)

    canvas.Update()
    canvas.Draw()

    return canvas, h_pull