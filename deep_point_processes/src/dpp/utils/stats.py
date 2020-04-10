import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import kstest, ks_2samp
from scipy.stats import norm, pareto, lognorm, gamma, expon


def best_fit_scaled(x, plot=False,
                    dist_to_test=None,
                    number_of_bins=100, log_scaled=False, the_label="something", color="red", symbol="o"):
    if dist_to_test is None:
        dist_to_test = ["norm", "pareto", "lognorm", "gamma", "expon", "weibull_min", "weibull_max"]
    x_min = min(x)
    x_max = max(x)
    dX = (x_max - x_min) * 0.01

    support = np.arange(x_min, x_max, dX)
    best_fit = []
    error = []

    if "pareto" in dist_to_test:
        try:
            dist_param = pareto.fit(x)
            my_dist = pareto(*dist_param)
            kt, p_value = kstest(x, "pareto", dist_param)
            best_fit.append({"distribution": "pareto", "ktest": kt, "pvalue": p_value, "parameters": dist_param})
            if plot:
                Y = my_dist.pdf(support)
                plt.plot(support, Y, linewidth=2.0)
                if log_scaled:
                    hist, bins = np.histogram(x, bins=number_of_bins, normed=True)
                    plt.plot(bins[:-1], hist, symbol, color=color, label=the_label)
                # stuff = plt.hist(X,bins=numberOfBins,normed=True)
        except Exception as e:
            error.append(("pareto_err", e))

    if "lognorm" in dist_to_test:
        try:
            dist_param = lognorm.fit(x)
            my_dist = lognorm(*dist_param)
            kt, p_value = kstest(x, "lognorm", dist_param)
            best_fit.append({"distribution": "lognorm", "ktest": kt, "pvalue": p_value, "parameters": dist_param})
            if plot:
                Y = my_dist.pdf(support)
                plt.plot(support, Y, color=color, linewidth=2.0)
                if log_scaled:
                    hist, bins = np.histogram(x, bins=number_of_bins, normed=True)
                    plt.plot(bins[:-1], hist, symbol, color=color, label=the_label)

        except Exception as e:
            error.append(("lognorm_err", e))

    # FINISH PLOT
    if plot:
        if log_scaled:
            plt.yscale("log")
            plt.xscale("log")

        plt.legend(loc="best")
        plt.show()

    return best_fit, error


def best_fit(x, plot=False, log_scaled=False, dist_to_test=None, number_of_bins=100, ax=None):
    """

    Version of August 2015
    X: data

    return
    """

    if dist_to_test is None:
        dist_to_test = ["norm", "pareto", "lognorm", "gamma", "expon", "weibull_min", "weibull_max"]
    x_min = min(x)
    x_max = max(x)
    dX = (x_max - x_min) * 0.01

    support = np.arange(x_min, x_max, dX)
    best_fit = []
    error = []
    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        hist = ax.hist(x, bins=number_of_bins)
    for dist_name in dist_to_test:
        try:
            dist = getattr(stats, dist_name)
            dist_params = dist.fit(x)
            my_dist = dist(*dist_params)
            ks, p_value = kstest(x, dist_name, dist_params)
            best_fit.append({"distribution": dist_name, "ktest": ks, "pvalue": p_value, "parameters": dist_params})
            if plot:
                y = my_dist.pdf(support)
                ax.plot(support, y, label=dist_name.replace('_', '\-'), linewidth=2.0)
        except:
            error.append((f"{dist_name}", sys.exc_info()))

    # FINISH PLOT
    if plot:
        if log_scaled:
            ax.set_yscale('log')
        plt.legend(loc="best")
        return best_fit, error, ax
    else:
        return best_fit, error


def real_best_fit(x, plot=False, dist_to_test=None, number_of_bins=100):
    """
    """
    if dist_to_test is None:
        dist_to_test = ["norm", "pareto", "expon", "lognorm", "gamma", "weibull_min", "weibull_max"]
    bF = best_fit(x, plot, False, dist_to_test, number_of_bins)[0]
    a = [(b["ktest"], b["distribution"]) for b in bF]
    a.sort()
    return a[0][1]


def loglog_distribution_plots(data, service_simulated=None, plot_dist=None, x_lim=None, y_lim=None, ax=None,
                              title=None):
    if plot_dist is None:
        plot_dist = ["norm", "pareto", "lognorm", "gamma", "expon", "weibull_min", "weibull_max"]
    if x_lim is None:
        x_min = min(data)
        x_max = max(data)
    else:
        x_min, x_max = x_lim
    d_x = (x_max - x_min) * 0.001
    support = np.arange(x_min, x_max, d_x)

    if "pareto" in plot_dist:
        dist_param = pareto.fit(data)
        my_dist = pareto(*dist_param)
        y_pareto = my_dist.pdf(support)
        print("Pareto: " + str(ks_2samp(data, y_pareto)[0]))

    if "norm" in plot_dist:
        dist_param = norm.fit(data)
        my_dist = norm(*dist_param)
        y_norm = my_dist.pdf(support)
        print("Norm: " + str(ks_2samp(data, y_norm)[0]))

    if "expon" in plot_dist:
        dist_param = expon.fit(data)
        my_dist = expon(*dist_param)
        y_exp = my_dist.pdf(support)
        kt, p_value = kstest(data, "lognorm", dist_param)
        print("Exp: " + str(ks_2samp(data, y_exp)[0]), kt, p_value)

    if "lognorm" in plot_dist:
        dist_param = lognorm.fit(data)
        my_dist = lognorm(*dist_param)
        y_lognorm = my_dist.pdf(support)
        print("LogNorm: " + str(ks_2samp(data, y_lognorm)[0]))

    if "gamma" in plot_dist:
        dist_param = gamma.fit(data)
        my_dist = gamma(*dist_param)
        y_gamma = my_dist.pdf(support)
        print("Gamma: " + str(ks_2samp(data, y_gamma)[0]))

    if "lognorm" in plot_dist:
        if ax is None:
            plt.plot(support, y_lognorm, label="ln", linewidth=6.0)
        else:
            ax.plot(support, y_lognorm, linewidth=6.0)
    if "gamma" in plot_dist:
        if ax is None:
            plt.plot(support, y_gamma, label="gamma", linewidth=6.0)
        else:
            ax.plot(support, y_gamma, linewidth=6.0)

    if "norm" in plot_dist:
        if ax is None:
            plt.plot(support, y_norm, linewidth=6.0)
        else:
            ax.plot(support, y_norm, linewidth=6.0)
    if "pareto" in plot_dist:
        if ax is None:
            plt.plot(support, y_pareto, label="pareto", linewidth=6.0)
        else:
            ax.plot(support, y_pareto, linewidth=6.0)
    if "expon" in plot_dist:
        if ax is None:
            plt.plot(support, y_exp, linewidth=6.0)
        else:
            ax.plot(support, y_exp, linewidth=6.0)

    if x_lim is not None:
        if ax is None:
            plt.xlim(x_lim)
        else:
            ax.set_xlim(x_lim)
    if y_lim is not None:
        if ax is None:
            plt.ylim(y_lim)
        else:
            ax.set_ylim(y_lim)

    if ax is None:
        plt.yscale("log")
        plt.xscale("log")
    else:
        ax.set_yscale("log")
        ax.set_xscale("log")

    if ax is None:
        plt.ylabel("Frequency")
        plt.xlabel("Service Time (s)")
    else:
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Service Time (s)")
    count, bins = np.histogram(data, bins=100, normed=True)
    if ax is None:
        plt.plot(bins[1:], count, "o", label="empirical", markersize=10)
    else:
        ax.plot(bins[1:], count, "o", label="empirical", markersize=10)
    if service_simulated is not None:
        count, _ = np.histogram(service_simulated, bins=bins, normed=True)
        if ax is None:
            plt.plot(bins[1:], count, "D", label="model", markersize=10)
        else:
            ax.plot(bins[1:], count, "D", label="model", markersize=10)
    if ax is None:
        plt.grid(True)
        plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                   mode="expand", borderaxespad=0, ncol=2)
    else:
        ax.grid(True)
        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                  mode="expand", borderaxespad=0, ncol=2)
    if title is not None:
        if ax is None:
            plt.title(title)
        else:
            ax.set_title(title)
    return ax
