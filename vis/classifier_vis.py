"""Visualize states of a classifier"""
import scipy.stats
import matplotlib
matplotlib.use('Agg')  # nopep8
import matplotlib.pyplot as plt


def classifier_vis(truth, expected, thresh=0.02, out_file='out.png'):
    """truth is the black-box model, expected is the surrogate"""
    # slope, intercept, r_value, p_value, std_err
    # = scipy.stats.linregress(truth, expected)
    slope, intercept, r_value, _, _ = scipy.stats.linregress(truth, expected)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([0.5, 0.5], [0, 1], 'r')
    ax.plot([0, 1], [0.5, 0.5], 'r')
    ax.plot([0, 1], [0.5 - thresh, 0.5 - thresh], 'g--')
    ax.plot([0, 1], [0.5 + thresh, 0.5 + thresh], 'g--')
    ax.plot([0.5 - thresh, 0.5 - thresh], [0, 1], 'g--')
    ax.plot([0.5 + thresh, 0.5 + thresh], [0, 1], 'g--')
    ax.scatter(truth, expected)
    ax.plot([0, 1], [intercept, slope + intercept])
    ax.text(0.1, 0.95, '$r^2$ = %0.4f' % r_value**2)
    ax.set_xlabel('$p$')
    ax.set_ylabel('$\\mathrm{E}[x]$')
    plt.tight_layout()
    plt.savefig(out_file)
    plt.clf()