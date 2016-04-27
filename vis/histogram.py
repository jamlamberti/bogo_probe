"""Histogram generator to show ham and spam distribution"""
import matplotlib.pyplot as plt


def histogram_analysis(x_range, xbins, out_file='hist.png'):
    """Visualize ham and spam graphical placement"""
    plt.hist(x_range, bins=xbins, color='blue')
    # Add labels later

    # Save and clear
    plt.savefig(out_file)
    plt.clf()
