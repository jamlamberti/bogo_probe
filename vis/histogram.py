"""Histogram generator to show ham and spam distribution"""
import numpy as np
import matplotlib.pyplot as plt

def histogram_analysis(xrange, xbins, out_file='hist.png'):
    """Visualize ham and spam graphical placement"""
    plt.hist(xrange, bins=xbins, color='blue')
    #Add labels later
    
    #Save and clear
    plt.savefig(out_file)
    plt.clf()