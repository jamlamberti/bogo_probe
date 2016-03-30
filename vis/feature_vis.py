"""A simple tool for generating heatmaps"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # nopep8


def feature_vis(data, out_file='out.png'):
    """Visualize a vector of features"""
    # Should be sorted based on class
    fig, ax = plt.subplots(figsize=(10, 10))

    heatmap = ax.pcolor(data)
    fig.colorbar(heatmap)
    fig.tight_layout()

    # save and clear
    plt.savefig(out_file)
    plt.clf()
