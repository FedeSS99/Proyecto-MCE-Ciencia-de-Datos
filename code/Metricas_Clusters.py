from sklearn.metrics import rand_score, mutual_info_score
from pandas import DataFrame
from mplcatppuccin.palette import load_color
from mplcatppuccin.colormaps import get_colormap_from_list

def GetMetricsClustering(labels_pred, labels_true, name):
    rand = rand_score(labels_true=labels_true, labels_pred=labels_pred)
    NMI = mutual_info_score(labels_true=labels_true, labels_pred=labels_pred)

    metrics_cluster = DataFrame({"Rand": rand, "NMI": NMI}, index = [name])
    return metrics_cluster

group_colors = {k:load_color("mocha", color_str) for k, color_str in enumerate(["red", "green", "blue"], 1)}
mocha_cmap = get_colormap_from_list("mocha", ["rosewater", "pink", "mauve"])
