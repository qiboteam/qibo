import os
import argparse
import h5py
import json
import matplotlib.pyplot as plt
import matplotlib
import shutil
import seaborn as sns


matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["font.size"] = 18
DEFAULT_DIR = "/home/stavros/Documents/qibo-logs"

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default=DEFAULT_DIR, type=str)
parser.add_argument("--name", default=None, type=str)
parser.add_argument("--format", default="png", type=str)


def plot_json(data, save_dir):
    cp = iter(sns.color_palette())
    marker = iter(5 * ["^", "d", "v", "s", "o"])

    plt.figure(figsize=(7, 4))
    for label, log in data.items():
        plt.semilogy(log["nqubits"], log["simulation_time"], color=next(cp), marker=next(marker),
                     linestyle="--", markersize=7, label=label)

    plt.legend(bbox_to_anchor=(1.0, 1.0))
    #plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.xlabel("Number of Qubits")
    plt.ylabel("Simulation Time (sec)")
    plt.savefig(save_dir, bbox_inches="tight")


def main(directory: str, name: str, format: str = "png"):
    def read_file(filename):
        file = h5py.File(os.path.join(directory, "{}.h5".format(filename)), "r")
        d = {k: file[k][()] for k in file.keys()}
        file.close()
        return d

    plot_dir = os.path.join(directory, "plots")
    shutil.rmtree(plot_dir)
    os.mkdir(plot_dir)

    json_dir = os.path.join(directory, "jsons")
    for filename in os.listdir(json_dir):
        filedir = os.path.join(json_dir, filename)
        with open(filedir, "r") as file:
            to_plot = json.load(file)
        data = {k: read_file(v) for k, v in to_plot.items()}

        name = filename.split(".")[0]
        save_dir = os.path.join(plot_dir, ".".join([name, format]))

        plot_json(data, save_dir)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
