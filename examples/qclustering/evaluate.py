import argparse

import util as u


def evaluate_qkmedians(
    centroids_file,
    data_qcd_file,
    data_signal_file,
    k=2,
    test_size=10000,
    title="Anomaly detection results",
    results_dir=None,
    data_dir=None,
    save_dir_roc=None,
    xlabel="TPR",
    ylabel="1/FPR",
):
    """Evaluation of quantum k-medians.

    Parameters
    ----------
    centroids_file : str
        Name of the file for saved centroids coordinates.
    data_qcd_file : str
        Name of the file for test QCD dataset.
    data_signal_file : str
        Name of the file for test signal dataset.
    k : int
        Number of classes in quantum k-medians.
    test_size : int
        Number of test samples.
    title : str
        Title of ROC curve plot.
    results_dir : str
        Path to file with saved centroids.
    data_dir : str
        Path to file with test datasets.
    save_dir_roc : str
        Path to directory for saving ROC plot.
    xlabel : str
        Name of x-axis in ROC plot.
    ylabel : str
        Name of y-axis in ROC plot.
    """

    # calculate anomaly detection scores
    loss = u.calc_AD_scores(
        centroids_file,
        data_qcd_file,
        data_signal_file,
        k=k,
        test_size=test_size,
        results_dir=results_dir,
        data_dir=data_dir,
    )
    # plot roc curve
    u.plot_ROCs_compare(
        loss,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        legend_loc="best",
        save_dir=save_dir_roc,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="read arguments for qkmedians evaluation"
    )
    parser.add_argument(
        "--centroids_file",
        dest="centroids_file",
        type=str,
        help="name of the file for saved centroids coordinates",
    )
    parser.add_argument(
        "--data_qcd_file",
        dest="data_qcd_file",
        type=str,
        help="name of the file for test QCD dataset",
    )
    parser.add_argument(
        "--data_signal_file",
        dest="data_signal_file",
        type=str,
        help="name of the file for test signal dataset",
    )
    parser.add_argument("--k", dest="k", type=int, default=2, help="number of classes")
    parser.add_argument(
        "--test_size", dest="test_size", type=int, default=10000, help="test size"
    )
    parser.add_argument(
        "--title",
        dest="title",
        type=str,
        default="Anomaly detection results",
        help="title of ROC curve plot",
    )
    parser.add_argument(
        "--results_dir",
        dest="results_dir",
        type=str,
        help="path to file with saved centroids",
    )
    parser.add_argument(
        "--data_dir", dest="data_dir", type=str, help="path to file with test datasets"
    )
    parser.add_argument(
        "--save_dir_roc",
        dest="save_dir_roc",
        type=str,
        help="path to directory for saving ROC plot",
    )
    parser.add_argument(
        "--xlabel",
        dest="xlabel",
        type=str,
        default="TPR",
        help="name of x-axis in ROC plot",
    )
    parser.add_argument(
        "--ylabel",
        dest="ylabel",
        type=str,
        default="1/FPR",
        help="name of y-axis in ROC plot",
    )

    args = parser.parse_args()

    evaluate_qkmedians(
        args.centroids_file,
        args.data_qcd_file,
        args.data_signal_file,
        args.k,
        args.test_size,
        args.title,
        args.results_dir,
        args.data_dir,
        args.save_dir_roc,
        args.xlabel,
        args.ylabel,
    )
