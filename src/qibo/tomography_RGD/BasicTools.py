#
#   some tools
#   1. generate all labels
#   2. plot the results
#

# --------------------------------- #
#   to generate all symbols         #
# --------------------------------- #
from itertools import product

import matplotlib.pyplot as plt

def Generate_All_labels(nqubits, symbols=["I", "X", "Y", "Z"]):
    all_labels = ["".join(elem) for elem in product(symbols, repeat=nqubits)]
    return all_labels

def Plt_Err_Time(worker):
    """plot the Error w.r.t. optimization run time

    Args:
        worker (class): the optimization class instance
    """

    Target_Err_Xk = worker.Target_Err_Xk
    step_Time = worker.step_Time

    RunT = []
    Ttot = 0
    for ti in range(-1, len(step_Time) - 1):
        Each_Step = step_Time[ti]
        Ttot += Each_Step
        RunT.append(Ttot)

    mk_list = ["+", "^", "o", "x", ">", "<", 2, 3]
    ln_list = ["-", "-.", "--", "--", ":", "-"]

    fig, axNow = plt.subplots(1, 1, figsize=(8, 6))

    info = "{} qubits with sampling {} labels".format(worker.n, worker.num_labels)
    axNow.plot(
        RunT,
        Target_Err_Xk,
        marker=mk_list[0],
        linestyle=ln_list[0],
        label="{}".format(info),
    )

    axNow.set_xlabel("  Run time (sec)", fontsize=14)
    axNow.set_ylabel(r"$\left\Vert X_k -\rho \right\Vert_F$", fontsize=14)

    axNow.set_title("Error w.r.t. run time", y=1.0, fontsize=14)

    plt.legend(loc="upper left")
    plt.show()
