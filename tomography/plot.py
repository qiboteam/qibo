import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot(rho_theory, rho_linear, rho_fit, fidelity, success, width=0.8, depth=0.8):
    _x, _y = np.meshgrid(np.arange(4), np.arange(4))
    x, y = _x.ravel(), _y.ravel()

    top_real_th = rho_theory.real.ravel()
    top_imag_th = rho_theory.imag.ravel()

    top_real_exp = rho_linear.real.ravel()
    top_imag_exp = rho_linear.imag.ravel()

    top_real_fit = rho_fit.real.ravel()
    top_imag_fit = rho_fit.imag.ravel()

    plt.style.use('default')
    ticks = [0.5, 1.5, 2.5, 3.5]
    tick_labels = ['|00>','|01>', '|10>', '|11>']
    fig = plt.figure(figsize=(12, 16))
    fig.suptitle("Fidelity: {:f}".format(fidelity))
    ax1 = fig.add_subplot(231, projection='3d')
    ax2 = fig.add_subplot(234, projection='3d')
    ax3 = fig.add_subplot(232, projection='3d')
    ax4 = fig.add_subplot(235, projection='3d')
    ax5 = fig.add_subplot(233, projection='3d')
    ax6 = fig.add_subplot(236, projection='3d')

    bottom = np.zeros_like(top_real_th)
    ax1.bar3d(x, y, bottom, width, depth, top_real_th, shade=True, color='C0')
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(tick_labels)
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(tick_labels)
    ax1.set_zlim3d(-1, 1)
    ax1.set_title('Real part, Theory')

    ax2.bar3d(x, y, bottom, width, depth, top_imag_th, shade=True, color='C0')
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(tick_labels)
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(tick_labels)
    ax2.set_zlim3d(-1, 1)
    ax2.set_title('Imaginary part, Theory')

    ax3.bar3d(x, y, bottom, width, depth, top_real_exp, shade=True, color='C1')
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(tick_labels)
    ax3.set_yticks(ticks)
    ax3.set_yticklabels(tick_labels)
    ax3.set_zlim3d(-1, 1)
    ax3.set_title('Real part, Linear')

    ax4.bar3d(x, y, bottom, width, depth, top_imag_exp, shade=True, color='C1')
    ax4.set_xticks(ticks)
    ax4.set_xticklabels(tick_labels)
    ax4.set_yticks(ticks)
    ax4.set_yticklabels(tick_labels)
    ax4.set_zlim3d(-1, 1)
    ax4.set_title('Imaginary part, Linear')

    ax5.bar3d(x, y, bottom, width, depth, top_real_fit, shade=True, color='C2')
    ax5.set_xticks(ticks)
    ax5.set_xticklabels(tick_labels)
    ax5.set_yticks(ticks)
    ax5.set_yticklabels(tick_labels)
    ax5.set_zlim3d(-1, 1)
    ax5.set_title('Real part, MLE_{}'.format(success))

    ax6.bar3d(x, y, bottom, width, depth, top_imag_fit, shade=True, color='C2')
    ax6.set_xticks(ticks)
    ax6.set_xticklabels(tick_labels)
    ax6.set_yticks(ticks)
    ax6.set_yticklabels(tick_labels)
    ax6.set_zlim3d(-1, 1)
    ax6.set_title('Imaginary part, MLE_{}'.format(success))

    plt.show()
