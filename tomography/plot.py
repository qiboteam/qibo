import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot():
    plt.style.use('default')
    ticks=[0.5, 1.5, 2.5, 3.5]
    tick_lable=['|00>','|01>', '|10>', '|11>']
    fig = plt.figure(figsize=(12, 16))
    fig.suptitle("Fidelity: {:f}".format(fidelity))
    ax1 = fig.add_subplot(231, projection='3d')
    ax2 = fig.add_subplot(234, projection='3d')
    ax3 = fig.add_subplot(232, projection='3d')
    ax4 = fig.add_subplot(235, projection='3d')
    ax5 = fig.add_subplot(233, projection='3d')
    ax6 = fig.add_subplot(236, projection='3d')

    _x = np.arange(4)
    _y = np.arange(4)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    p_rho_thr = copy.copy(utils.matrices.rho_th_plot(index))
    p_rho_lin = copy.copy(rho_linear)
    p_rho_fit = copy.copy(rho_fit)

    top_real_th = np.real(p_rho_thr).flatten()
    top_imag_th= np.imag(p_rho_thr).flatten()

    top_real_exp = np.real(p_rho_lin).flatten()
    top_imag_exp= np.imag(p_rho_lin).flatten()

    top_real_fit = np.real(p_rho_fit).flatten()
    top_imag_fit= np.imag(p_rho_fit).flatten()

    bottom = np.zeros_like(top_real_th)
    width = depth = 0.8

    ax1.bar3d(x,y,  bottom,width, depth, top_real_th, shade=True,color='C0')
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(tick_lable)
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(tick_lable)
    ax1.set_zlim3d(-1, 1)
    ax1.set_title('Real part, Theory')

    ax2.bar3d(x,y,  bottom,width, depth, top_imag_th, shade=True,color='C0')
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(tick_lable)
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(tick_lable)
    ax2.set_zlim3d(-1, 1)
    ax2.set_title('Imaginary part, Theory')

    ax3.bar3d(x,y,  bottom,width, depth, top_real_exp, shade=True,color='C1')
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(tick_lable)
    ax3.set_yticks(ticks)
    ax3.set_yticklabels(tick_lable)
    ax3.set_zlim3d(-1, 1)
    ax3.set_title('Real part, Linear')

    ax4.bar3d(x,y,  bottom,width, depth, top_imag_exp, shade=True,color='C1')
    ax4.set_xticks(ticks)
    ax4.set_xticklabels(tick_lable)
    ax4.set_yticks(ticks)
    ax4.set_yticklabels(tick_lable)
    ax4.set_zlim3d(-1, 1)
    ax4.set_title('Imaginary part, Linear')

    ax5.bar3d(x,y,  bottom,width, depth, top_real_fit, shade=True,color='C2')
    ax5.set_xticks(ticks)
    ax5.set_xticklabels(tick_lable)
    ax5.set_yticks(ticks)
    ax5.set_yticklabels(tick_lable)
    ax5.set_zlim3d(-1, 1)
    ax5.set_title('Real part, MLE_{}'.format(res.success))

    ax6.bar3d(x,y,  bottom,width, depth, top_imag_fit, shade=True,color='C2')
    ax6.set_xticks(ticks)
    ax6.set_xticklabels(tick_lable)
    ax6.set_yticks(ticks)
    ax6.set_yticklabels(tick_lable)
    ax6.set_zlim3d(-1, 1)
    ax6.set_title('Imaginary part, MLE_{}'.format(res.success))

    plt.show()
