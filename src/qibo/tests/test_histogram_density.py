from histogram_density import plot_density_hist
from qibo import gates, models


def test_plot_density_hist():
    c = models.Circuit(2)
    c.add(gates.H(0))
    c.add(gates.CNOT(0, 1))

    plot_density_hist(c)


test_plot_density_hist()
