from qibo import Circuit
from qibo import gates
from qibo.ui import plot_circuit
import matplotlib.pyplot as plt
from qibo.models import QFT

def main():
    # Create a simple circuit
    circuit = Circuit(2)
    for _ in range(20):
        circuit.add(gates.H(0))
        circuit.add(gates.CNOT(0,1))
        circuit.add(gates.SWAP(0, 1))
    circuit.add(gates.M(*range(2)))
    plot_circuit(circuit)
    plt.show()

if __name__ == "__main__":
    main()