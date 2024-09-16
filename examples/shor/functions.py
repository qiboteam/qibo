import numpy as np

from qibo import Circuit, gates


def adder_angles(a, n):
    """Classical computation of the angles needed for adder in Fourier space.
    Args:
        a (int) = number to add.
        n (int) = number of bits without overflow.

    Returns:
        angles (list) = list of angles in order of application.
    """
    A = "{0:0{bits}b}".format(a, bits=n + 1)
    angles = []
    for i in reversed(range(len(A))):
        angle = 0
        m = 1
        for j in range(i, len(A)):
            if int(A[j]) == 1:
                angle += 2 * np.pi / (2**m)
            m += 1
        angles.append(angle)
    return angles


def phi_adder(b, angles):
    """Quantum adder in Fourier space.
    Args:
        b (list): quantum register where the addition is implemented.
        angles (list): list of angles that encode the number to be added.

    Returns:
        generator with the required quantum gates applied on the quantum circuit.
    """
    for i in range(0, len(b)):
        yield gates.U1(b[i], angles[i])


def i_phi_adder(b, angles):
    """(Inverse) Quantum adder in Fourier space.
    Args:
        b (list): quantum register where the addition is implemented.
        angles (list): list of angles that encode the number to be added.

    Returns:
        generator with the required quantum gates applied on the quantum circuit.
    """
    for i in reversed(range(0, len(b))):
        yield gates.U1(b[i], -angles[i])


def c_phi_adder(c, b, angles):
    """(1 Control) Quantum adder in Fourier space.
    Args:
        c (int): qubit acting as control.
        b (list): quantum register where the addition is implemented.
        angles (list): list of angles that encode the number to be added.

    Returns:
        generator with the required quantum gates applied on the quantum circuit.
    """
    for i in range(0, len(b)):
        yield gates.U1(b[i], angles[i]).controlled_by(c)


def i_c_phi_adder(c, b, angles):
    """(Inverse) (1 Control) Quantum adder in Fourier space.
    Args:
        c (int): qubit acting as control.
        b (list): quantum register where the addition is implemented.
        angles (list): list of angles that encode the number to be added.

    Returns:
        generator with the required quantum gates applied on the quantum circuit.
    """
    for i in reversed(range(0, len(b))):
        yield gates.U1(b[i], -angles[i]).controlled_by(c)


def cc_phi_adder(c1, c2, b, angles):
    """(2 Controls) Quantum adder in Fourier space.
    Args:
        c1 (int): qubit acting as first control.
        c2 (int): qubit acting as second control.
        b (list): quantum register where the addition is implemented.
        angles (list): list of angles that encode the number to be added.

    Returns:
        generator with the required quantum gates applied on the quantum circuit.
    """
    for i in range(0, len(b)):
        yield gates.U1(b[i], angles[i]).controlled_by(c1, c2)


def i_cc_phi_adder(c1, c2, b, angles):
    """(Inverse) (2 Controls) Quantum adder in Fourier space.
    Args:
        c1 (int): qubit acting as first control.
        c2 (int): qubit acting as second control.
        b (list): quantum register where the addition is implemented.
        angles (list): list of angles that encode the number to be added.

    Returns:
        generator with the required quantum gates applied on the quantum circuit.
    """
    for i in reversed(range(0, len(b))):
        yield gates.U1(b[i], -angles[i]).controlled_by(c1, c2)


def qft(q):
    """Quantum Fourier Transform on a quantum register.
    Args:
        q (list): quantum register where the QFT is applied.

    Returns:
        generator with the required quantum gates applied on the quantum circuit.
    """
    for i1 in range(len(q)):
        yield gates.H(q[i1])
        for i2 in range(i1 + 1, len(q)):
            theta = np.pi / 2 ** (i2 - i1)
            yield gates.CU1(q[i2], q[i1], theta)
    for i in range(len(q) // 2):
        yield gates.SWAP(i, len(q) - i - 1)


def i_qft(q):
    """(Inverse) Quantum Fourier Transform on a quantum register.
    Args:
        q (list): quantum register where the QFT is applied.

    Returns:
        generator with the required quantum gates applied on the quantum circuit.
    """
    for i in range(len(q) // 2):
        yield gates.SWAP(i, len(q) - i - 1)
    for i1 in reversed(range(len(q))):
        for i2 in reversed(range(i1 + 1, len(q))):
            theta = np.pi / 2 ** (i2 - i1)
            yield gates.CU1(q[i2], q[i1], -theta)
        yield gates.H(q[i1])


def cc_phi_mod_adder(c1, c2, b, ang_a, ang_N, ancilla):
    """(2 Controls) Quantum modular addition with fixed a and N in Fourier space.
    Args:
        c1 (int): qubit acting as first control.
        c2 (int): qubit acting as second control.
        b (list): quantum register where the addition is implemented.
        ang_a (list): list of angles that encode the number to be added.
        ang_N (list): list of angles that encode the modulo number.
        ancilla (int): extra qubit needed for the computation.

    Returns:
        generator with the required quantum gates applied on the quantum circuit.
    """
    yield cc_phi_adder(c1, c2, b, ang_a)
    yield i_phi_adder(b, ang_N)
    yield i_qft(b)
    yield gates.CNOT(b[0], ancilla)
    yield qft(b)
    yield c_phi_adder(ancilla, b, ang_N)
    yield i_cc_phi_adder(c1, c2, b, ang_a)
    yield i_qft(b)
    yield gates.X(b[0]), gates.CNOT(b[0], ancilla), gates.X(b[0])
    yield qft(b)
    yield cc_phi_adder(c1, c2, b, ang_a)


def i_cc_phi_mod_adder(c1, c2, b, ang_a, ang_N, ancilla):
    """(Inverse) (2 Controls) Quantum modular addition with fixed a and N in Fourier space.
    Args:
        c1 (int): qubit acting as first control.
        c2 (int): qubit acting as second control.
        b (list): quantum register where the addition is implemented.
        ang_a (list): list of angles that encode the number to be added.
        ang_N (list): list of angles that encode the modulo number.
        ancilla (int): extra qubit needed for the computation.

    Returns:
        generator with the required quantum gates applied on the quantum circuit.
    """
    yield i_cc_phi_adder(c1, c2, b, ang_a)
    yield i_qft(b)
    yield gates.X(b[0]), gates.CNOT(b[0], ancilla), gates.X(b[0])
    yield qft(b)
    yield cc_phi_adder(c1, c2, b, ang_a)
    yield i_c_phi_adder(ancilla, b, ang_N)
    yield i_qft(b)
    yield gates.CNOT(b[0], ancilla)
    yield qft(b)
    yield phi_adder(b, ang_N)
    yield i_cc_phi_adder(c1, c2, b, ang_a)


def c_mult_mod(c, x, b, a, N, ancilla, n):
    """(1 Control) Quantum modular multiplication. |1>|x>|b> --> |1>|x>|(b+a*x)%N>
    Args:
        c (int): qubit acting as control.
        x (list): quantum register encoding the number of times a is multiplied.
        b (list): quantum register where the final result is encoded.
        a (int): number to multiply.
        N (int): modulo.
        ancilla (int): extra quantum register needed for the modular addition.
        n (int): number of bits needed to encode N

    Returns:
        generator with the required quantum gates applied on the quantum circuit.
    """
    ang_N = adder_angles(N, n)
    yield qft(b)
    for i in range(len(x)):
        ang_a = adder_angles((a * 2 ** (n - 1 - i)) % N, n)
        yield cc_phi_mod_adder(c, x[i], b, ang_a, ang_N, ancilla)
    yield i_qft(b)


def i_c_mult_mod(c, x, b, a, N, ancilla, n):
    """(Inverse) (1 Control) Quantum modular multiplication.  |1>|x>|(b+a*x)%N> --> |1>|x>|b>
    Args:
        c (int): qubit acting as control.
        x (list): quantum register encoding the number of times a is multiplied.
        b (list): quantum register where the final result is encoded.
        a (int): number to multiply.
        N (int): modulo.
        ancilla (int): extra quantum register needed for the modular addition.
        n (int): number of bits needed to encode N

    Returns:
        generator with the required quantum gates applied on the quantum circuit.
    """
    ang_N = adder_angles(N, n)
    yield qft(b)
    for i in reversed(range(len(x))):
        ang_a = adder_angles((a * 2 ** (n - 1 - i)) % N, n)
        yield i_cc_phi_mod_adder(c, x[i], b, ang_a, ang_N, ancilla)
    yield i_qft(b)


def c_U(c, x, b, a, N, ancilla, n):
    """(1 Control) Quantum circuit that applies the modular multiplication as a part of the
    modular exponentiation. |1>|x>|0> ---> |1>|(a*x)%N>|0>
    Args:
        c (int): qubit acting as control.
        x (list): quantum register encoding the number of times a is multiplied.
        b (list): quantum register initialized at |0>
        a (int): number to multiply.
        N (int): modulo.
        ancilla (int): extra quantum register needed for the modular addition.
        n (int): number of bits needed to encode N

    Returns:
        generator with the required quantum gates applied on the quantum circuit.
    """
    yield c_mult_mod(c, x, b, a, N, ancilla, n)
    for i in range(1, len(b)):
        yield gates.SWAP(b[i], x[i - 1]).controlled_by(c)
    a_inv = mod_inv(a, N)
    yield i_c_mult_mod(c, x, b, a_inv, N, ancilla, n)


def find_factor_of_prime_power(N):
    """Classical algorithm to check if given number is a prime power.
    Args:
        N (int): number to factorize.

    Returns:
        If the number is not a prime power: None
        If the number is a prime power: f1 (int) or f2 (int)
    """
    for i in range(2, int(np.floor(np.log2(N))) + 1):
        f = N ** (1 / i)
        f1 = np.floor(f)
        if f1**i == N:
            return f1
        f2 = np.ceil(f)
        if f2**i == N:
            return f2
    return None


def egcd(a, b):
    """Extended Euclid's algorithm for the modular inverse calculation."""
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)


def mod_inv(a, N):
    """Classical algorithm to calculate the inverse of a modulo N.
    Args:
        a (int): number to invert.
        N (int): modulo.

    Returns:
        x%N (int): inverse of a modulo N.
    """
    g, x, y = egcd(a, N)
    if g != 1:
        raise ValueError("modular inverse does not exist")
    else:
        return x % N


def quantum_order_finding_full(N, a):
    """Quantum circuit that performs the order finding algorithm using a fully quantum iQFT.
    Args:
        N (int): number to factorize.
        a (int): chosen number to use in the algorithm.

    Returns:
        s (float): value of the state measured by the quantum computer.
    """
    print("  - Performing algorithm using a fully quantum iQFT.\n")
    # Creating the parts of the needed quantum circuit.
    n = int(np.ceil(np.log2(N)))
    b = [i for i in range(n + 1)]
    x = [n + 1 + i for i in range(n)]
    ancilla = 2 * n + 1
    q_reg = [2 * n + 2 + i for i in range(2 * n)]
    circuit = Circuit(4 * n + 2)
    print(f"  - Total number of qubits used: {4*n+2}.\n")

    # Building the quantum circuit
    for i in range(len(q_reg)):
        circuit.add(gates.H(q_reg[i]))
    circuit.add(gates.X(x[len(x) - 1]))
    exponents = []
    exp = a % N
    for i in range(len(q_reg)):
        exponents.append(exp)
        exp = (exp**2) % N
    # a**(2**i)
    circuit.add(
        (c_U(q, x, b, exponents[i], N, ancilla, n) for i, q in enumerate(q_reg))
    )
    circuit.add(i_qft(q_reg))

    # Adding measurement gates
    circuit.add(gates.M(*q_reg))
    result = circuit(nshots=1)
    s = result.frequencies(binary=False).most_common()[0][0]
    print(f"The quantum circuit measures s = {s}.\n")
    return s


def quantum_order_finding_semiclassical(N, a):
    """Quantum circuit that performs the order finding algorithm using a semiclassical iQFT.
    Args:
        N (int): number to factorize.
        a (int): chosen number to use in the algorithm.

    Returns:
        s (float): value of the state measured by the quantum computer.
    """
    print("  - Performing algorithm using a semiclassical iQFT.\n")
    # Creating the parts of the needed quantum circuit.
    n = int(np.ceil(np.log2(N)))
    b = [i for i in range(n + 1)]
    x = [n + 1 + i for i in range(n)]
    ancilla = 2 * n + 1
    q_reg = 2 * n + 2
    print(f"  - Total number of qubits used: {2*n+3}.\n")
    results = []
    exponents = []
    exp = a % N
    for i in range(2 * n):
        exponents.append(exp)
        exp = (exp**2) % N

    circuit = Circuit(2 * n + 3)
    # Building the quantum circuit
    circuit.add(gates.H(q_reg))
    circuit.add(gates.X(x[len(x) - 1]))
    # a_i = (a**(2**(2*n - 1)))
    circuit.add(c_U(q_reg, x, b, exponents[-1], N, ancilla, n))
    circuit.add(gates.H(q_reg))
    results.append(circuit.add(gates.M(q_reg, collapse=True)))
    # Using multiple measurements for the semiclassical QFT.
    for i in range(1, 2 * n):
        # reset measured qubit to |0>
        circuit.add(gates.RX(q_reg, theta=np.pi * results[-1].symbols[0]))
        circuit.add(gates.H(q_reg))
        # a_i = (a**(2**(2*n - 1 - i)))
        circuit.add(c_U(q_reg, x, b, exponents[-1 - i], N, ancilla, n))
        angle = 0
        for k in range(2, i + 2):
            angle += 2 * np.pi * results[i + 1 - k].symbols[0] / (2**k)
        circuit.add(gates.U1(q_reg, -angle))
        circuit.add(gates.H(q_reg))
        results.append(circuit.add(gates.M(q_reg, collapse=True)))
    circuit.add(gates.M(q_reg))

    circuit(nshots=1)  # execute
    s = sum(int(r.symbols[0].outcome()) * (2**i) for i, r in enumerate(results))
    print(f"The quantum circuit measures s = {s}.\n")
    return s


def find_factors(r, a, N):
    if r % 2 != 0:
        print("The value found for r is not even. Trying again.\n")
        print("-" * 60 + "\n")
        return None
    if a ** (r // 2) == -1 % N:
        print("Unusable value for r found. Trying again.\n")
        print("-" * 60 + "\n")
        return None
    f1 = np.gcd((a ** (r // 2)) - 1, N)
    f2 = np.gcd((a ** (r // 2)) + 1, N)
    if (f1 == N or f1 == 1) and (f2 == N or f2 == 1):
        print(f"Trivial factors 1 and {N} found. Trying again.\n")
        print("-" * 60 + "\n")
        return None
    if f1 != 1 and f1 != N:
        f2 = N // f1
    elif f2 != 1 and f2 != N:
        f1 = N // f2
    print(f"Found as factors for {N}:  {f1}  and  {f2}.\n")
    return f1, f2
