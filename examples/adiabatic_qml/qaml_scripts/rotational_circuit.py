from functools import lru_cache

import numpy as np
from qaml_scripts.evolution import generate_schedule
from scipy.integrate import quad

from qibo import Circuit, gates


class rotational_circuit:
    def __init__(self, best_p, finalT, nqubits=1, q=0):
        """
        Class containing all functions used for generating the circuit composed
        of three rotations.

        Args:
            best_p (float array): best parameters array.
            finalT (float): final real time in the evolution.
            nqubits (int): number of qubits of the quantum device (default 1).
            q (int): target qubit used into the device (default 0).
        """

        self.poly, self.derpoly = generate_schedule(best_p)
        self.finalT = finalT
        self.nqubits = nqubits
        self.q = q

    @lru_cache
    def sched(self, t):
        """The schedule at a time t is poly(t/finalT)"""
        return self.poly(t / self.finalT)

    @lru_cache
    def eigenval(self, t):
        """Compute the eigenvalue of the Hamiltonian at a time t"""
        s = self.sched(t)
        return np.sqrt(s**2 + (1 - s) ** 2)

    @lru_cache
    def integral_eigen(self, t):
        """Compute the integral of eigenval from 0 to t"""
        res = quad(self.eigenval, 0, t)
        return res[0]

    @lru_cache
    def u00(self, t, swap=1.0):
        """Compute the value of u00 (real and imag) at a time T"""
        if t == self.finalT:
            t -= 1e-2
        integral = self.integral_eigen()
        l = self.eigenval(t)
        s = self.sched(t)

        # Normalization for the change of basis matrix P^{-1}HP = H_diagonal so that PP^-1 = I
        normalize = 2.0 / (1 - s) * np.sqrt(l * (l - s))
        fac = swap * (l - s) / (1 - s)

        # (the multiplication by t not sure where does it come from)
        ti = t / self.finalT
        real_part = swap * np.cos(integral) * (1 + fac) / normalize
        imag_part = np.sin(integral) * (1 - fac) / normalize

        return real_part, imag_part

    @lru_cache
    def u10(self, t):
        """Compute the value of u10 (real and imag), the offdiagonal term"""
        pr, pi = self.u00(t, swap=-1.0)
        return pr, pi

    @lru_cache
    def old_rotation_angles(self, t):
        x, y = self.u00(t)
        u, z = self.u10(t)

        a = x + 1j * y
        b = -u + 1j * z

        arga = np.angle(a)
        moda = np.absolute(a)
        argb = np.angle(b)

        theta = -2 * np.arccos(moda)
        psi = -0.5 * np.pi - arga + argb
        phi = -arga + np.pi * 0.5 - argb
        return psi, theta, phi

    @lru_cache
    def sched_p(self, t):
        """The schedule at a time t is poly(t/finalT)"""
        return self.derpoly(t / self.finalT) / self.finalT

    @lru_cache
    def eigenvalp(self, l, s, sp):
        return sp * (2 * s - 1) / l

    @lru_cache
    def n(self, l, s):
        return (1 - s) / 2 / np.sqrt(l * (l - s))

    @lru_cache
    def nder(self, l, s, lp, sp):
        roote = l * (l - s)
        upder = (1 - s) * (2 * lp * l - lp * s - sp * l)
        inter = -sp - upder / 2 / roote
        return 1 / 2 / np.sqrt(roote) * inter

    @lru_cache
    def f(self, l, s):
        return (l - s) / (1 - s)

    @lru_cache
    def fp(self, l, s, lp, sp):
        return (lp - sp - lp * s + sp * l) / (1 - s) ** 2

    @lru_cache
    def rotation_angles(self, t):
        """Calculates rotation angles."""

        I = self.integral_eigen(t)
        l = self.eigenval(t)
        s = self.sched(t)

        fac = self.f(l, s)

        norma = self.n(l, s)
        inside00 = gt = 1 + fac**2 + 2 * fac * np.cos(2 * I)

        absu00 = norma * np.sqrt(inside00)

        upf = 1 - l
        dpf = 1 + l - 2 * s
        sinIt = np.sin(I)
        cosIt = np.cos(I)

        arga = np.arctan2(sinIt * upf, cosIt * dpf)
        argb = np.arctan2(sinIt * dpf, -cosIt * upf)

        theta = -2 * np.arccos(absu00)
        psi = -0.5 * np.pi - arga + argb
        phi = -arga + np.pi * 0.5 - argb

        return phi, theta, psi

    @lru_cache
    def derivative_rotation_angles(self, t):
        """Calculates derivatives of the rotation angles."""
        s = self.sched(t)
        l = self.eigenval(t)
        I = self.integral_eigen(t)

        derI = l
        sp = self.sched_p(t)
        lp = self.eigenvalp(l, s, sp)

        nt = self.n(l, s)
        ntp = self.nder(l, s, lp, sp)

        ft = self.f(l, s)
        ftp = self.fp(l, s, lp, sp)

        gt = 1 + ft**2 + 2 * ft * np.cos(2 * I)

        # Terms of the final sum for the derivative of the theta
        x1 = ntp * np.sqrt(gt)
        y1 = 2 * ft * ftp

        y2 = 2 * ftp * np.cos(2 * I)
        y3 = -2 * ft * np.sin(2 * I) * (2 * derI)

        dgt = y1 + y2 + y3

        absu00 = nt * np.sqrt(gt)
        dabsu = x1 + nt / np.sqrt(gt) * (dgt) / 2.0
        darcos = 2.0 / np.sqrt(1 - absu00**2)
        dtheta = darcos * dabsu

        # Let's do the derivative of the phi,psi
        upf = 1 - l
        dpf = 1 + l - 2 * s
        tanI = np.tan(I)

        dtan = l / np.cos(I) ** 2
        dfrac_01 = 2 * (lp - sp + sp * l - s * lp) / upf**2
        dfrac_00 = -2 * (lp - sp + sp * l - s * lp) / dpf**2

        inside_arga = tanI * (upf / dpf)
        inside_argb = -tanI * (dpf / upf)

        dinside_arga = dtan * (upf / dpf) + tanI * dfrac_00
        darga = dinside_arga / (1 + inside_arga**2)

        dinside_argb = -dtan * (dpf / upf) - tanI * dfrac_01
        dargb = dinside_argb / (1 + inside_argb**2)

        dpsi = -darga + dargb
        dphi = -darga - dargb

        return dphi, dtheta, dpsi

    @lru_cache
    def rotations_circuit(self, t):
        psi, theta, phi = self.rotation_angles(t)

        c = Circuit(self.nqubits, density_matrix=True)

        # H gate
        c.add(gates.RZ(q=self.q, theta=np.pi / 2, trainable=False))
        c.add(gates.RX(q=self.q, theta=np.pi / 2, trainable=False))
        c.add(gates.RZ(q=self.q, theta=np.pi / 2, trainable=False))

        # RZ(psi)
        c.add(gates.RZ(q=self.q, theta=psi))

        # RX(theta)
        c.add(gates.RZ(q=self.q, theta=np.pi / 2, trainable=False))
        c.add(gates.RX(q=self.q, theta=-np.pi / 2, trainable=False))
        c.add(gates.RZ(q=self.q, theta=-theta))
        c.add(gates.RX(q=self.q, theta=np.pi / 2, trainable=False))

        # RZ(phi)
        c.add(gates.RZ(q=self.q, theta=phi))

        c.add(gates.M(self.q))

        return c

    @lru_cache
    def numeric_derivative(self, t, h=1e-7):
        # Do the derivative with 4 points
        a1, b1, c1 = self.rotation_angles(t + 2 * h)
        a2, b2, c2 = self.rotation_angles(t + h)
        a3, b3, c3 = self.rotation_angles(t - h)
        a4, b4, c4 = self.rotation_angles(t - 2 * h)

        dd1 = (-a1 + 8 * a2 - 8 * a3 + a4) / 12 / h
        dd2 = (-b1 + 8 * b2 - 8 * b3 + b4) / 12 / h
        dd3 = (-c1 + 8 * c2 - 8 * c3 + c4) / 12 / h
        return dd1, dd2, dd3
