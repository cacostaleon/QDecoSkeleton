#!/usr/bin/env python3
"""
Example usage of QDecoSkeleton on Intel FPGA platform.
Assumes Intel FPGA SDK for OpenCL and a valid device (e.g., Stratix 10 GX).
"""
import numpy as np
from qdecoskeleton import QDecoSkeleton

def main():
    # Parameters
    N = 16               # number of qubits in the chain
    gamma = np.array([0.05 * i / N for i in range(N)])

    # Create skeleton (use 'cpu' for debugging without FPGA)
    skel = QDecoSkeleton(n_qubits=N, device='fpga')
    skel.set_decoherence_params(gamma)

    # Initial state: pure |0><0|
    rho_0 = np.array([[1, 0], [0, 0]], dtype=np.complex64)

    # Run emulation for 1024 random initial states (or fixed steps)
    result = skel.run(rho_0, steps=1024)

    print(f"Global decoherence D_bar = {result.global_decoherence:.6f}")
    print("Decoherence profile per qubit:")
    for i, d in enumerate(result.decoherence_profile):
        print(f"  Qubit {i:2d}: D = {d:.6f}")

if __name__ == "__main__":
    main()