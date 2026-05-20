# Using QDecoSkeleton from the host CPU
from qdecoskeleton import QDecoSkeleton
import numpy as np

N = 16  # Number of qubits in the chain
# Decoherence parameters: linear ramp
gamma = np.array([0.1*i/N for i in range(N)])
# Instantiate skeleton with FPGA as the computing device
skel = QDecoSkeleton(n_qubits=N, device='fpga', kraus_type='amplitude_damping')
skel.set_decoherence_params(gamma)
# Initial state: pure density matrix |0><0|
rho_0 = np.array([[1, 0], [0, 0]], dtype=np.complex64)
# Run the simulation: Q_input -> FPGA -> Q_output
result = skel.run(rho_0, steps=100)
D_global = result.global_decoherence
print(f"Calculated global decoherence: {D_global:.4f}")