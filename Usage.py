from qdecoskeleton import QDecoSkeleton
import numpy as np

skel = QDecoSkeleton(n_qubits=16, device='fpga')
gamma = np.array([0.05*i/16 for i in range(16)])
skel.set_decoherence_params(gamma)
rho_0 = np.array([[1,0],[0,0]], dtype=np.complex64)
result = skel.run(rho_0, steps=1024)
print(result.global_decoherence)