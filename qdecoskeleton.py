"""
QDecoSkeleton — Python/PyOpenCL implementation
Parallel-Recursive Emulation of Quantum Decoherence (PARECOREH model)
Platform: Intel Stratix 10 GX FPGA + Intel Core i7-13700HX
Requires: pyopencl >= 2024.1, numpy, Intel FPGA SDK for OpenCL 22.1
"""
import numpy as np
import pyopencl as cl
from dataclasses import dataclass
from typing import Optional

# OpenCL kernel source codes (embedded)
FCORE_KERNEL_SRC = r"""
__kernel void fcore_kernel(
    __global const float2* rho_in,
    __global       float2* rho_out,
    __global const float*  gamma,
    const int level)
{
    int id = get_global_id(0);
    float2 r00 = rho_in[4*id+0], r01 = rho_in[4*id+1];
    float2 r10 = rho_in[4*id+2], r11 = rho_in[4*id+3];
    float g = gamma[level];
    float sq = sqrt(1.0f - g);
    rho_out[4*id+0] = r00 + (float2)(g, 0.0f) * r11;
    rho_out[4*id+1] = (float2)(sq, 0.0f) * r01;
    rho_out[4*id+2] = (float2)(sq, 0.0f) * r10;
    rho_out[4*id+3] = (float2)(1.0f - g, 0.0f) * r11;
}
"""

BCORE_KERNEL_SRC = r"""
__kernel void bcore_kernel(
    __global const float2* rho,
    __global       float*  D_out,
    const int level,
    const int N)
{
    int id = get_global_id(0);
    float2 r00 = rho[4*id+0], r01 = rho[4*id+1];
    float2 r10 = rho[4*id+2], r11 = rho[4*id+3];
    float purity = r00.x*r00.x + r00.y*r00.y
                 + r01.x*r01.x + r01.y*r01.y
                 + r10.x*r10.x + r10.y*r10.y
                 + r11.x*r11.x + r11.y*r11.y;
    D_out[id * N + level] = 1.0f - purity;
}
"""

@dataclass
class QDecoResult:
    decoherence_profile: np.ndarray   # D_i for each qubit level
    global_decoherence:  float        # D_bar = mean(D_i)
    rho_chain:           np.ndarray   # evolved density matrices (final level)

class QDecoSkeleton:
    def __init__(self, n_qubits: int, device: str = 'fpga', kraus_type: str = 'amplitude_damping'):
        self.N = n_qubits
        self.kraus_type = kraus_type
        self.gamma = None

        # PyOpenCL platform/device selection
        platforms = cl.get_platforms()
        fpga_platform = next((p for p in platforms if 'Intel' in p.name), platforms[0])
        if device == 'fpga':
            devices = fpga_platform.get_devices(device_type=cl.device_type.ACCELERATOR)
        else:
            devices = fpga_platform.get_devices(device_type=cl.device_type.CPU)
        self.ctx = cl.Context(devices=devices)
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # Build program from embedded kernel sources
        prg = cl.Program(self.ctx, FCORE_KERNEL_SRC + BCORE_KERNEL_SRC).build(options='-cl-std=CL2.0')
        self.fcore_kn = prg.fcore_kernel
        self.bcore_kn = prg.bcore_kernel

    def set_decoherence_params(self, gamma: np.ndarray) -> None:
        assert len(gamma) == self.N
        self.gamma = gamma.astype(np.float32)
        mf = cl.mem_flags
        self._gamma_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.gamma)

    def run(self, rho_0: np.ndarray, steps: int = 1) -> QDecoResult:
        assert self.gamma is not None, "Call set_decoherence_params first"
        mf = cl.mem_flags
        m = steps
        N = self.N

        # Prepare Q_input: m copies of rho_0
        rho_host = np.tile(rho_0.astype(np.complex64).flatten(), (m, 1)).astype(np.complex64)
        # Device buffers for each pipeline stage
        rho_bufs = [cl.Buffer(self.ctx, mf.READ_WRITE, size=rho_host.nbytes) for _ in range(N+1)]
        D_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=m * N * np.dtype(np.float32).itemsize)

        # Upload Q_input
        cl.enqueue_copy(self.queue, rho_bufs[0], rho_host)

        # Forward pipeline (map)
        for i in range(N):
            self.fcore_kn(self.queue, (m,), None,
                          rho_bufs[i], rho_bufs[i+1], self._gamma_buf, np.int32(i))

        # Backward pipeline (reduce)
        for i in range(N-1, -1, -1):
            self.bcore_kn(self.queue, (m,), None,
                          rho_bufs[i+1], D_buf, np.int32(i), np.int32(N))

        self.queue.finish()

        # Download results
        D_host = np.empty(m * N, dtype=np.float32)
        cl.enqueue_copy(self.queue, D_host, D_buf)
        rho_final = np.empty_like(rho_host)
        cl.enqueue_copy(self.queue, rho_final, rho_bufs[N])
        self.queue.finish()

        D_matrix = D_host.reshape(m, N)
        D_profile = D_matrix.mean(axis=0)
        D_global = float(D_profile.mean())

        return QDecoResult(decoherence_profile=D_profile, global_decoherence=D_global, rho_chain=rho_final.reshape(m,4))