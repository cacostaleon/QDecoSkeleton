import pyopencl as cl
import numpy as np

# Setup OpenCL context and queue for Intel Stratix 10
platform = cl.get_platforms()[0]  # Intel FPGA platform
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Load and build kernels from source (fcore_kernel, bcore_kernel)
with open('kernels.cl', 'r') as f:
    kernel_src = f.read()
program = cl.Program(context, kernel_src).build()

# Input queue: m initial density matrices (4 floats each, complex as float2)
m = 1024
rho_input = np.random.rand(m, 4).astype(np.float32) + 1j*np.random.rand(m,4).astype(np.float32)
# flatten and convert to float2 representation
rho_flat = rho_input.view(np.float32).reshape(-1, 2)  # each complex becomes (re, im)

# Allocate device buffers
rho_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=rho_flat)
# For each level i (0..N-1), we need intermediate storage for forward results
intermediate_buffers = [cl.Buffer(context, cl.mem_flags.READ_WRITE, rho_flat.nbytes) for _ in range(N)]

# Forward pipeline: recursive depth N
for i in range(N):
    kernel = program.fcore_kernel
    kernel.set_arg(0, rho_buf if i==0 else intermediate_buffers[i-1])
    kernel.set_arg(1, intermediate_buffers[i])
    kernel.set_arg(2, gamma_i)  # parameter for this qubit
    kernel.set_arg(3, i)
    # Launch kernel for all m inputs in parallel (global size = m)
    cl.enqueue_nd_range_kernel(queue, kernel, (m,), None)
# After forward phase, the final states for each qubit i are stored in intermediate_buffers[i]

# Backward pipeline: reduction
D_out = np.zeros((m, N), dtype=np.float32)
D_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=D_out)
for i in range(N-1, -1, -1):
    kernel = program.bcore_kernel
    kernel.set_arg(0, intermediate_buffers[i])
    kernel.set_arg(1, D_buf)
    kernel.set_arg(2, i)
    kernel.set_arg(3, N)
    cl.enqueue_nd_range_kernel(queue, kernel, (m,), None)
# Final D_global per input is the average across i
cl.enqueue_copy(queue, D_out, D_buf)
D_global_avg = np.mean(D_out, axis=1)