# QDecoSkeleton: Parallel-Recursive Quantum Decoherence Emulation on FPGA

This repository contains the Python/PyOpenCL implementation of **QDecoSkeleton**, an algorithmic skeleton for emulating quantum coherence and decoherence in qubit chains using parallel recursion on FPGAs. It is based on the PARECOREH coprocessor model and the formal mapping of Kraus operators to linear recursive functions.

## Features
- Full OpenCL 2.0 support for Intel FPGAs (Stratix 10 tested)
- Dual forward/backward pipeline architecture
- Superlinear speedup (up to 62× for 64 qubits)
- Linear FPGA resource scaling
- Portable PyOpenCL abstraction

## Requirements
- Intel FPGA SDK for OpenCL (or any OpenCL 2.0 FPGA platform)
- Python 3.12+
- PyOpenCL and NumPy (see `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/QDecoSkeleton.git
   cd QDecoSkeleton# QDecoSkeleton
