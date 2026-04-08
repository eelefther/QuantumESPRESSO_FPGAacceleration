# QuantumESPRESSO_FPGAacceleration
This repository contains an FPGA-accelerated implementation of the PWscf package from Quantum ESPRESSO, a widely used open-source suite for electronic-structure calculations and materials modeling.

After profiling with Intel VTune Profiler, the Fast Fourier Transform (FFT) from the FFTW3 library and the vloc_psi_k function were identified as major performance bottlenecks and selected for acceleration.

The acceleration was implemented using Xilinx Vitis and Vitis HLS, targeting the computational hotspots and optimizing both compute and memory access patterns through custom FPGA kernels.

For this purpose, the Xilinx Alveo U200 acceleration card was used.

The accelerated implementation achieved:

4.81× speedup for the FFT (FFTW3)
4.14× speedup for the vloc_psi_k function

These results demonstrate significant performance improvements compared to the baseline CPU implementation.

This project was developed as part of my diploma thesis at the Electrical and Computer Engineering Department of the Aristotle University of Thessaloniki.
