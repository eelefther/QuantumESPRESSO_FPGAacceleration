# QuantumESPRESSO_FPGAacceleration

An FPGA-accelerated implementation of the **PWscf** package from **Quantum ESPRESSO**, an open-source suite for electronic-structure calculations and materials modeling.

---

## 📌 Overview

This project focuses on accelerating computational bottlenecks of Quantum ESPRESSO using custom FPGA kernels.

* **Profiling**: Identified critical performance bottlenecks using **Intel VTune Profiler**:
  * Fast Fourier Transform (`FFT`) from the **FFTW3** library
  * `vloc_psi_k` function
* **Target Hardware**: **Xilinx Alveo U200** Acceleration Card
* **Development Tools**: Xilinx Vitis & Vitis HLS

---

## 🚀 Performance & Results

By optimizing compute patterns and memory access through dedicated FPGA kernels, significant speedups were achieved compared to the baseline CPU implementation:

| Kernel / Function | Speedup |
| :--- | :---: |
| **FFT (FFTW3)** | **4.81×** |
| **`vloc_psi_k`** | **4.14×** |

> These performance improvements demonstrate the efficiency of hardware acceleration for high-performance scientific computing tasks.

---

## 🎓 Academic Context

This repository was developed as part of a **Diploma Thesis** at the:
* **Department**: Electrical and Computer Engineering (ECE)
* **Institution**: Aristotle University of Thessaloniki (AUTH)

---

## 🛠️ Tech Stack & Tools

* **Software / Libraries**: Quantum ESPRESSO (PWscf), FFTW3, Intel VTune Profiler
* **FPGA Tools**: Xilinx Vitis, Vitis HLS
* **Hardware**: Xilinx Alveo U200
