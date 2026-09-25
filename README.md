# QuantumESPRESSO_FPGAacceleration

An FPGA-accelerated implementation of the **PWscf** package from **Quantum ESPRESSO**, an open-source suite for electronic-structure calculations and materials modeling.

---

## 📌 Overview

This project focuses on accelerating computational bottlenecks of Quantum ESPRESSO using custom FPGA kernels. 

* **Profiling**: Identified critical performance bottlenecks using **Intel VTune Profiler**:
  * Fast Fourier Transform (`FFT`) from the **FFTW3** library[cite: 1]
  * `vloc_psi_k` function[cite: 1]
* **Target Hardware**: **Xilinx Alveo U200** Acceleration Card[cite: 1]
* **Development Tools**: Xilinx Vitis & Vitis HLS[cite: 1]

---

## 🚀 Performance & Results

By optimizing compute patterns and memory access through dedicated FPGA kernels, significant speedups were achieved compared to the baseline CPU implementation[cite: 1]:

| Kernel / Function | Speedup |
| :--- | :---: |
| **FFT (FFTW3)** | **4.81×** |
| **`vloc_psi_k`** | **4.14×** |

> These performance improvements demonstrate the efficiency of hardware acceleration for high-performance scientific computing tasks[cite: 1].

---

## 🎓 Academic Context

This repository was developed as part of a **Diploma Thesis** at the:
* **Department**: Electrical and Computer Engineering (ECE)[cite: 1]
* **Institution**: Aristotle University of Thessaloniki (AUTH)[cite: 1]

---

## 🛠️ Tech Stack & Tools

* **Software / Libraries**: Quantum ESPRESSO (PWscf), FFTW3, Intel VTune Profiler[cite: 1]
* **FPGA Tools**: Xilinx Vitis, Vitis HLS[cite: 1]
* **Hardware**: Xilinx Alveo U200[cite: 1]
