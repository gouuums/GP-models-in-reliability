This repository provides open-source Python implementations of advanced active learning algorithms for structural reliability analysis[cite: 607]. [cite_start]Developed as a final project for **ENSAI** in collaboration with the **DGA** (Direction Générale de l'Armement), this project focuses on the **AMGPRA** method[cite: 1, 5, 137].

## 📌 Project Overview
The primary objective is to estimate the **Probability of Failure ($P_f$)** for complex physical systems where standard numerical simulations are computationally prohibitive. By leveraging **multi-fidelity metamodels** (Gaussian processes-Kriging), we integrate high-fidelity (HF) evaluations with significantly cheaper low-fidelity (LF) simulations. This approach reduces overall computational costs while maintaining high precision in identifying the limit state boundary.

### Key Implemented Methods
[cite_start]This repository includes scratch-built implementations for the following strategies:

* **AK-MCS** (Active learning Kriging - Monte Carlo Simulation): A baseline single-fidelity approach using the $U$ learning function to identify the most informative enrichment points.
* **mfEGRA** (Multi-fidelity Efficient Global Reliability Analysis): A multi-fidelity method utilizing a two-step process that first selects the point and then the fidelity level based on information gain.
* **AMGPRA** (Adaptive Multi-fidelity Gaussian Process for Reliability Analysis): An advanced algorithm that jointly selects both the optimal sample point and the fidelity source simultaneously using a **Collective Learning Function (CLF)**.

## 🛠️ Implementation Details
All algorithms are developed in **Python** using the **SMT (Surrogate Modeling Toolbox)** library as the underlying engine.

* **Surrogate Model:** Multi-Fidelity Kriging (**MFK**) based on the additive first-order autoregressive formulation by Kennedy and O'Hagan.
* **Acquisition Functions:** Support for the **EFF** (Expected Feasibility Function) and **U-function**.
* **Performance:** Implementations are optimized using vectorization to process large Monte Carlo populations ($10^4$ to $10^6$ samples) efficiently.

## 🚀 Application Examples
The repository includes benchmark cases demonstrating method robustness and efficiency:

### 1. Analytical Reference (Example 5.1)
A reproducibility test involving a 2D non-linear, multi-modal limit state function. 
* **Result:** Successfully recovered the cost hierarchy and accuracy reported in original literature ($\hat{P}_f \approx 0.0313$).

### 2. Robustness Study (Shifted Contours)
A sensitivity analysis exploring method behavior when low-fidelity models provide biased indications of the failure boundary.

### 3. Industrial Case Study: `solar` Benchmark
Application to the **Solar 10.1** simulator (Concentrated Solar Power plant).
* **Dimension:** 5D continuous input space (geometry and temperatures).
* **Objective:** Quantify the robustness of designs against budget overruns.
* **Features:** Construction of a **fragility curve** to identify the system's "robust zone".

## 👥 Contributors 
* **Thomas Goumont** (ENSAI) 
* **Léo Gruchociak** (ENSAI) 
* **Inès Ben Moussa** (ENSAI) 

## 📜 Selected References
* [1] Zhang, C., Song, C., & Shafieezadeh, A. (2022). *Adaptive Reliability Analysis for Multi-fidelity Models using a Collective Learning Strategy*. Structural Safety.
* [2] Chaudhuri, A., Marques, A. N., & Willcox, K. E. (2020). *mfEGRA: Multifidelity Efficient Global Reliability Analysis*.
* [3] Echard, B., Gayton, N., & Lemaire, M. (2011). *AK-MCS: An active learning reliability method combining Kriging and Monte Carlo Simulation*. Structural Safety.
