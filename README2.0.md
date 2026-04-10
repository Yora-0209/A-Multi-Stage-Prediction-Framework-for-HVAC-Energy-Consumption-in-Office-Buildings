# A-Multi-Stage-Prediction-Approach-for-HVAC-Energy-Consumption-in-Open-Plan-Offices

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

This directory contains the core implementation of the multi-stage prediction approach described in the paper:
**"Post-Layout Energy Assessment Framework for Open-Plan Offices: A Physics-Constrained Graph Neural Network-Based Multi-Stage Prediction Approach"**

> The code is intended to demonstrate the model architecture and algorithmic logic for review. It is not a plug-and-play executable because it omits proprietary datasets (LBNL Building 59) and environment-specific configurations. However, the complete framework is laid out with detailed comments.


## Approach Overview

The proposed approach follows a **three-stage interpretable prediction chain**:

- **Stage 1: Multi-Zone Temperature Prediction** – A Physics-Constrained Graph Neural Network (PC-GNN) that combines LSTM temporal encoding with spatial thermal coupling. Three physical constraints (SP, PL, MC) are injected to ensure thermodynamic consistency.
- **Stage 2: Zone-Level Thermal Load Calculation** – Predicted temperatures are fed into a heat-balance differential equation (Eq. 3) to solve for the cooling/heating load ($Q_{HVAC}$) of each thermal zone.
- **Stage 3: HVAC Energy Estimation & Calibration** – A hybrid fusion model dynamically weights a physics-based energy conversion and a data-driven predictor using real-time error calibration.


## Physics Constraints
We explicitly implement three types of physical constraints as discussed in Section 2.3.1:
- **SP (Structural Physics)**: Graph topology mapped from office layouts.
- **PL (Physical Loss)**: Directionality constraint for heat transfer.
- **MC (Multi-step Correction)**: Inertia-based temporal smoothing.


## Key Modules Explained

### 1. `temperature_prediction.py`

| Class / Function               | Description                                                                                                                                                              | Paper Reference        |
|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| `PCGNN`                        | Main temperature prediction model. It includes an LSTM encoder, multi-head GNN layers with dynamic edge weights, zone-specific prediction heads, and multi-step correction. | Fig. 4, Eq. (1)-(2)  |
| `EnhancedGNNLayer`             | Multi-head graph convolution with temperature-aware gating. Learns dynamic thermal coupling strengths between adjacent zones.                                            | Fig. 5                |
| `PhysicalConsistencyLoss`      | Loss term that penalizes predictions violating the basic direction of heat transfer (PL constraint).                                                                    | Supplementary S2      |
| `apply_multi_step_correction` | Implements the multi-step correction (MC) using influence matrix, momentum, and time decay to enforce physical smoothness across the prediction horizon.                 | Supplementary S2      |

### 2. `energy_prediction.py`

| Class / Function         | Description                                                                                                                                                           | Paper Reference        |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| `ThermalLoadCalculator`  | Solves the heat-balance differential equation (Eq. 3) to compute net thermal load for each zone. Uses building parameters from Table 1.                               | Eq. (3), Table 1, Supplementary S3 |
| `HVACEnergyEstimator`    | Converts zone thermal loads into HVAC electrical energy using equipment COP and capacity constraints (physics-based model).                                          | Supplementary S4       |
| `HybridEnergyPredictor`  | Dynamically fuses the physics-based estimate with a data-driven model (placeholder) using a moving-window MAE to compute fusion weights $\alpha_k^t$.               | Eq. (14)-(17)          |
| `predict_step`           | One-step prediction routine that can be called iteratively to generate 24-timesteps forecasts.                                                                              | -                      |

## Dependencies

- Python >= 3.8
- PyTorch >= 1.10
- NumPy
- scikit-learn (for scalers and metrics)
- Matplotlib, Pandas (Optional, for visualization and data handling) 
