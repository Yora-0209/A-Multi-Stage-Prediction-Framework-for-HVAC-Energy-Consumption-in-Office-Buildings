# A-Multi-Stage-Prediction-Approach-for-HVAC-Energy-Consumption-in-Open-Plan-Offices

This directory contains the core implementation of the multi-stage prediction approach described in:
**"Post-Layout Energy Assessment Framework for Open-Plan Offices: A Physics-Constrained Graph Neural Network-Based Multi-Stage Prediction Approach"**

## Implementation Overview
The framework is organized into three interpretable stages to ensure thermodynamic consistency:
1. **Stage 1**: Multi-zone temperature prediction using Physics-Constrained Graph Neural Networks (PC-GNN).
2. **Stage 2**: Zonal thermal load calculation based on heat balance equations.
3. **Stage 3**: HVAC energy estimation using a dynamic hybrid fusion mechanism.

## Physics Constraints
We explicitly implement three types of physical constraints as discussed in Section 2.3.1:
- **SP (Structural Physics)**: Graph topology mapped from office layouts.
- **PL (Physical Loss)**: Directionality constraint for heat transfer.
- **MC (Multi-step Correction)**: Inertia-based temporal smoothing.

## Requirements
- Python 3.8+
- PyTorch 1.10+
- NumPy
