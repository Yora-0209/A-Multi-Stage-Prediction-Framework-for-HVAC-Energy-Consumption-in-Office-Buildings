"""
HVAC Energy Prediction via Thermal Load Calculation and Hybrid Fusion

This module implements Stage 2 (Thermal Load Calculation) and Stage 3 (HVAC Energy Estimation)
of the multi-stage framework described in the associated paper.

Stage 2: Solve heat-balance equation (Eq. 3) to obtain zone-level cooling/heating loads.
Stage 3: Fuse physics-based conversion and data-driven prediction with dynamic calibration.

The code is intended for methodological review and is not executable without the
proprietary dataset and environment configuration.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class ThermalLoadCalculator:
    """
    Calculates zone-level thermal loads based on the first law of thermodynamics.
    Uses building physical parameters as described in Table 1 and Supplementary S3.
    Zones 1-4: interior zones, Zones 5-8: exterior zones.
    """
    def __init__(self, num_zones: int = 8, building_params: Optional[Dict] = None):
        self.num_zones = num_zones

        # Default parameters (should be overridden with actual building data from Table 1)
        default_params = {
            "thermal_capacitance": [3.5e7] * 4 + [3.2e7] * 4,      # J/K (interior + exterior)
            "zone_conductance": np.ones((num_zones, num_zones)) * 0.5,  # W/K
            "envelope_conductance": [0.0380] * 4 + [0.0096, 0.0100, 0.0167, 0.0105],  # W/K (Table 1)
            "solar_gain_coef": [0.018] * 4 + [0.121, 0.102, 0.075, 0.102],            # Table 1
            "internal_gain_coef": [22.0] * num_zones,               # W/m2
            "zone_area": [284, 405, 262, 260, 230, 238, 211, 213],  # m2 (Table 1)
        }
        self.params = building_params if building_params is not None else default_params

    def calculate_loads(
        self,
        zone_temps: np.ndarray,           # [num_zones]
        outdoor_temp: float,
        solar_radiation: float,
        occupancy: float = 1.0,
        setpoints: Optional[np.ndarray] = None,
        timestep: float = 0.25,           # hours (15 min)
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute net thermal load (Q_HVAC,i) for each zone (Eq. 3).
        Returns loads (positive = heating demand, negative = cooling demand) and component breakdown.
        """
        loads = np.zeros(self.num_zones)
        components = {
            "envelope": np.zeros(self.num_zones),
            "interzonal": np.zeros(self.num_zones),
            "solar": np.zeros(self.num_zones),
            "internal": np.zeros(self.num_zones),
        }

        # Inter-zonal heat transfer (Q_ij)
        for i in range(self.num_zones):
            for j in range(self.num_zones):
                if i != j:
                    heat_flow = self.params["zone_conductance"][i, j] * (zone_temps[j] - zone_temps[i])
                    components["interzonal"][i] += heat_flow

        # Envelope and solar gains
        for i in range(self.num_zones):
            components["envelope"][i] = self.params["envelope_conductance"][i] * (outdoor_temp - zone_temps[i])
            components["solar"][i] = solar_radiation * self.params["solar_gain_coef"][i] * self.params["zone_area"][i]
            components["internal"][i] = self.params["internal_gain_coef"][i] * self.params["zone_area"][i] * occupancy

        # Total load: HVAC must supply the opposite to maintain thermal balance
        for i in range(self.num_zones):
            loads[i] = -(components["envelope"][i] + components["interzonal"][i] +
                        components["solar"][i] + components["internal"][i])

        return loads, components


class HVACEnergyEstimator:
    """
    Converts zone thermal loads to HVAC energy consumption using equipment performance curves.
    Includes COP calculations and RTU capacity constraints (Supplementary S4).
    """
    def __init__(self, rtu_params: Optional[Dict] = None):
        default_rtu = {
            "cooling_cop": 2.8,
            "heating_cop": 2.5,
            "rtu_cooling_capacity": [105.5] * 4,  # kW (Table A2)
            "rtu_heating_capacity": [105.5] * 4,
            "rtu_rated_power": [20.5] * 4,        # kW
            "rtu_to_zones": {0: [0, 4], 1: [1, 5], 2: [2, 6], 3: [3, 7]},
        }
        self.params = rtu_params if rtu_params is not None else default_rtu

    def estimate_energy(
        self,
        thermal_loads: np.ndarray,  # [num_zones]
        zone_temps: np.ndarray,     # [num_zones]
        outdoor_temp: float,
        timestep: float = 0.25,
    ) -> Dict:
        """
        Physics-based energy estimation (E_phy in Eq. 14).
        Returns per-RTU energy consumption in kWh.
        """
        num_zones = len(thermal_loads)
        cooling_power = np.zeros(num_zones)
        heating_power = np.zeros(num_zones)
        fan_power = np.zeros(num_zones)

        for i in range(num_zones):
            if thermal_loads[i] < -10:  # cooling demand
                load = abs(thermal_loads[i])
                cop = self.params["cooling_cop"]
                cooling_power[i] = load / cop
                fan_power[i] = 0.2 * self.params["rtu_rated_power"][i // 2]
            elif thermal_loads[i] > 10:  # heating demand
                load = thermal_loads[i]
                cop = self.params["heating_cop"]
                heating_power[i] = load / cop
                fan_power[i] = 0.15 * self.params["rtu_rated_power"][i // 2]

        rtu_energy = np.zeros(4)
        for rtu_idx, zones in self.params["rtu_to_zones"].items():
            total_power = 0.0
            for z in zones:
                total_power += cooling_power[z] + heating_power[z] + fan_power[z]
            max_power = self.params["rtu_cooling_capacity"][rtu_idx] * 0.9
            total_power = min(total_power, max_power)
            rtu_energy[rtu_idx] = total_power * timestep / 1000.0  # kWh

        return {
            "rtu_energy": rtu_energy,
            "total_energy": np.sum(rtu_energy),
            "cooling": np.sum(cooling_power) * timestep / 1000.0,
            "heating": np.sum(heating_power) * timestep / 1000.0,
        }


class HybridEnergyPredictor:
    """
    Dynamic fusion of physics-based and data-driven energy predictions (Eqs. 14-17).
    Uses a moving window of recent performance to adapt fusion weights alpha_k^t.
    """
    def __init__(
        self,
        thermal_calc: ThermalLoadCalculator,
        energy_estimator: HVACEnergyEstimator,
        dl_model: Optional[nn.Module] = None,  # Placeholder for deep learning model
        window_size: int = 24,
    ):
        self.thermal_calc = thermal_calc
        self.energy_estimator = energy_estimator
        self.dl_model = dl_model
        self.window_size = window_size

        self.phy_errors = {k: [] for k in range(4)}
        self.dl_errors = {k: [] for k in range(4)}
        self.epsilon = 1e-6

    def predict_step(
        self,
        temp_pred: np.ndarray,       # [num_zones]
        outdoor_temp: float,
        solar_rad: float,
        occupancy: float,
        timestep: float = 0.25,
    ) -> Dict:
        """
        One-step energy prediction with dynamic fusion.
        """
        # Physics-based path
        loads, _ = self.thermal_calc.calculate_loads(temp_pred, outdoor_temp, solar_rad, occupancy)
        phy_result = self.energy_estimator.estimate_energy(loads, temp_pred, outdoor_temp, timestep)
        E_phy = phy_result["rtu_energy"]

        # Data-driven path (placeholder)
        if self.dl_model is not None:
            # E_dl = self.dl_model.predict(...)
            E_dl = E_phy.copy()
        else:
            E_dl = E_phy.copy()

        # Dynamic fusion weights (Eqs. 15-17)
        fused = np.zeros(4)
        for k in range(4):
            P_phy = 1.0 / (self._get_mae(self.phy_errors[k]) + self.epsilon)
            P_dl = 1.0 / (self._get_mae(self.dl_errors[k]) + self.epsilon)
            alpha = P_phy / (P_phy + P_dl)
            fused[k] = alpha * E_phy[k] + (1 - alpha) * E_dl[k]

        return {
            "rtu_energy": fused,
            "total_energy": np.sum(fused),
            "phy_energy": E_phy,
            "dl_energy": E_dl,
            "fusion_weights": [alpha],
        }

    def update_performance(self, rtu: int, pred: float, actual: float, source: str):
        """Update error history for calibration."""
        error = abs(pred - actual)
        if source == "phy":
            self.phy_errors[rtu].append(error)
        else:
            self.dl_errors[rtu].append(error)
        if len(self.phy_errors[rtu]) > self.window_size:
            self.phy_errors[rtu].pop(0)
        if len(self.dl_errors[rtu]) > self.window_size:
            self.dl_errors[rtu].pop(0)

    def _get_mae(self, errors):
        return np.mean(errors) if errors else 1.0


# ---------------------------------------------------------------------
# Scenario assessment example (pseudo-code)
# ---------------------------------------------------------------------
def run_scenario_assessment_example():
    """
    Demonstrates how the framework updates only spatial parameters (occupant density,
    equipment power density) to compare layout scenarios (Baseline, Optimized A, B).
    """
    thermal_calc = ThermalLoadCalculator()
    energy_est = HVACEnergyEstimator()
    hybrid = HybridEnergyPredictor(thermal_calc, energy_est)

    # Example predicted temperatures from PCGNN for one timestep
    temp_pred = np.array([23.0, 23.2, 22.8, 23.1, 24.5, 24.8, 25.0, 24.7])
    outdoor_temp = 30.0
    solar_rad = 600.0

    # Baseline scenario
    result_base = hybrid.predict_step(temp_pred, outdoor_temp, solar_rad, occupancy=1.0)

    # Scenario A (updated internal gain coefficients would be applied in thermal_calc)
    result_A = hybrid.predict_step(temp_pred, outdoor_temp, solar_rad, occupancy=0.9)

    print(f"Baseline total energy: {result_base['total_energy']:.2f} kWh")
    print(f"Scenario A total energy: {result_A['total_energy']:.2f} kWh")