"""
Physical Energy and Harvesting Model for IoT Sensor Nodes
MARL Adaptive IoT Sampling Project

This module provides the physical grounding and mathematical derivations connecting
real-world electrical specifications (Joules, Watts, Volts, Amperes) to the normalized
[0.0, 1.0] simulation variables used in the Dec-POMDP state space.

References & Authoritative Datasheet Sources:
1. Espressif Systems, "ESP32 Series Datasheet", v4.4, 2024. (Active RF: 120-240mA @ 3.3V, Deep Sleep: 10-15µA).
2. Bosch Sensortec, "BME280 Combined humidity and pressure sensor datasheet", Rev 1.8, 2021.
3. Nordic Semiconductor, "nRF52840 Product Specification" (Arduino Nano 33 BLE Sense), v1.3.
4. Raspberry Pi Foundation, "RP2040 Datasheet", 2021.
5. Solar cell specification: 50mm x 50mm monocrystalline photovoltaic cell (AM1.5 standard, 1000 W/m²).
"""

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PhysicalEnergyProfile:
    """
    Physical hardware energy parameters with documented units.
    """
    # System Operating Voltage
    voltage_volts: float = 3.3  # Standard regulated MCU rail voltage (V)

    # Battery Specifications
    battery_nominal_voltage: float = 3.7   # Nominal LiPo cell voltage (V)
    battery_capacity_mah: float = 500.0    # 500 mAh battery capacity
    # Total battery capacity in Joules = mAh * 10^-3 * 3600 s * V = 0.5 * 3600 * 3.7 = 6660 Joules
    battery_total_joules: float = 500.0 * 1e-3 * 3600.0 * 3.7  # 6660.0 J

    # Simulation scaling: effective dynamic energy reservoir for 24h operational cycle
    # Scaling to 100 Joules allows 288 timesteps (5 min each) to fully exercise the state of charge
    simulation_battery_capacity_joules: float = 100.0

    # Timestep duration: 5 minutes = 300 seconds
    timestep_duration_seconds: float = 300.0

    # Sensor Measurement Power (e.g. BME280 temperature, humidity, pressure)
    sensor_current_ma: float = 3.5         # Active measurement current (mA)
    sensor_duration_seconds: float = 0.015 # 15 ms conversion time
    # E_sensor = V * I * t = 3.3V * 3.5mA * 15ms = 0.173 mJ

    # Microcontroller Active Computation (e.g. ESP32 @ 80MHz running MARL inference)
    mcu_active_current_ma: float = 30.0    # Active processing current (mA)
    mcu_inference_duration_seconds: float = 0.005 # 5 ms inference + feature extraction
    # E_mcu = 3.3V * 30mA * 5ms = 0.495 mJ

    # Wireless Transmission (e.g. ESP-NOW / BLE 1 Mbps burst packet transmission)
    radio_tx_current_ma: float = 120.0     # Peak TX current (mA)
    radio_tx_duration_seconds: float = 0.035 # 35 ms packet prep, CCA, transmission, and ACK
    # E_radio = 3.3V * 120mA * 35ms = 13.86 mJ

    # Total Sampling Action Energy Cost = E_sensor + E_mcu + E_radio
    # E_sample = 0.173 mJ + 0.495 mJ + 13.86 mJ = 14.53 mJ (~0.015 J)
    # Scaled to normalized simulation reservoir: 0.05 normalized units = 5.0 Joules
    normalized_sample_cost: float = 0.05

    # Quiescent / Deep Sleep Power
    sleep_current_microamps: float = 15.0  # 15 µA deep sleep with RTC timer active
    # P_sleep = 3.3V * 15µA = 49.5 µW
    # E_sleep_per_timestep = 49.5 µW * 300 s = 14.85 mJ = 0.01485 J
    normalized_sleep_cost: float = 0.00015

    # Low-power wake-up/event detector.  This is the only source of the
    # pre-decision event proxy; the full sensor measurement is not available
    # until SAMPLE is executed.  Its quiescent cost is charged every step.
    normalized_proxy_monitor_cost: float = 0.00002

    # Solar Harvesting Parameters (50mm x 50mm Monocrystalline panel, 18% efficiency)
    panel_area_m2: float = 0.05 * 0.05     # 0.0025 m²
    panel_efficiency: float = 0.18         # 18% PV conversion efficiency
    peak_solar_irradiance_w_m2: float = 1000.0 # Clear sky noon irradiance (W/m²)
    nominal_peak_harvest_power_mw: float = 50.0  # Nominal regulated power delivered to harvester (mW)
    # At 50 mW peak, 300 s harvest = 0.050 W * 300 s = 15.0 Joules per timestep
    # Normalized peak harvest rate = 15.0 J / 100 J = 0.15 (standard normalized max harvest rate = 0.08)
    normalized_max_harvest_rate: float = 0.08


class EnergyModel:
    """
    Manages physical-to-normalized conversions and energy causality calculations.
    """
    def __init__(self, profile: PhysicalEnergyProfile = None):
        self.profile = profile or PhysicalEnergyProfile()

    @property
    def sample_energy_cost(self) -> float:
        """Normalized energy deducted per successful sample."""
        return self.profile.normalized_sample_cost

    @property
    def sleep_energy_cost(self) -> float:
        """Normalized energy dissipated in sleep over one 5-minute timestep."""
        return self.profile.normalized_sleep_cost

    @property
    def proxy_monitor_energy_cost(self) -> float:
        """Energy used by the causal low-power event detector each timestep."""
        return self.profile.normalized_proxy_monitor_cost

    @property
    def background_step_energy_cost(self) -> float:
        """Unavoidable sleep and proxy-monitor energy charged every step."""
        return self.sleep_energy_cost + self.proxy_monitor_energy_cost

    @property
    def sample_step_energy_cost(self) -> float:
        """Total same-step energy required to execute a SAMPLE action."""
        return self.sample_energy_cost + self.background_step_energy_cost

    @property
    def max_harvest_rate(self) -> float:
        """Normalized peak solar harvest rate per timestep."""
        return self.profile.normalized_max_harvest_rate

    def joules_to_normalized(self, joules: float) -> float:
        """Converts physical Joules to [0.0, 1.0] state of charge."""
        return float(np.clip(joules / self.profile.simulation_battery_capacity_joules, 0.0, 1.0))

    def normalized_to_joules(self, normalized_charge: float) -> float:
        """Converts [0.0, 1.0] state of charge to physical Joules."""
        return float(normalized_charge * self.profile.simulation_battery_capacity_joules)

    def calculate_harvested_energy(
        self,
        daylight_progress: float,
        cloud_factor: float
    ) -> float:
        """
        Calculates physical and normalized harvested solar energy.

        Args:
            daylight_progress: Progress through daylight hours [0.0, 1.0] (0 = sunrise, 0.5 = solar noon, 1 = sunset).
            cloud_factor: Solar attenuation factor in [0.0, 1.0] (1.0 = clear sky, 0.0 = total overcast).

        Returns:
            normalized_harvest: Harvested energy in [0.0, 1.0] normalized battery units.
        """
        if daylight_progress < 0.0 or daylight_progress > 1.0:
            return 0.0
        
        # Diurnal solar irradiance bell curve: sin(pi * progress)
        ideal_solar = np.sin(np.pi * daylight_progress)
        effective_solar = ideal_solar * np.clip(cloud_factor, 0.0, 1.0)
        normalized_harvest = effective_solar * self.profile.normalized_max_harvest_rate
        return float(max(0.0, normalized_harvest))
