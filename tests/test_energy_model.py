"""
Unit Tests: Physical Energy Model
MARL Adaptive IoT Sampling Project
"""

import pytest
import numpy as np
from environment.energy_model import EnergyModel, PhysicalEnergyProfile

def test_energy_conversions():
    """Verify conversion accuracy between physical Joules and normalized battery charge."""
    profile = PhysicalEnergyProfile(simulation_battery_capacity_joules=100.0)
    model = EnergyModel(profile)

    # 50 Joules should map to 0.50 normalized
    assert model.joules_to_normalized(50.0) == pytest.approx(0.50)
    assert model.normalized_to_joules(0.75) == pytest.approx(75.0)

    # Bounds clipping
    assert model.joules_to_normalized(150.0) == 1.0
    assert model.joules_to_normalized(-10.0) == 0.0

def test_diurnal_solar_harvesting():
    """Verify diurnal solar cycle yields peak harvest at midday and zero at night."""
    model = EnergyModel()

    # Solar Noon (progress = 0.50)
    noon_harvest = model.calculate_harvested_energy(daylight_progress=0.50, cloud_factor=1.0)
    assert noon_harvest > 0.0
    assert noon_harvest == pytest.approx(model.max_harvest_rate, abs=1e-4)

    # Sunrise / Sunset (progress = 0.0 or 1.0)
    assert model.calculate_harvested_energy(daylight_progress=0.0, cloud_factor=1.0) == pytest.approx(0.0)
    assert model.calculate_harvested_energy(daylight_progress=1.0, cloud_factor=1.0) == pytest.approx(0.0)

    # Night time (progress < 0 or > 1)
    assert model.calculate_harvested_energy(daylight_progress=-0.2, cloud_factor=1.0) == 0.0
    assert model.calculate_harvested_energy(daylight_progress=1.5, cloud_factor=1.0) == 0.0

    # Overcast attenuation (cloud factor = 0.2)
    overcast_harvest = model.calculate_harvested_energy(daylight_progress=0.50, cloud_factor=0.2)
    assert overcast_harvest == pytest.approx(noon_harvest * 0.2, abs=1e-4)
