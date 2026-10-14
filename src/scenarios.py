"""Derive sensitivity scenarios from a base ``InputData`` without touching the JSON files.

Each helper returns a *new* ``InputData`` (the base data is never modified), so scenarios can
be chained::

    from src.data_loader import load_question
    from src.scenarios import scale_prices, set_tariffs, scale_pv

    base = load_question("Q1_caseA")
    high_spread = scale_prices(base, factor=2.0, keep_mean=True)      # same mean, doubled spread
    no_export_tariff = set_tariffs(high_spread, export_tariff=0.0)

Add your own helpers here (e.g. shifting the price peak towards the PV peak, changing the
minimum daily energy or the reference profile) and document them in the README.
"""
from __future__ import annotations

from dataclasses import replace

import copy
import numpy as np
import pandas as pd
from src.model import QuadraticConsumerModel


from .data_loader import InputData


def scale_prices(data: InputData, factor: float, keep_mean: bool = False) -> InputData:
    """Multiply hourly prices by ``factor``. With ``keep_mean=True`` only the spread around the
    mean is scaled, which isolates the effect of price variability from the effect of price level."""
    p = data.energy_price
    new_p = p.mean() + factor * (p - p.mean()) if keep_mean else factor * p
    return replace(data, energy_price=np.clip(new_p, 0.0, None))   # prices stay non-negative


def shift_prices(data: InputData, delta: float) -> InputData:
    """Add ``delta`` DKK/kWh to every hourly price (changes the mean, keeps the spread)."""
    return replace(data, energy_price=np.clip(data.energy_price + delta, 0.0, None))


def roll_prices(data: InputData, hours: int) -> InputData:
    """Shift the price profile in time by ``hours`` (positive = later). Useful to study the
    temporal correlation between the price peak and the PV peak."""
    return replace(data, energy_price=np.roll(data.energy_price, hours))


def set_tariffs(data: InputData, import_tariff: float | None = None, export_tariff: float | None = None) -> InputData:
    """Override the import and/or export grid tariff (DKK/kWh)."""
    return replace(
        data,
        import_tariff=data.import_tariff if import_tariff is None else import_tariff,
        export_tariff=data.export_tariff if export_tariff is None else export_tariff,
    )


def scale_pv(data: InputData, factor: float) -> InputData:
    """Scale the available PV production (e.g. 0.5 for a cloudy day, 0.0 for no PV)."""
    profile = np.clip(data.pv_profile * factor, 0.0, 1.0)
    return replace(data, pv_profile=profile, pv_available=data.pv_max_kW * profile)


def set_load_preferences(
    data: InputData,
    load_max_kWh: float | None = None,
    load_min_kWh: float | None = None,
    min_daily_energy_kWh: float | None = None,
) -> InputData:
    """Override the hourly load bounds and/or the minimum daily energy requirement."""
    return replace(
        data,
        load_max_kWh=data.load_max_kWh if load_max_kWh is None else load_max_kWh,
        load_min_kWh=data.load_min_kWh if load_min_kWh is None else load_min_kWh,
        min_daily_energy_kWh=data.min_daily_energy_kWh if min_daily_energy_kWh is None else min_daily_energy_kWh,
    )


def run_cQ_sweep(base_data, cQ_values, tol=1e-5):
    """
    Run a parameter sweep over the quadratic disutility coefficient c^Q.
    
    Inputs:
        base_data (InputData): Base input data for Q2_quadratic.
        cQ_values (list of float): Values of c^Q [DKK/kWh^2] to evaluate.
        tol (float): Numerical tolerance to check if load bounds are binding.
    Outputs:
        pd.DataFrame: Summary table of daily metrics for each c^Q value.
    """
    rows = []
    p_imp = base_data.energy_price + base_data.import_tariff
    p_exp = base_data.energy_price - base_data.export_tariff
    l_ref = base_data.reference_load

    for cQ in cQ_values:
        d_scen = copy.deepcopy(base_data)
        d_scen.quadratic_disutility = float(cQ)
        
        res = QuadraticConsumerModel(d_scen).build().solve()
        df_h = res.hourly
        
        l_opt  = df_h["load"].to_numpy()
        g_imp  = df_h["import"].to_numpy()
        g_exp  = df_h["export"].to_numpy()
        pv_opt = df_h["pv"].to_numpy()
        
        # Compute required daily metrics
        procurement_cost = np.sum(p_imp * g_imp - p_exp * g_exp + d_scen.pv_marginal_cost * pv_opt)
        total_disutility = np.sum(cQ * (l_opt - l_ref) ** 2)
        daily_energy     = np.sum(l_opt)
        total_abs_dev    = np.sum(np.abs(l_opt - l_ref))
        
        # Count hours where load_min_kWh or load_max_kWh is binding
        binding_min = np.abs(l_opt - d_scen.load_min_kWh) <= tol
        binding_max = np.abs(l_opt - d_scen.load_max_kWh) <= tol
        n_binding_hours = int(np.sum(binding_min | binding_max))
        
        rows.append({
            "c^Q [DKK/kWh^2]": cQ,
            "Procurement Cost [DKK]": round(procurement_cost, 3),
            "Total Disutility [DKK]": round(total_disutility, 3),
            "Total Objective [DKK]": round(procurement_cost + total_disutility, 3),
            "Daily Energy [kWh]": round(daily_energy, 3),
            "Total Abs. Deviation [kWh]": round(total_abs_dev, 3),
            "Binding Hours [h]": n_binding_hours
        })
        
    return pd.DataFrame(rows)