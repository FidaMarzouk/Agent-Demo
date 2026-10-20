"""
The agent's tools: deterministic calculations over src/data/energy_data.json.

Each tool is a plain function traced as a DeepEval tool span
(@observe(type="tool")), which is what the trajectory evaluation and the
loop/budget guards read. LANGCHAIN_TOOLS wraps the same functions for the
LangGraph agent.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from deepeval.tracing import observe
from langchain_core.tools import StructuredTool

DATA_FILE = Path(__file__).resolve().parent / "data" / "energy_data.json"
HOURS_PER_YEAR = 8760

# kWh per unit
_ENERGY_UNITS = {"kwh": 1.0, "mwh": 1_000.0, "gwh": 1_000_000.0, "twh": 1_000_000_000.0, "gj": 1 / 0.0036}


@lru_cache(maxsize=1)
def _technologies() -> dict[str, dict[str, Any]]:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))["technologies"]


def _technology_key(technology: str) -> str | None:
    key = re.sub(r"[^a-z]+", "_", technology.strip().lower()).strip("_")
    aliases = {"solar": "solar_pv", "pv": "solar_pv", "wind": "onshore_wind", "gas": "natural_gas",
               "hydro": "hydropower", "geothermal_power": "geothermal", "nuclear_power": "nuclear"}
    key = aliases.get(key, key)
    return key if key in _technologies() else None


@observe(type="tool")
def get_technology_facts(technology: str) -> dict[str, Any]:
    """Typical capacity factor (range and typical value), lifetime in years and
    lifecycle CO2 emissions (g/kWh) for an electricity generation technology:
    solar_pv, onshore_wind, offshore_wind, hydropower, geothermal, nuclear,
    natural_gas or coal."""
    key = _technology_key(technology)
    if key is None:
        return {"error": f"unknown technology '{technology}'", "known_technologies": sorted(_technologies())}
    return {"technology": key, **_technologies()[key]}


@observe(type="tool")
def estimate_annual_generation(capacity_mw: float, capacity_factor: float) -> dict[str, Any]:
    """Electricity a plant produces in a year, in MWh, from its capacity (MW)
    and capacity factor (a fraction between 0 and 1, e.g. 0.35)."""
    capacity_mw, capacity_factor = float(capacity_mw), float(capacity_factor)
    if capacity_factor > 1:  # tolerate percentages
        capacity_factor /= 100
    if not 0 < capacity_factor <= 1 or capacity_mw <= 0:
        return {"error": "capacity_mw must be > 0 and capacity_factor between 0 and 1"}
    return {"mwh_per_year": round(capacity_mw * capacity_factor * HOURS_PER_YEAR, 2)}


@observe(type="tool")
def convert_energy(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
    """Convert an amount of energy between kWh, MWh, GWh, TWh and GJ."""
    src, dst = from_unit.strip().lower(), to_unit.strip().lower()
    if src not in _ENERGY_UNITS or dst not in _ENERGY_UNITS:
        return {"error": f"unsupported unit; supported: {sorted(_ENERGY_UNITS)}"}
    converted = float(value) * _ENERGY_UNITS[src] / _ENERGY_UNITS[dst]
    return {"value": round(converted, 6), "unit": to_unit}


@observe(type="tool")
def estimate_co2_avoided(energy_mwh: float, replaced_technology: str, new_technology: str) -> dict[str, Any]:
    """Tonnes of lifecycle CO2 avoided when `energy_mwh` of electricity from
    replaced_technology is produced by new_technology instead."""
    old, new = _technology_key(replaced_technology), _technology_key(new_technology)
    if old is None or new is None:
        return {"error": "unknown technology", "known_technologies": sorted(_technologies())}
    g_old = _technologies()[old]["lifecycle_gco2_per_kwh"]
    g_new = _technologies()[new]["lifecycle_gco2_per_kwh"]
    # MWh * 1000 kWh/MWh * g/kWh / 1e6 g/t = MWh * g/kWh / 1000
    return {"tonnes_co2_avoided": round(float(energy_mwh) * (g_old - g_new) / 1000, 2),
            "replaced_gco2_per_kwh": g_old, "new_gco2_per_kwh": g_new}


TOOL_FUNCTIONS = {
    f.__name__: f for f in (get_technology_facts, estimate_annual_generation, convert_energy, estimate_co2_avoided)
}

LANGCHAIN_TOOLS = [
    StructuredTool.from_function(func=f, name=name, description=(f.__doc__ or "").strip())
    for name, f in TOOL_FUNCTIONS.items()
]
