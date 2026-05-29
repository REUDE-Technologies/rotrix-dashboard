# type: ignore
"""
Pure Python calculators for RotriDASH hub.

Quick estimates use explicit assumptions documented in return dicts.
BEMT / blade-element path follows the RotriX Excel workbook structure
(Drone_Propeller_Blade_Design_sheet.xlsx): segment geometry, relative
velocity, and thrust/torque/power aggregation with Excel-matched
non-dimensional coefficients (same exponents as sheet formulas).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Workbook uses 3.14 for pi in angular velocity columns.
PI_SHEET = 3.14
GAMMA_AIR = 1.4
R_GAS_AIR = 287.0
MU_AIR_PA_S = 1.81e-5  # dynamic viscosity at ~15 °C, for Reynolds number
WORKBOOK_NAME = "Drone_Propeller_Blade_Design_sheet.xlsx"


def air_density_from_temperature_k(temp_k: float, pressure_pa: float = 101325.0) -> float:
    """Ideal-gas density (kg/m³) when workbook density cells are filled from temperature."""
    if temp_k <= 0:
        return float("nan")
    return pressure_pa / (R_GAS_AIR * temp_k)


def chord_hub_from_prop_diameter(prop_diameter_m: float) -> float:
    """Workbook hint: chord at hub ≈ prop diameter / 12.6."""
    return prop_diameter_m / 12.6


def chord_tip_from_hub_chord(chord_hub_m: float) -> float:
    """Workbook hint: chord at tip ≈ hub chord × 0.66667."""
    return chord_hub_m * 0.66667


def estimate_static_thrust_momentum(
    *,
    rho_kg_m3: float,
    rpm: float,
    diameter_m: float,
    thrust_coefficient: float = 0.08,
) -> dict[str, Any]:
    """
    Order-of-magnitude static thrust: T ≈ C_T * ρ * n² * D⁴ (n in rev/s).

    C_T is not universal; treat as a tunable lumped parameter (default 0.08).
    """
    if rho_kg_m3 <= 0 or diameter_m <= 0 or rpm <= 0:
        return {"ok": False, "error": "ρ, D, and RPM must be positive."}
    if thrust_coefficient <= 0:
        return {"ok": False, "error": "Thrust coefficient must be positive."}
    n = rpm / 60.0
    t_n = thrust_coefficient * rho_kg_m3 * (n**2) * (diameter_m**4)
    return {
        "ok": True,
        "thrust_n": t_n,
        "thrust_kgf": t_n / 9.80665,
        "n_rev_s": n,
        "note": "Momentum-style scaling; C_T is a lumped tune parameter, not a measured prop C_T.",
    }


def propeller_tip_speed_m_s(*, diameter_m: float, rpm: float) -> dict[str, Any]:
    if diameter_m <= 0 or rpm < 0:
        return {"ok": False, "error": "D must be positive and RPM non-negative."}
    omega = 2.0 * math.pi * (rpm / 60.0)
    v_tip = omega * (diameter_m / 2.0)
    return {"ok": True, "tip_speed_m_s": v_tip, "omega_rad_s": omega}


def motor_kv_rpm(*, kv_rpm_per_v: float, voltage_v: float) -> dict[str, Any]:
    """Ideal no-load RPM ≈ KV * V (ignores Io, iron losses)."""
    if kv_rpm_per_v <= 0 or voltage_v <= 0:
        return {"ok": False, "error": "KV and voltage must be positive."}
    rpm = kv_rpm_per_v * voltage_v
    return {
        "ok": True,
        "rpm_no_load": rpm,
        "note": "No-load ideal; loaded RPM is lower (I·R drop, back-EMF).",
    }


def battery_flight_time_minutes(
    *,
    capacity_mah: float,
    avg_current_a: float,
    usable_fraction: float = 0.8,
) -> dict[str, Any]:
    if capacity_mah <= 0 or avg_current_a <= 0:
        return {"ok": False, "error": "Capacity and average current must be positive."}
    if not (0.0 < usable_fraction <= 1.0):
        return {"ok": False, "error": "Usable fraction must be in (0, 1]."}
    cap_ah = (capacity_mah / 1000.0) * usable_fraction
    hours = cap_ah / avg_current_a
    return {
        "ok": True,
        "time_min": hours * 60.0,
        "time_h": hours,
        "note": "Hover-average current model; real missions vary with throttle and sag.",
    }


def battery_c_rate(*, current_a: float, capacity_mah: float) -> dict[str, Any]:
    if capacity_mah <= 0 or current_a < 0:
        return {"ok": False, "error": "Capacity must be positive; current non-negative."}
    cap_ah = capacity_mah / 1000.0
    c_rate = current_a / cap_ah if cap_ah > 0 else float("inf")
    return {"ok": True, "c_rate": c_rate}


# DC resistance ohms per meter at 20 °C (approximate, single-strand copper).
# Simplified from standard tables; sufficient for drop screening.
_AWG_OHM_PER_M: dict[int, float] = {
    10: 0.00338,
    12: 0.0053,
    14: 0.00844,
    16: 0.0134,
    18: 0.0211,
    20: 0.0336,
    22: 0.053,
    24: 0.084,
    26: 0.133,
    28: 0.213,
}

AWG_TABLE_SIZES: tuple[int, ...] = tuple(sorted(_AWG_OHM_PER_M.keys()))


def wire_voltage_drop_awg(
    *,
    current_a: float,
    length_m_one_way: float,
    awg: int,
) -> dict[str, Any]:
    """Round-trip drop ≈ I * R * (2 * length)."""
    if current_a < 0 or length_m_one_way < 0:
        return {"ok": False, "error": "Current and length must be non-negative."}
    r = _AWG_OHM_PER_M.get(awg)
    if r is None:
        return {"ok": False, "error": f"AWG {awg} not in built-in table (10–28)."}
    r_total = r * (2.0 * length_m_one_way)
    v_drop = current_a * r_total
    return {
        "ok": True,
        "v_drop_v": v_drop,
        "r_total_ohm": r_total,
        "note": "Copper ~20 °C, single conductor; parallel paths and connectors add drop.",
    }


def suggest_awg_for_drop(
    *,
    current_a: float,
    length_m_one_way: float,
    pack_voltage_v: float,
    max_drop_percent: float = 3.0,
) -> dict[str, Any]:
    if current_a <= 0 or length_m_one_way <= 0 or pack_voltage_v <= 0:
        return {"ok": False, "error": "Current, length, and pack voltage must be positive."}
    if not (0.0 < max_drop_percent <= 20.0):
        return {"ok": False, "error": "max_drop_percent should be between 0 and 20."}
    v_budget = pack_voltage_v * (max_drop_percent / 100.0)
    best: int | None = None
    best_drop: float | None = None
    for awg in sorted(_AWG_OHM_PER_M.keys()):
        res = wire_voltage_drop_awg(
            current_a=current_a, length_m_one_way=length_m_one_way, awg=awg
        )
        if not res["ok"]:
            continue
        vd = float(res["v_drop_v"])
        if vd <= v_budget:
            best = awg
            best_drop = vd
            break
    if best is None:
        return {
            "ok": False,
            "error": "No AWG in table meets drop budget; shorten leads or raise budget.",
            "v_budget_v": v_budget,
        }
    return {
        "ok": True,
        "awg": best,
        "v_drop_v": best_drop,
        "v_budget_v": v_budget,
    }


def _blade_performance_row(
    *,
    thrust_n: float,
    torque_nm: float,
    power_w: float,
    rho: float,
    n_rev_s: float,
    diameter_m: float,
    flight_speed_m_s: float,
) -> dict[str, float]:
    """Blade performance summary on sheet `Blade iteration 1` (lab or 200 m altitude)."""
    d = diameter_m
    cp = power_w / (rho * (n_rev_s**3) * (d**5)) if rho > 0 and n_rev_s > 0 else float("nan")
    ct = thrust_n / (rho * (n_rev_s**2) * (d**4)) if rho > 0 and n_rev_s > 0 else float("nan")
    cq = torque_nm / (rho * (n_rev_s**2) * (d**5)) if rho > 0 and n_rev_s > 0 else float("nan")
    j = flight_speed_m_s / (n_rev_s * d) if n_rev_s > 0 else float("nan")
    eta = (j * ct) / cp if cp and cp == cp and cp != 0 else float("nan")
    return {
        "Total Thrust (N)": thrust_n,
        "Total Torque (N·m)": torque_nm,
        "Total Power (W)": power_w,
        "Power Coefficient (Cp)": cp,
        "Thrust Coefficient (CT)": ct,
        "Torque Coefficient (Cq)": cq,
        "Advance ratio (J)": j,
        "Propeller efficiency (η)": eta,
    }


def bemt_blade_performance(
    *,
    rho_lab_kg_m3: float,
    rho_alt_kg_m3: float,
    rpm: float,
    prop_diameter_m: float,
    hub_diameter_m: float,
    temp_lab_k: float,
    temp_alt_k: float,
    gas_constant_j_kg_k: float = R_GAS_AIR,
    beta_tip_deg: float,
    beta_hub_deg: float,
    chord_hub_m: float,
    chord_tip_m: float,
    n_blades: int,
    flight_speed_m_s: float,
    n_segments: int = 10,
    alpha_induced_rad: float = 0.0,
    alpha_zero_lift_deg: list[float] | None = None,
    cl_per_segment: list[float] | None = None,
    cd_per_segment: list[float] | None = None,
) -> dict[str, Any]:
    """
    Blade-element aggregation matching workbook layout (10 segments default).

    Uses cos/sin of (helix_deg + alpha_i_deg) in radians for element orientation,
    which matches the intent of the sheet's COS(E+G) / SIN(E+G) when angles
    are interpreted in degrees for the cosine argument.
    """
    if rho_lab_kg_m3 <= 0 or rho_alt_kg_m3 <= 0:
        return {"ok": False, "error": "Densities must be positive."}
    if prop_diameter_m <= 0 or hub_diameter_m < 0 or hub_diameter_m >= prop_diameter_m:
        return {"ok": False, "error": "Require 0 ≤ d_hub < D."}
    if rpm <= 0 or n_blades < 1 or n_segments < 2:
        return {"ok": False, "error": "RPM > 0, at least 1 blade, ≥2 segments."}
    if temp_lab_k <= 0 or temp_alt_k <= 0:
        return {"ok": False, "error": "Temperatures (K) must be positive."}

    r_tip = prop_diameter_m / 2.0
    r_hub = hub_diameter_m / 2.0
    span = r_tip - r_hub
    dr = (span / n_segments) if n_segments else 0.0
    if dr <= 0:
        return {"ok": False, "error": "Invalid segment width."}

    n_rev_s = rpm / 60.0
    omega = (2.0 * PI_SHEET * rpm) / 60.0

    a_sound_lab = math.sqrt(GAMMA_AIR * gas_constant_j_kg_k * temp_lab_k)
    a_sound_alt = math.sqrt(GAMMA_AIR * gas_constant_j_kg_k * temp_alt_k)

    azl = alpha_zero_lift_deg or [0.0] * n_segments
    cls = cl_per_segment or [0.5] * n_segments
    cds = cd_per_segment or [0.05] * n_segments
    if len(azl) != n_segments:
        return {"ok": False, "error": f"alpha_zero_lift_deg length must be {n_segments}."}
    if len(cls) != n_segments or len(cds) != n_segments:
        return {"ok": False, "error": f"CL/CD lists must have length {n_segments}."}

    alpha_i_deg = math.degrees(alpha_induced_rad)

    def one_density(rho: float, a_sound: float) -> tuple[list[dict], list[dict], list[dict], dict[str, float]]:
        geometry_rows: list[dict] = []
        airfoil_rows: list[dict] = []
        blade_rows: list[dict] = []
        r = r_hub + dr / 2.0
        sum_t = sum_q = sum_p = 0.0

        for i in range(n_segments):
            seg_no = i + 1
            # Sheet: Chord(r) = C14 + ((C15-C14)/Tip radius)*r
            chord = chord_hub_m + ((chord_tip_m - chord_hub_m) / r_tip) * r
            d_area = chord * dr
            w = omega * r
            v_rot = w
            v_res = math.hypot(flight_speed_m_s, w)
            helix_rad = math.atan(v_res / w) if w > 0 else 0.0
            helix_deg = math.degrees(helix_rad)
            x_norm = r / r_tip if r_tip > 0 else 0.0
            mach = v_res / a_sound if a_sound > 0 else float("nan")

            if r_tip > r_hub:
                beta_geom = beta_hub_deg + (beta_tip_deg - beta_hub_deg) * (r - r_hub) / (r_tip - r_hub)
            else:
                beta_geom = beta_hub_deg

            # Sheet: AOA = β_geom − αi − φ − α0
            aoa_deg = beta_geom - alpha_i_deg - helix_deg - azl[i]
            reynolds = rho * v_res * chord / MU_AIR_PA_S if MU_AIR_PA_S > 0 else float("nan")

            cl_i = cls[i]
            cd_i = cds[i]
            q = 0.5 * rho * (v_res**2) * chord * dr
            dL = q * cl_i
            dD = q * cd_i

            # Sheet: COS/SIN(φ_deg + αi_deg) in degrees
            ph_rad = math.radians(helix_deg + alpha_i_deg)
            c_ph = math.cos(ph_rad)
            s_ph = math.sin(ph_rad)
            dT = dL * c_ph - dD * s_ph
            dQ = (dL * s_ph + dD * c_ph) * r
            dP = (dL * s_ph + dD * c_ph) * w

            geometry_rows.append(
                {
                    "Segment": seg_no,
                    "Prop diameter (m)": prop_diameter_m,
                    "Hub diameter (m)": hub_diameter_m,
                    "n (rev/s)": n_rev_s,
                    "ω (rad/s)": omega,
                    "Hub radius (m)": r_hub,
                    "Tip radius (m)": r_tip,
                    "dr (m)": dr,
                    "r (m)": r,
                    "x = r / tip": x_norm,
                    "Chord (m)": chord,
                    "dA (m²)": d_area,
                    "Flight speed (m/s)": flight_speed_m_s,
                    "Speed of sound (m/s)": a_sound,
                    "ωr (m/s)": v_rot,
                    "Resultant speed (m/s)": v_res,
                    "Mach no.": mach,
                }
            )
            airfoil_rows.append(
                {
                    "Segment": seg_no,
                    "Helix φ (deg)": helix_deg,
                    "Induced αi (deg)": alpha_i_deg,
                    "Geometric pitch β (deg)": beta_geom,
                    "α zero-lift (deg)": azl[i],
                    "AOA α (deg)": aoa_deg,
                    "Reynolds number": reynolds,
                }
            )
            blade_rows.append(
                {
                    "Segment": seg_no,
                    "CL": cl_i,
                    "CD": cd_i,
                    "dL (N)": dL,
                    "dD (N)": dD,
                    "dT (N)": dT,
                    "dQ (N·m)": dQ,
                    "dP (W)": dP,
                }
            )
            sum_t += dT
            sum_q += dQ
            sum_p += dP
            r += dr

        thrust = n_blades * sum_t
        torque = n_blades * sum_q
        power = n_blades * sum_p
        perf = _blade_performance_row(
            thrust_n=thrust,
            torque_nm=torque,
            power_w=power,
            rho=rho,
            n_rev_s=n_rev_s,
            diameter_m=prop_diameter_m,
            flight_speed_m_s=flight_speed_m_s,
        )
        return geometry_rows, airfoil_rows, blade_rows, perf

    geom_lab, air_lab, blade_lab, perf_lab = one_density(rho_lab_kg_m3, a_sound_lab)
    geom_alt, air_alt, blade_alt, perf_alt = one_density(rho_alt_kg_m3, a_sound_alt)

    return {
        "ok": True,
        "workbook": WORKBOOK_NAME,
        "sheets": ["propeller calculations", "Airfoil properties", "Blade iteration 1"],
        "geometry_table": geom_lab,
        "geometry_table_alt": geom_alt,
        "airfoil_table": air_lab,
        "airfoil_table_alt": air_alt,
        "blade_elemental_lab": blade_lab,
        "blade_elemental_alt": blade_alt,
        "blade_performance": {
            "lab_conditions": perf_lab,
            "altitude_200m": perf_alt,
        },
        "meta": {
            "n_rev_s": n_rev_s,
            "omega_rad_s": omega,
            "a_sound_lab_m_s": a_sound_lab,
            "a_sound_alt_m_s": a_sound_alt,
            "n_segments": n_segments,
            "n_blades": n_blades,
            "rho_lab_kg_m3": rho_lab_kg_m3,
            "rho_alt_kg_m3": rho_alt_kg_m3,
        },
        # Legacy keys for any older callers
        "segments_lab": blade_lab,
        "segments_alt": blade_alt,
        "aggregate_lab": perf_lab,
        "aggregate_alt": perf_alt,
    }
