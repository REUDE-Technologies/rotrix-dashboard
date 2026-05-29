# type: ignore
"""
RotriDASH calculator hub — organized by drone component, then by purpose.
"""
from __future__ import annotations

import streamlit as st

import calculator_engine as ce

# Component → list of (purpose id, label, short description)
CALCULATOR_TREE: dict[str, list[tuple[str, str, str]]] = {
    "Propeller": [
        (
            "prop_bemt",
            "Propeller sizing (RotriX Excel BEMT)",
            "Full workbook outputs: geometry, airfoil, blade elements, and blade performance (lab + 200 m).",
        ),
    ],
    "Motor": [
        ("motor_kv_rpm", "KV → no-load RPM", "Ideal unloaded RPM from KV rating and pack voltage."),
    ],
    "Battery": [
        ("batt_flight_time", "Flight time", "Endurance from capacity, average current, and usable DoD."),
        ("batt_c_rate", "C-rate", "Discharge rate vs pack capacity."),
    ],
}

COMPONENT_ORDER = ["Propeller", "Motor", "Battery"]


def _back_to_app() -> None:
    st.session_state.show_calculators = False
    st.session_state.show_profile_popover = False
    if st.session_state.get("files_submitted"):
        st.session_state.show_upload_area = False
    else:
        st.session_state.show_upload_area = True
    st.rerun()


def _format_segment_csv(values: list[float]) -> str:
    return ", ".join(f"{v:g}" for v in values)


def _demo_segment_lists(n: int, cl_def: float, cd_def: float) -> tuple[str, str, str]:
    """Editable hub→tip demo polars (10 segments by default)."""
    n = max(int(n), 1)
    azl = [0.0] * n
    cl = [round(cl_def * (0.85 + 0.15 * i / max(n - 1, 1)), 4) for i in range(n)]
    cd = [round(cd_def * (1.0 + 0.25 * i / max(n - 1, 1)), 5) for i in range(n)]
    return (
        _format_segment_csv(azl),
        _format_segment_csv(cl),
        _format_segment_csv(cd),
    )


def _parse_segment_list(txt: str, n: int, default: float) -> list[float]:
    txt = (txt or "").strip()
    if not txt:
        return [default] * n
    parts = [float(x.strip()) for x in txt.split(",") if x.strip()]
    if not parts:
        return [default] * n
    if len(parts) == 1:
        return [parts[0]] * n
    if len(parts) < n:
        return parts + [parts[-1]] * (n - len(parts))
    if len(parts) > n:
        return parts[:n]
    return parts


_DEMO_D = 0.5
_DEMO_CHORD_H = ce.chord_hub_from_prop_diameter(_DEMO_D)
_DEMO_CHORD_T = ce.chord_tip_from_hub_chord(_DEMO_CHORD_H)

# Pre-filled workbook test case (all fields editable).
BEMT_DEMO: dict[str, object] = {
    "bm_auto_rho": False,
    "bm_rhol": 1.225,
    "bm_rhoa": 1.202,
    "bm_rpm": 3000.0,
    "bm_d": _DEMO_D,
    "bm_dh": 0.05,
    "bm_tl": 288.15,
    "bm_ta": 287.15,
    "bm_nseg": 10,
    "bm_bh": 30.0,
    "bm_bt": 15.0,
    "bm_wb_ch": True,
    "bm_ch": _DEMO_CHORD_H,
    "bm_ct": _DEMO_CHORD_T,
    "bm_nb": 2,
    "bm_vf": 5.0,
    "bm_ai": 0.0,
    "bm_cl0": 0.45,
    "bm_cd0": 0.02,
}


def _sync_bemt_segment_fields(n_seg: int, cl_def: float, cd_def: float, *, force: bool = False) -> None:
    prev = st.session_state.get("bm_nseg_sync")
    if force or prev != n_seg:
        azl, cl, cd = _demo_segment_lists(n_seg, cl_def, cd_def)
        st.session_state["bm_azl"] = azl
        st.session_state["bm_cll"] = cl
        st.session_state["bm_cdl"] = cd
        st.session_state["bm_nseg_sync"] = n_seg


def _init_bemt_demo_inputs() -> None:
    if not st.session_state.get("bemt_demo_ready"):
        for key, val in BEMT_DEMO.items():
            st.session_state.setdefault(key, val)
        n = int(st.session_state.get("bm_nseg", 10))
        _sync_bemt_segment_fields(
            n,
            float(st.session_state.get("bm_cl0", 0.45)),
            float(st.session_state.get("bm_cd0", 0.02)),
            force=True,
        )
        st.session_state["bemt_demo_ready"] = True


def _reset_bemt_demo_inputs() -> None:
    for key, val in BEMT_DEMO.items():
        st.session_state[key] = val
    n = int(BEMT_DEMO["bm_nseg"])
    _sync_bemt_segment_fields(n, float(BEMT_DEMO["bm_cl0"]), float(BEMT_DEMO["bm_cd0"]), force=True)


def _render_prop_static_thrust() -> None:
    st.markdown("##### Static thrust (momentum scaling)")
    st.caption("Sizing estimate — not a measured propeller map.")
    st.markdown(
        "[Reference — thrust equation background]"
        "(https://www.electricrcaircraftguy.com/2013/09/propeller-static-dynamic-thrust-equation.html)"
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rho = st.number_input("ρ (kg/m³)", value=1.225, min_value=0.001, format="%.4f", key="ct_rho")
    with c2:
        rpm = st.number_input("RPM", value=8000.0, min_value=1.0, key="ct_rpm")
    with c3:
        d_m = st.number_input("Diameter (m)", value=0.254, min_value=1e-6, key="ct_d")
    with c4:
        ct = st.number_input("C_T (lumped)", value=0.08, min_value=1e-6, format="%.4f", key="ct_ct")
    if st.button("Compute", key="ct_go"):
        r = ce.estimate_static_thrust_momentum(
            rho_kg_m3=rho, rpm=rpm, diameter_m=d_m, thrust_coefficient=ct
        )
        if r["ok"]:
            st.success(f"**Thrust:** {r['thrust_n']:.3f} N ({r['thrust_kgf']:.4f} kgf)")
            st.caption(r["note"])
        else:
            st.error(r["error"])


def _render_prop_tip_speed() -> None:
    st.markdown("##### Tip speed")
    st.caption("Noise and Mach-limit screening.")
    st.markdown(
        "[Reference — propeller speed calculator]"
        "(https://www.mrd-rc.com/tutorials-tools-and-testing/useful-tools/propeller-speed-calculator/)"
    )
    c1, c2 = st.columns(2)
    with c1:
        d_m = st.number_input("Diameter (m)", value=0.3048, min_value=1e-6, key="tip_d")
    with c2:
        rpm = st.number_input("RPM", value=6000.0, min_value=0.0, key="tip_rpm")
    if st.button("Compute", key="tip_go"):
        r = ce.propeller_tip_speed_m_s(diameter_m=d_m, rpm=rpm)
        if r["ok"]:
            st.success(f"**Tip speed:** {r['tip_speed_m_s']:.2f} m/s")
            st.caption(f"ω = {r['omega_rad_s']:.2f} rad/s")
        else:
            st.error(r["error"])


def _show_workbook_performance_block(title: str, perf: dict[str, float]) -> None:
    st.markdown(f"**{title}**")
    rows = [{"Quantity": k, "Value": v} for k, v in perf.items()]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_prop_bemt() -> None:
    _init_bemt_demo_inputs()

    st.markdown("##### Propeller sizing — RotriX workbook")
    st.caption(
        f"Mirrors `{ce.WORKBOOK_NAME}`: sheet **propeller calculations** (geometry), "
        "**Airfoil properties**, **Blade iteration 1** (elemental + blade performance)."
    )
    if st.button("Reset to demo inputs", key="bm_reset", help="Restore the pre-filled test case."):
        _reset_bemt_demo_inputs()
        st.rerun()
    with st.expander("How this maps to the Excel sheet"):
        st.markdown(
            "| Workbook input (col C) | Calculator field |\n"
            "|---|---|\n"
            "| Density at sea level | ρ lab |\n"
            "| Density @ 200 m | ρ altitude |\n"
            "| RPM, D, d_hub, T_lab, β_tip/hub, chords, blades, V_flight, segments | Same labels below |\n"
            "| Chord hub ≈ D/12.6, chord tip ≈ hub×0.66667 | **Apply workbook chord defaults** |\n"
            "\n**Outputs:** geometry per segment, airfoil properties, elemental dL/dD/dT/dQ/dP, "
            "and blade performance (Total T, Q, P, Cp, CT, Cq, J, η) for lab and 200 m."
        )

    auto_rho = st.checkbox(
        "Compute ρ from temperature (ideal gas, 101325 Pa)",
        key="bm_auto_rho",
    )

    st.markdown("**Inputs — `propeller calculations` (column C)**")
    st.caption("Pre-filled demo values — edit any field, then run sizing.")
    c1, c2, c3 = st.columns(3)
    with c1:
        rho_lab = st.number_input(
            "Density at sea level ρ (kg/m³)",
            min_value=1e-6,
            disabled=auto_rho,
            key="bm_rhol",
        )
        rho_alt = st.number_input(
            "Density @ 200 m altitude (kg/m³)",
            min_value=1e-6,
            disabled=auto_rho,
            key="bm_rhoa",
        )
    with c2:
        rpm = st.number_input("RPM", min_value=1.0, key="bm_rpm")
        d_prop = st.number_input("Propeller diameter D (m)", min_value=1e-6, key="bm_d")
        d_hub = st.number_input("Hub diameter d_hub (m)", min_value=0.0, key="bm_dh")
    with c3:
        t_lab = st.number_input("Temperature lab T (K)", min_value=1.0, key="bm_tl")
        t_alt = st.number_input("Temperature @ 200 m (K)", min_value=1.0, key="bm_ta")
        n_seg = st.number_input(
            "Number of radial segments (workbook uses 10)",
            min_value=2,
            max_value=50,
            step=1,
            key="bm_nseg",
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        beta_hub = st.number_input("Geometric pitch at hub β (deg)", key="bm_bh")
        beta_tip = st.number_input("Geometric pitch at tip β (deg)", key="bm_bt")
    with c2:
        use_wb_chords = st.checkbox(
            "Apply workbook chord defaults (D/12.6, tip = hub×0.66667)",
            key="bm_wb_ch",
        )
        chord_h = st.number_input(
            "Chord at hub (m)",
            min_value=1e-6,
            disabled=use_wb_chords,
            key="bm_ch",
        )
        chord_t = st.number_input(
            "Chord at tip (m)",
            min_value=1e-6,
            disabled=use_wb_chords,
            key="bm_ct",
        )
    with c3:
        n_blades = st.number_input("Number of blades", min_value=1, max_value=12, step=1, key="bm_nb")
        v_flight = st.number_input("Flight speed V (m/s)", min_value=0.0, key="bm_vf")
        alpha_i = st.number_input("Induced angle α_i (rad)", format="%.6f", key="bm_ai")

    if use_wb_chords:
        chord_h = ce.chord_hub_from_prop_diameter(float(d_prop))
        chord_t = ce.chord_tip_from_hub_chord(chord_h)

    st.markdown("**Airfoil data per segment** (10 values pre-filled; one value repeats for all segments)")
    c1, c2 = st.columns(2)
    with c1:
        cl_def = st.number_input("CL default (fallback if list empty)", key="bm_cl0")
    with c2:
        cd_def = st.number_input("CD default (fallback if list empty)", key="bm_cd0")
    _sync_bemt_segment_fields(int(n_seg), float(cl_def), float(cd_def))
    c1, c2 = st.columns(2)
    with c1:
        azl_txt = st.text_input("α zero-lift per segment (deg, comma-separated)", key="bm_azl")
        cl_txt = st.text_input("CL per segment (comma-separated)", key="bm_cll")
    with c2:
        cd_txt = st.text_input("CD per segment (comma-separated)", key="bm_cdl")

    if st.button("Run propeller sizing (workbook BEMT)", key="bm_go"):
        rho_l = float(rho_lab)
        rho_a = float(rho_alt)
        if auto_rho:
            rho_l = ce.air_density_from_temperature_k(float(t_lab))
            rho_a = ce.air_density_from_temperature_k(float(t_alt))
            st.info(f"ρ from T: lab = {rho_l:.4f} kg/m³, 200 m = {rho_a:.4f} kg/m³")

        out = None
        try:
            azl = _parse_segment_list(azl_txt, int(n_seg), 0.0)
            cl_list = _parse_segment_list(cl_txt, int(n_seg), float(cl_def))
            cd_list = _parse_segment_list(cd_txt, int(n_seg), float(cd_def))
            out = ce.bemt_blade_performance(
                rho_lab_kg_m3=rho_l,
                rho_alt_kg_m3=rho_a,
                rpm=float(rpm),
                prop_diameter_m=float(d_prop),
                hub_diameter_m=float(d_hub),
                temp_lab_k=float(t_lab),
                temp_alt_k=float(t_alt),
                beta_tip_deg=float(beta_tip),
                beta_hub_deg=float(beta_hub),
                chord_hub_m=float(chord_h),
                chord_tip_m=float(chord_t),
                n_blades=int(n_blades),
                flight_speed_m_s=float(v_flight),
                n_segments=int(n_seg),
                alpha_induced_rad=float(alpha_i),
                alpha_zero_lift_deg=azl,
                cl_per_segment=cl_list,
                cd_per_segment=cd_list,
            )
        except ValueError as e:
            st.error(str(e))
        if out is not None:
            if not out["ok"]:
                st.error(out["error"])
            else:
                st.markdown("#### Blade performance — `Blade iteration 1`")
                perf = out["blade_performance"]
                col_lab, col_alt = st.columns(2)
                with col_lab:
                    _show_workbook_performance_block("Lab conditions", perf["lab_conditions"])
                with col_alt:
                    _show_workbook_performance_block("200 m altitude", perf["altitude_200m"])

                st.markdown("#### Geometry — `propeller calculations`")
                st.dataframe(out["geometry_table"], use_container_width=True, hide_index=True)

                st.markdown("#### Airfoil properties")
                st.dataframe(out["airfoil_table"], use_container_width=True, hide_index=True)

                st.markdown("#### Blade elements — `Blade iteration 1`")
                tab_lab, tab_alt = st.tabs(["Lab", "200 m altitude"])
                with tab_lab:
                    st.dataframe(out["blade_elemental_lab"], use_container_width=True, hide_index=True)
                with tab_alt:
                    st.dataframe(out["blade_elemental_alt"], use_container_width=True, hide_index=True)

                with st.expander("Run metadata"):
                    st.json(out["meta"])


def _render_motor_kv_rpm() -> None:
    st.markdown("##### KV → no-load RPM")
    st.caption("Ideal unloaded speed; loaded RPM is lower under prop load.")
    st.markdown(
        "[Reference — BLDC KV calculator]"
        "(https://www.calculatorultra.com/en/tool/bldc-motor-speed-voltage-kv-rating-calculator.html)"
    )
    c1, c2 = st.columns(2)
    with c1:
        kv = st.number_input("KV (RPM/V)", value=920.0, min_value=1e-6, key="kv_kv")
    with c2:
        v = st.number_input("Voltage (V)", value=22.2, min_value=1e-6, key="kv_v")
    if st.button("Compute", key="kv_go"):
        r = ce.motor_kv_rpm(kv_rpm_per_v=kv, voltage_v=v)
        if r["ok"]:
            st.success(f"**No-load RPM (ideal):** {r['rpm_no_load']:.0f}")
            st.caption(r["note"])
        else:
            st.error(r["error"])


def _render_batt_flight_time() -> None:
    st.markdown("##### Flight time")
    st.caption("Hover-average current model.")
    st.markdown("[Reference — drone flight time](https://omnicalculator.com/other/drone-flight-time)")
    c1, c2, c3 = st.columns(3)
    with c1:
        mah = st.number_input("Capacity (mAh)", value=5000.0, min_value=1e-6, key="ft_mah")
    with c2:
        i_a = st.number_input("Average current (A)", value=25.0, min_value=1e-6, key="ft_i")
    with c3:
        frac = st.slider("Usable fraction", 0.1, 1.0, 0.8, 0.05, key="ft_frac")
    if st.button("Compute", key="ft_go"):
        r = ce.battery_flight_time_minutes(
            capacity_mah=mah, avg_current_a=i_a, usable_fraction=frac
        )
        if r["ok"]:
            st.success(f"**Time:** {r['time_min']:.1f} min ({r['time_h']:.3f} h)")
            st.caption(r["note"])
        else:
            st.error(r["error"])


def _render_batt_c_rate() -> None:
    st.markdown("##### C-rate")
    st.caption("Stress indicator for the battery pack.")
    st.markdown("[Reference — C-rate tool](https://ridewattly.com/tools/c-rate-calculator-tool/)")
    c1, c2 = st.columns(2)
    with c1:
        mah = st.number_input("Capacity (mAh)", value=5000.0, min_value=1e-6, key="cr_mah")
    with c2:
        i_a = st.number_input("Current (A)", value=100.0, min_value=0.0, key="cr_i")
    if st.button("Compute", key="cr_go"):
        r = ce.battery_c_rate(current_a=i_a, capacity_mah=mah)
        if r["ok"]:
            st.success(f"**C-rate:** {r['c_rate']:.2f} C")
        else:
            st.error(r["error"])


_RENDERERS: dict[str, callable] = {
    "prop_static_thrust": _render_prop_static_thrust,
    "prop_tip_speed": _render_prop_tip_speed,
    "prop_bemt": _render_prop_bemt,
    "motor_kv_rpm": _render_motor_kv_rpm,
    "batt_flight_time": _render_batt_flight_time,
    "batt_c_rate": _render_batt_c_rate,
}


def render() -> None:
    st.markdown("### Calculators")
    st.caption("Pick a **component**, then a **calculator purpose** for that part of the powertrain.")
    if st.button("← Back to RotriDASH", key="calc_back_btn"):
        _back_to_app()

    col_nav, col_body = st.columns([1, 2.2])

    with col_nav:
        st.markdown("**Component**")
        if st.session_state.get("calc_component") not in COMPONENT_ORDER:
            st.session_state["calc_component"] = COMPONENT_ORDER[0]
        component = st.radio(
            "component",
            COMPONENT_ORDER,
            key="calc_component",
            label_visibility="collapsed",
        )
        branches = CALCULATOR_TREE.get(component, [])
        purpose_labels = [b[1] for b in branches]
        purpose_ids = [b[0] for b in branches]
        purpose_key = f"calc_purpose_{component}"
        if st.session_state.get(purpose_key) not in range(len(purpose_labels)):
            st.session_state[purpose_key] = 0
        st.markdown("**Purpose**")
        purpose_idx = st.radio(
            "purpose",
            range(len(purpose_labels)),
            format_func=lambda i: purpose_labels[i],
            key=purpose_key,
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption(branches[purpose_idx][2])

    with col_body:
        st.markdown(f"#### {component}")
        st.markdown(f"**{purpose_labels[purpose_idx]}**")
        purpose_id = purpose_ids[purpose_idx]
        renderer = _RENDERERS.get(purpose_id)
        if renderer:
            renderer()
        else:
            st.warning("Calculator not available yet.")

    st.markdown("---")
    st.caption("Parameter glossary and external tool URLs: `RotriX tools/Calculator_Parameter_Comparison_Tables.md`.")
