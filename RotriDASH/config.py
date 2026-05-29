#type: ignore
"""
Constants, column mappings, display names, and small config helpers.
"""
import os
import base64

# ---------------------------------------------------------------------------
# Path to REUDE logo (relative to this file's folder)
# ---------------------------------------------------------------------------
REUDE_LOGO_PATH = os.path.join(os.path.dirname(__file__), "Logo.png")

# Set True to show the header "Calculators" shortcut and enable the calculators page route.
SHOW_CALCULATORS_BUTTON = True

# Set True to show "Multi-Parameter Analysis" / "Multi-File Comparison" on the analysis page.
SHOW_ANALYSIS_TYPE_SELECTOR = False


def _get_reude_logo_b64() -> str | None:
    """Load REUDE logo as base64 string for inline HTML display."""
    try:
        if os.path.exists(REUDE_LOGO_PATH):
            with open(REUDE_LOGO_PATH, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# ULog assessment topic ↔ category pairs
# ---------------------------------------------------------------------------
TOPIC_ASSESSMENT_PAIRS = [
    ("vehicle_local_position", "Actualposition"),
    ("vehicle_local_position_setpoint", "Setpointposition"),
    ("vehicle_local_position_setpoint", "Thrust"),
    ("vehicle_torque_setpoint", "Torque"),
    ("px4io_status", "Control"),
    ("battery_status", "Battery"),
]

ASSESSMENT_Y_AXIS_MAP = {
    "Actualposition": ["x", "y", "z"],
    "Setpointposition": ["x", "y", "z"],
    "Thrust": ["thrust[0]", "thrust[1]", "thrust[2]", "thrust[3]", "thrust[4]", "thrust[5]"],
    "Torque": ["xyz[0]", "xyz[1]", "xyz[2]"],
    "Control": ["pwm[0]", "pwm[1]", "pwm[2]", "pwm[3]", "pwm[4]", "pwm[5]"],
    "Battery": ["voltage_v", "current_average_a", "discharged_mah"],
}

# ---------------------------------------------------------------------------
# Column display-name mappings
# ---------------------------------------------------------------------------
COLUMN_DISPLAY_NAMES = {
    "pwm[0]": "Motor 1 pwm",
    "pwm[1]": "Motor 2 pwm",
    "pwm[2]": "Motor 3 pwm",
    "pwm[3]": "Motor 4 pwm",
    "pwm[4]": "Motor 5 pwm",
    "pwm[5]": "Motor 6 pwm",
    "thrust[0]": "Thrust Motor 1",
    "thrust[1]": "Thrust Motor 2",
    "thrust[2]": "Thrust Motor 3",
    "thrust[3]": "Thrust Motor 4",
    "thrust[4]": "Thrust Motor 5",
    "thrust[5]": "Thrust Motor 6",
    "xyz[0]": "Torque x",
    "xyz[1]": "Torque y",
    "xyz[2]": "Torque z",
    "voltage_v": "Battery Voltage",
    "current_average_a": "Current",
    "discharged_mah": "Discharged Capacity",
}

# Short display names specifically for the Sorted Performance Table in the PDF.
# This keeps column headers compact so they fit better on an A4 portrait page.
SORTED_TABLE_PDF_COLUMN_SHORT_NAMES = {
    # Core operating columns
    "Throttle": "Thr (%)",
    "Throttle (%)": "Thr (%)",
    "Throttle - %": "Thr (%)",
    "Throttle Input (%)": "Thr (%)",
    "throttle": "Thr (%)",

    "Voltage": "V",
    "Voltage (V)": "V",
    "Voltage [V]": "V",
    "Vol - V": "V",
    "voltage": "V",
    "Vol": "V",

    "Current": "I (A)",
    "Current (A)": "I (A)",
    "Current [A]": "I (A)",
    "Cur - A": "I (A)",
    "current": "I (A)",
    "Cur": "I (A)",

    "RPM": "RPM",
    "RPM1 - RPM": "RPM",
    "RPM1": "RPM",
    "rpm1": "RPM",
    "Motor Electrical Speed (RPM)": "RPM",
    "Motor Electrical Speed": "RPM",
    "Electrical Speed (RPM)": "RPM",
    "Electrical Speed": "RPM",
    "Rotational Speed (RPM)": "RPM",
    "Rotational Speed": "RPM",

    "Thrust (gf)": "Thrust (gf)",
    "Thrust - gf": "Thrust (gf)",
    "Thrust (kgf)": "Thrust (kgf)",
    "Thrust [g]": "Thrust (gf)",
    "Thrust": "Thrust (gf)",
    "thrust": "Thrust (gf)",

    "Torque (N·m)": "Torque (N·m)",
    "Torque (N*m)": "Torque (N·m)",
    "Torque - N*m": "Torque (N·m)",
    "Torque (Nm)": "Torque (Nm)",
    "Torque (N.m)": "Torque (N·m)",
    "Torque [N·m]": "Torque (N·m)",
    "Torque [N*m]": "Torque (N·m)",
    "Torque": "Torque (N·m)",
    "torque": "Torque (N·m)",

    # Power / efficiency
    "SysEffect - gf/W": "SysEff (gf/W)",
    "SysEffect (gf/W)": "SysEff (gf/W)",
    "SysEffect": "SysEff (gf/W)",
    "Overall Efficiency (gf/W)": "SysEff (gf/W)",
    "Overall Efficiency": "SysEff (gf/W)",

    "MotorPower - W": "Mtr Pwr (W)",
    "MotorPower": "Mtr Pwr (W)",
    "Motor Power - W": "Mtr Pwr (W)",
    "Mechanical Power (W)": "Mtr Pwr (W)",

    "Electrical Power - W": "Elec Pwr (W)",
    "Electrical Power (W)": "Elec Pwr (W)",
    "ElectricalPower - W": "Elec Pwr (W)",
    "Electrical (W)": "Elec Pwr (W)",
    "Power": "Elec Pwr (W)",

    "Mechanical (W)": "Mech Pwr (W)",
    "Mechanical (bhp)": "Mech (bhp)",

    # wingflyingtech / VT-100KG bench columns
    "Propulsion (GPW)": "Sys Eff (g/W)",
    "Propeller (GPW)": "Prop Eff (g/W)",
    "ESC & Motor Efficiency(%)": "ESC+Mtr Eff (%)",
    "Ambient T (C)": "Amb T (°C)",
    "Winding T (C)": "Wind T (°C)",
    "Motor T (C)": "Mtr T (°C)",
    "ESC T (C)": "ESC T (°C)",
    "Baring T (C)": "Brg T (°C)",
    "Pressure": "Press",
    "U Phase (A)": "U (A)",
    "V Phase (A)": "V Pha (A)",
    "W Phase (A)": "W Pha (A)",
    "Voltage 2(V)": "V2 (V)",

    # Vibration / acceleration
    "AccX (g)": "Ax (g)",
    "AccY (g)": "Ay (g)",
    "AccZ (g)": "Az (g)",
    "Vibration (g)": "Vib (g)",
    "Vibration RMS (g)": "Vib (g)",
    "Vibration - g": "Vib (g)",
    "Vibration": "Vib (g)",

    # Efficiencies
    "Motor Efficiency (%)": "Mot Eff (%)",
    "MotorEfficiency (%)": "Mot Eff (%)",
    "Propeller Mech. Efficiency (gf/W)": "Prop Mech Eff (gf/W)",
}

# Columns to exclude from the Sorted Performance Table in reports (PDF and HTML).
SORTED_TABLE_REPORT_DROP_COLUMNS = [
    "AccX (g)", "AccY (g)", "AccZ (g)",
    "Acceleration X", "Acceleration Y", "Acceleration Z",
    "Motor Efficiency (%)", "MotorEfficiency (%)",
    "Propeller Mech. Efficiency (gf/W)", "Propeller Mech Efficiency (gf/W)",
    "Propeller Mech. Efficiency", "Propeller Efficiency", "Prop Mech Eff (gf/W)",
]


# ---------------------------------------------------------------------------
# Helper functions that depend only on the constants above
# ---------------------------------------------------------------------------

def get_display_name(col):
    """Get a display-friendly name for a column."""
    if col == 'timestamp_seconds':
        return 'Time (secs)'
    if col == 'Index':
        return 'Index'
    return COLUMN_DISPLAY_NAMES.get(col, col)


def get_axis_title(axis_name):
    if axis_name == 'timestamp_seconds':
        return 'TIME(secs)'
    return axis_name


def _drop_sorted_table_report_columns(df):
    """Drop acceleration, motor efficiency, and prop mechanical eff columns for report."""
    if df is None or df.empty:
        return df
    drop_set = set(SORTED_TABLE_REPORT_DROP_COLUMNS)
    cols_to_drop = [
        c for c in df.columns
        if c in drop_set
        or "accx" in str(c).lower() or "accy" in str(c).lower() or "accz" in str(c).lower()
        or "acceleration" in str(c).lower()
        or "motor efficiency" in str(c).lower() or "motorefficiency" in str(c).lower()
        or "propeller mech" in str(c).lower() or "prop mech" in str(c).lower()
    ]
    if not cols_to_drop:
        return df
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")


# Common word abbreviations for auto-shortening parameter names.
_WORD_ABBREVS = {
    "electrical": "Elec",
    "power": "Pwr",
    "motor": "Mtr",
    "throttle": "Thr",
    "thrust": "Thrust",
    "torque": "Torque",
    "voltage": "Vol",
    "current": "Cur",
    "efficiency": "Eff",
    "propeller": "Prop",
    "mechanical": "Mech",
    "vibration": "Vib",
    "acceleration": "Acc",
    "temperature": "Temp",
    "speed": "Spd",
    "average": "Avg",
    "frequency": "Freq",
    "position": "Pos",
    "setpoint": "SP",
    "overall": "Ovrl",
    "system": "Sys",
    "battery": "Batt",
    "discharged": "Disch",
    "capacity": "Cap",
}


def _auto_abbreviate(name: str, max_len: int = 12) -> str:
    """
    Intelligently shorten a parameter name for graph labels.
    1. Extract the unit part (in parentheses or after ' - ').
    2. Abbreviate known words.
    3. Truncate if still too long.
    """
    import re
    if not name:
        return ""

    # Extract unit from patterns like "Name - unit" or "Name (unit)"
    unit = ""
    core = name
    m = re.match(r'^(.+?)\s*-\s*(\S+.*)$', name)
    if m:
        core, unit = m.group(1).strip(), m.group(2).strip()
    else:
        m = re.match(r'^(.+?)\s*\(([^)]+)\)\s*$', name)
        if m:
            core, unit = m.group(1).strip(), m.group(2).strip()

    # Abbreviate each word in the core name
    words = core.split()
    abbreviated_words = []
    for w in words:
        lower_w = w.lower().rstrip(".,;:")
        if lower_w in _WORD_ABBREVS:
            abbreviated_words.append(_WORD_ABBREVS[lower_w])
        else:
            abbreviated_words.append(w)

    short_name = " ".join(abbreviated_words)

    # If still too long, take first 3 chars of each word
    if len(short_name) > max_len and len(abbreviated_words) > 1:
        short_name = " ".join(w[:3] for w in abbreviated_words)

    # Combine with unit
    if unit:
        return f"{short_name}<br>({unit})"
    return short_name


def get_short_param_label(col: str) -> str:
    """
    Get a compact, graph-friendly label for a parameter.
    1. Prefer the short names in SORTED_TABLE_PDF_COLUMN_SHORT_NAMES.
    2. Otherwise, try auto-abbreviating the display name.
    """
    if not col:
        return ""
    # Check the hardcoded short-name dict first
    base = SORTED_TABLE_PDF_COLUMN_SHORT_NAMES.get(col)
    if base:
        # Format with unit on separate line if it has parentheses
        try:
            if "(" in base and ")" in base:
                idx = base.rfind(" (")
                if idx != -1:
                    name_part = base[:idx]
                    unit_part = base[idx + 1:]
                    return f"{name_part}<br>{unit_part}"
        except Exception:
            pass
        return base

    # Also check by display name
    display = get_display_name(col)
    base2 = SORTED_TABLE_PDF_COLUMN_SHORT_NAMES.get(display)
    if base2:
        try:
            if "(" in base2 and ")" in base2:
                idx = base2.rfind(" (")
                if idx != -1:
                    return f"{base2[:idx]}<br>{base2[idx + 1:]}"
        except Exception:
            pass
        return base2

    # Fallback: auto-abbreviate the display name
    return _auto_abbreviate(display)
