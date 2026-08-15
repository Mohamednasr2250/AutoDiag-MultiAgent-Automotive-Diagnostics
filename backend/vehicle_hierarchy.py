"""
vehicle_hierarchy.py — Vehicle System Hierarchy + Repair Dependency Graph
Maps fault codes to systems/components and topologically sorts repairs.
"""

from typing import List, Dict

VEHICLE_HIERARCHY: Dict[str, Dict[str, List[str]]] = {
    "Engine System": {
        "Fuel System": ["Fuel Injectors", "Fuel Pump", "Fuel Pressure Regulator", "MAF Sensor"],
        "Ignition System": ["Spark Plugs", "Ignition Coils", "Crankshaft Position Sensor"],
        "Exhaust System": ["Catalytic Converter", "O2 Sensors", "Exhaust Manifold"],
    },
    "Transmission System": {
        "Automatic Transmission": ["Solenoids", "Torque Converter", "TCM"],
    },
    "Brake System": {
        "ABS": ["Wheel Speed Sensors", "ABS Module", "Brake Lines"],
    },
    "Safety System": {
        "Airbag": ["Airbag Sensors", "Clock Spring", "SRS Module"],
    },
}

CODE_SYSTEM_MAP: Dict[str, tuple] = {
    "P0101": ("Engine System", "Fuel System"),
    "P0171": ("Engine System", "Fuel System"),
    "P0300": ("Engine System", "Ignition System"),
    "P0301": ("Engine System", "Ignition System"),
    "P0302": ("Engine System", "Ignition System"),
    "P0303": ("Engine System", "Ignition System"),
    "P0304": ("Engine System", "Ignition System"),
    "P0420": ("Engine System", "Exhaust System"),
    "P0430": ("Engine System", "Exhaust System"),
    "P0442": ("Engine System", "Exhaust System"),
    "P0455": ("Engine System", "Exhaust System"),
    "P0700": ("Transmission System", "Automatic Transmission"),
    "P0730": ("Transmission System", "Automatic Transmission"),
    "C0035": ("Brake System", "ABS"),
    "C0040": ("Brake System", "ABS"),
    "C0045": ("Brake System", "ABS"),
    "B0001": ("Safety System", "Airbag"),
}

REPAIR_DEPENDENCIES: Dict[str, List[str]] = {
    "diagnose_exhaust_leak": [],
    "fix_o2_sensor": ["diagnose_exhaust_leak"],
    "replace_catalytic_converter": ["fix_o2_sensor"],
    "fix_misfires": [],
    "replace_spark_plugs": ["fix_misfires"],
    "check_spark_plugs": [],
    "check_maf_sensor": [],
    "fix_maf_sensor": ["check_maf_sensor"],
    "fix_fuel_lean": ["fix_maf_sensor"],
    "replace_wheel_speed_sensor": [],
    "fix_abs_module": ["replace_wheel_speed_sensor"],
}

CODE_TO_REPAIR: Dict[str, str] = {
    "P0101": "fix_maf_sensor", "P0171": "fix_fuel_lean",
    "P0300": "fix_misfires", "P0301": "fix_misfires", "P0302": "fix_misfires",
    "P0303": "fix_misfires", "P0304": "fix_misfires",
    "P0420": "replace_catalytic_converter", "P0430": "replace_catalytic_converter",
    "C0035": "replace_wheel_speed_sensor", "C0040": "replace_wheel_speed_sensor",
    "C0045": "replace_wheel_speed_sensor",
}


def get_system_context(code: str) -> str:
    code = code.strip().upper()
    if code not in CODE_SYSTEM_MAP:
        return f"No system mapping found for {code}."
    system, subsystem = CODE_SYSTEM_MAP[code]
    components = VEHICLE_HIERARCHY.get(system, {}).get(subsystem, [])
    return (f"System: {system} > {subsystem}\n"
            f"Components in this subsystem: {', '.join(components)}")


def get_affected_components(code: str) -> str:
    code = code.strip().upper()
    if code not in CODE_SYSTEM_MAP:
        return f"No component mapping found for {code}."
    system, subsystem = CODE_SYSTEM_MAP[code]
    components = VEHICLE_HIERARCHY.get(system, {}).get(subsystem, [])
    return f"Affected components ({subsystem}): {', '.join(components)}"


def topological_sort(repairs: List[str]) -> List[str]:
    repairs = list(dict.fromkeys(repairs))
    deps = {r: [d for d in REPAIR_DEPENDENCIES.get(r, []) if d in repairs] for r in repairs}

    ordered = []
    visited = set()
    temp_mark = set()

    def visit(node):
        if node in visited:
            return
        if node in temp_mark:
            return
        temp_mark.add(node)
        for dep in deps.get(node, []):
            visit(dep)
        temp_mark.discard(node)
        visited.add(node)
        ordered.append(node)

    for r in repairs:
        visit(r)

    return ordered


def get_repair_order(codes_str: str) -> str:
    codes = [c.strip().upper() for c in codes_str.replace(",", " ").split() if c.strip()]
    repairs = []
    for c in codes:
        repair = CODE_TO_REPAIR.get(c)
        if repair and repair not in repairs:
            repairs.append(repair)

    if not repairs:
        return f"No known repair mapping for codes: {codes}"

    ordered = topological_sort(repairs)
    lines = [f"Repair Order for {codes}:"]
    for i, r in enumerate(ordered, 1):
        lines.append(f"  {i}. {r.replace('_', ' ').title()}")
    return "\n".join(lines)