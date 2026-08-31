# =============================================================================
# STAGE 5 — EVENT GENERATOR
# Trustworthy Grid-Aware LLM Operator
# =============================================================================
#
# PURPOSE
# -------
# Turn the 4997 normal operating points from Stage 4 into a labelled corpus
# of scenarios, each one a complete before-and-after record:
#
#     PRE-EVENT CONDITION   the operating point, solved and normal
#     INJECTED EVENT        what was deliberately done to the system
#     POST-EVENT CONDITION  the solved state that resulted
#     CONSEQUENCES          which limits moved, which components violated
#
# THE CENTRAL RULE OF THIS STAGE
# ------------------------------
# The injected event and its consequences are recorded SEPARATELY and are
# never allowed to contaminate each other.
#
#     injected:      "+30 % demand at bus_14"
#     consequences:  "bus_14 undervoltage, line_9_14 overloaded"
#
# The first is what an operator DID. The second is what the physics DID
# ABOUT IT. Merging them would make the entire project circular: the model
# would be scored on recovering a label that was written into its own input.
# Keeping them apart is what lets Stage 22 ask "what happened here?" and
# Stage 23 check the answer against something the model could not have read
# off the prompt.
#
# NOTHING IS EVER ASSIGNED, ONLY CAUSED
# -------------------------------------
# For the outcome classes — E6 undervoltage, E7 overvoltage, E8 thermal
# overload — the result is NEVER written into the state. No line is set to
# "loading = 120 %". Instead a physical disturbance is applied, the power
# flow is solved, and the outcome is DETECTED. If the disturbance did not
# produce the target outcome, the scenario is escalated or discarded. Every
# E6 in the corpus is a state where the solver, not the author, decided the
# voltage was low.
#
# TWO AXES, NOT ONE
# -----------------
# E1-E5 are MECHANISM classes: they name what was injected.
# E6-E8 are OUTCOME classes: they name what resulted.
#
# These are different axes and they overlap by construction. A large E1 load
# surge may well cause undervoltage; it stays labelled E1 because E1
# describes the injection. The corpus therefore records BOTH for every
# scenario — `event_class` for the injected mechanism and the consequence
# flags for the outcome — and the overlap is measured and reported rather
# than hidden. Any paper using this taxonomy has to state which axis its
# classification task is on.
#
# WHAT THE NETWORK CAN AND CANNOT DO (measured, not assumed)
# ----------------------------------------------------------
# Every single-line outage converges; none islands the system. Several are
# severe: line_1_2 drives a branch to 228 %, line_9_14 drops a bus to
# 0.9425 p.u.
#
# E7 needed a change of approach. The execution plan suggests "very low load"
# and "high distributed generation" as overvoltage mechanisms. Measured on
# this network they do not work: minimum load with both renewables at 100 %
# and the battery discharging reaches only v_max = 1.0347 p.u., nowhere near
# the 1.05 limit. The four synchronous machines hold their buses at 1.02-1.03
# and the system is stiff enough that injections cannot lift it further.
#
# So E7 is generated through explicit REACTIVE mechanisms instead:
#
#     avr_setpoint_shift      excitation setpoints raised, as in a
#                             mis-set or malfunctioning AVR
#     shunt_overcompensation  the bus-9 capacitor bank left in or
#                             over-sized, the classic overvoltage cause
#
# Both are real operating conditions, both are injected and then detected
# like everything else. This is a finding about the IEEE 14-bus system, not
# a workaround, and it belongs in the paper's limitations section.
#
# NOT IN THIS STAGE
# -----------------
# Violation detection logic as a reusable component   -> STAGE 8
# Natural-language ground truth                       -> STAGE 9
# Systematic N-1 screening of every branch            -> STAGE 12
#
# Stage 5 causes and records. It does not explain.
#
# REQUIRED INPUTS
# ---------------
#   03_renewables_bess.py                          (imported: BESS model)
#   data/processed/ieee14_net_re.json              (Stage 3)
#   data/processed/ieee14_re_layout_hash.txt       (Stage 3)
#   data/processed/ieee14_operating_points.csv     (Stage 4)
#   data/processed/ieee14_op_load_factors.csv      (Stage 4)
#   data/processed/ieee14_op_bus_voltages.csv      (Stage 4)
#   data/processed/ieee14_op_branch_loading.csv    (Stage 4)
#
# OUTPUTS
# -------
# data/processed/
#     ieee14_scenarios.csv                 flat table, one row per scenario
#     ieee14_scenarios.jsonl               full nested record per scenario
#     ieee14_scenario_post_voltages.csv    post-event bus voltages
#     ieee14_scenario_post_loading.csv     post-event branch loadings
#     ieee14_scenario_class_summary.csv    per-class statistics
#     stage5_validation_summary.csv
#     stage5_metadata.json
#
# RUN
# ---
# python 05_event_generator.py
#
# Exit code 0 = Checkpoint 5 passed, proceed to Stage 6.
#
# =============================================================================


from __future__ import annotations

import copy
import importlib.util
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


try:
    import psutil

    PSUTIL_AVAILABLE = True

except ImportError:

    PSUTIL_AVAILABLE = False


try:
    import numba  # noqa: F401

    NUMBA_AVAILABLE = True

except ImportError:

    NUMBA_AVAILABLE = False


import pandapower as pp


# =============================================================================
# CONFIGURATION
# =============================================================================

STAGE_NAME = (
    "STAGE 5 — EVENT GENERATOR"
)

NETWORK_NAME = "ieee14_re"

LINE_WIDTH = 100

STAGE3_FILE = Path("03_renewables_bess.py")

STAGE3_MODULE_NAME = "stage3_renewables_bess"

SCENARIO_PREFIX = "IEEE14_SCN"

MASTER_SEED = 20260821


# -----------------------------------------------------------------------------
# CORPUS SIZE
# -----------------------------------------------------------------------------
#
# Per class. Ten classes at 300 gives 3000 scenarios, each needing one or
# two power flows and the outcome classes sometimes several more while they
# escalate. Around six minutes without numba.
#
# -----------------------------------------------------------------------------

N_PER_CLASS = 300

PROGRESS_EVERY = 250


# -----------------------------------------------------------------------------
# EVENT CLASSES
# -----------------------------------------------------------------------------

EVENT_CLASSES = {
    "E0": "Normal",
    "E1": "Load Surge",
    "E2": "Load Drop",
    "E3": "Transmission-Line Outage",
    "E4": "Generator Outage",
    "E5": "Renewable Ramp",
    "E6": "Undervoltage",
    "E7": "Overvoltage",
    "E8": "Thermal Overload",
    "E9": "Compound Event",
}

MECHANISM_CLASSES = ("E1", "E2", "E3", "E4", "E5")

OUTCOME_CLASSES = ("E6", "E7", "E8")


# -----------------------------------------------------------------------------
# EVENT MAGNITUDES
# -----------------------------------------------------------------------------

LOAD_SURGE_PERCENT = (10.0, 20.0, 30.0, 40.0)

LOAD_DROP_PERCENT = (-10.0, -20.0, -30.0)

RAMP_SEVERITY = {
    "small": 0.10,
    "medium": 0.30,
    "large": 0.60,
}


# -----------------------------------------------------------------------------
# REGIONS
# -----------------------------------------------------------------------------
#
# Named groups of load buses, so an E1 surge can hit "a region" the way a
# heatwave or an industrial estate would, rather than one arbitrary bus.
# Split by electrical geography: the 135 kV load centre, and the two halves
# of the low-voltage island fed through the bus-6 and bus-9 transformers.
#
# -----------------------------------------------------------------------------

REGIONS = {
    "hv_load_centre": [2, 3, 4, 5],
    "lv_north": [6, 11, 12, 13],
    "lv_south": [9, 10, 14],
}


# -----------------------------------------------------------------------------
# OUTCOME TARGETS
# -----------------------------------------------------------------------------
#
# Read from the network's own limit columns at run time. Declared here only
# as the names of the things being searched for.
#
# -----------------------------------------------------------------------------

MAX_SEARCH_ATTEMPTS = 12

MAX_OPERATING_POINT_RETRIES = 6

TOL_REPLAY_PU = 1.0e-12


# -----------------------------------------------------------------------------
# E7 REACTIVE MECHANISMS
# -----------------------------------------------------------------------------
#
# Escalation ladders for overvoltage. See the header: low load and high DG
# alone cannot lift this network above 1.05 p.u.
#
# -----------------------------------------------------------------------------

AVR_SHIFT_LADDER = (0.02, 0.025, 0.03, 0.035, 0.04, 0.05)

SHUNT_FACTOR_LADDER = (2.0, 3.0, 4.0, 6.0, 8.0)


# =============================================================================
# INPUT / OUTPUT FILES
# =============================================================================

OUTPUT_DIR = Path("data") / "processed"

NET_RE_FILE = OUTPUT_DIR / "ieee14_net_re.json"
RE_HASH_FILE = OUTPUT_DIR / "ieee14_re_layout_hash.txt"

POINTS_FILE = OUTPUT_DIR / "ieee14_operating_points.csv"
FACTORS_FILE = OUTPUT_DIR / "ieee14_op_load_factors.csv"
VOLTAGES_FILE = OUTPUT_DIR / "ieee14_op_bus_voltages.csv"
LOADING_FILE = OUTPUT_DIR / "ieee14_op_branch_loading.csv"

SCENARIOS_CSV = OUTPUT_DIR / "ieee14_scenarios.csv"
SCENARIOS_JSONL = OUTPUT_DIR / "ieee14_scenarios.jsonl"
POST_VOLTAGES_FILE = OUTPUT_DIR / "ieee14_scenario_post_voltages.csv"
POST_LOADING_FILE = OUTPUT_DIR / "ieee14_scenario_post_loading.csv"
CLASS_SUMMARY_FILE = OUTPUT_DIR / "ieee14_scenario_class_summary.csv"
VALIDATION_FILE = OUTPUT_DIR / "stage5_validation_summary.csv"
METADATA_OUTPUT_FILE = OUTPUT_DIR / "stage5_metadata.json"


# =============================================================================
# TERMINAL HELPERS
# =============================================================================

def print_header(
    title: str,
    char: str = "=",
) -> None:

    print()
    print(char * LINE_WIDTH)
    print(title)
    print(char * LINE_WIDTH)


def print_subheader(
    title: str,
) -> None:

    print()
    print("-" * LINE_WIDTH)
    print(title)
    print("-" * LINE_WIDTH)


def info(
    message: str,
) -> None:

    print(f"[INFO] {message}")


def passed(
    message: str,
) -> None:

    print(f"[PASS] {message}")


def warning(
    message: str,
) -> None:

    print(f"[WARNING] {message}")


def failed(
    message: str,
) -> None:

    print(f"[FAIL] {message}")


def check(
    condition: bool,
    message: str,
    results: Dict[str, bool],
    key: str,
) -> bool:

    results[key] = bool(condition)

    if condition:
        passed(message)

    else:
        failed(message)

    return bool(condition)


def require(
    condition: bool,
    message: str,
) -> None:

    if not condition:

        raise AssertionError(message)


# =============================================================================
# =============================================================================
# PART 1 — NETWORK STATE MANAGEMENT
# =============================================================================
#
# Every scenario mutates the network: loads change, branches go out of
# service, setpoints shift. Without a disciplined reset, scenario 42 would
# inherit the line outage from scenario 41 and no assertion in this file
# would catch it — the state would simply be quietly wrong.
#
# So a full snapshot of every mutable field is taken once, and restored
# before every single scenario.
#
# =============================================================================

MUTABLE_FIELDS = {
    "load": ("p_mw", "q_mvar", "in_service"),
    "line": ("in_service",),
    "trafo": ("in_service",),
    "gen": ("p_mw", "vm_pu", "in_service"),
    "ext_grid": ("vm_pu", "in_service"),
    "sgen": ("p_mw", "q_mvar", "in_service"),
    "storage": ("p_mw", "soc_percent", "in_service"),
    "shunt": ("q_mvar", "p_mw", "in_service"),
}


def snapshot_network(
    net,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Copy every field a scenario is allowed to touch.
    """

    snapshot: Dict[str, Dict[str, np.ndarray]] = {}

    for table, columns in MUTABLE_FIELDS.items():

        frame = getattr(net, table)

        if not len(frame):
            continue

        snapshot[table] = {
            column: frame[column].values.copy()
            for column in columns
            if column in frame.columns
        }

    return snapshot


def restore_network(
    net,
    snapshot: Dict[str, Dict[str, np.ndarray]],
) -> None:
    """
    Put the network back exactly as it was.
    """

    for table, columns in snapshot.items():

        frame = getattr(net, table)

        for column, values in columns.items():

            frame[column] = values.copy()


def solve(
    net,
) -> bool:
    """
    Solve, returning convergence rather than raising.

    A scenario that does not converge is a legitimate outcome — a severe
    enough disturbance can genuinely have no steady-state solution — and it
    is recorded as such rather than crashing the run.
    """

    try:

        pp.runpp(net, numba=NUMBA_AVAILABLE)

        return bool(net.converged)

    except Exception:  # noqa: BLE001

        return False


# =============================================================================
# =============================================================================
# PART 2 — EVENT INJECTION
# =============================================================================
#
# An injected event is a plain JSON-serialisable dictionary describing what
# to do, and `apply_injected_event` is the only thing that knows how to do
# it. Nothing else in this file writes to the network.
#
# That indirection buys the single most valuable property of this stage:
# REPLAYABILITY. Because the event is data rather than code, Stage 5 can
# re-apply a stored event to a restored pre-event state and confirm it
# reproduces the recorded post-event state exactly. Stage 9 and Stage 22 can
# do the same. An event that cannot be replayed is not evidence.
#
# =============================================================================

def describe_targets(
    targets: Sequence[str],
) -> str:

    if len(targets) == 1:

        return str(targets[0])

    if len(targets) <= 3:

        return ", ".join(str(t) for t in targets)

    return f"{len(targets)} components"


def make_load_event(
    targets: Sequence[str],
    percent: float,
    scope: str,
) -> Dict[str, Any]:

    direction = "increase" if percent > 0 else "decrease"

    return {
        "mechanism": "load_change",
        "scope": scope,
        "targets": list(targets),
        "magnitude_percent": float(percent),
        "direction": direction,
        "description": (
            f"{abs(percent):.0f} % demand {direction} at "
            f"{describe_targets(targets)}"
        ),
    }


def make_line_outage_event(
    line_id: str,
) -> Dict[str, Any]:

    return {
        "mechanism": "line_outage",
        "scope": "single_branch",
        "targets": [line_id],
        "description": f"outage of {line_id}",
    }


def make_generator_outage_event(
    gen_id: str,
) -> Dict[str, Any]:

    return {
        "mechanism": "generator_outage",
        "scope": "single_unit",
        "targets": [gen_id],
        "description": f"outage of {gen_id}",
    }


def make_ramp_event(
    targets: Sequence[str],
    delta: float,
    severity: str,
) -> Dict[str, Any]:

    direction = "up-ramp" if delta > 0 else "down-ramp"

    return {
        "mechanism": "renewable_ramp",
        "scope": "resource",
        "targets": list(targets),
        "delta_availability": float(delta),
        "severity": severity,
        "direction": direction,
        "description": (
            f"{severity} {direction} of {abs(delta) * 100:.0f} "
            f"availability points at {describe_targets(targets)}"
        ),
    }


def make_avr_event(
    targets: Sequence[str],
    delta_vm_pu: float,
) -> Dict[str, Any]:

    return {
        "mechanism": "avr_setpoint_shift",
        "scope": "voltage_control",
        "targets": list(targets),
        "delta_vm_pu": float(delta_vm_pu),
        "description": (
            f"excitation setpoints raised by {delta_vm_pu:.3f} p.u. at "
            f"{describe_targets(targets)}"
        ),
    }


def make_shunt_event(
    factor: float,
) -> Dict[str, Any]:

    return {
        "mechanism": "shunt_overcompensation",
        "scope": "reactive_support",
        "targets": ["shunt_bus_9"],
        "factor": float(factor),
        "description": (
            f"capacitor bank at bus_9 over-compensating by a factor of "
            f"{factor:.1f}"
        ),
    }


def make_bess_event(
    p_mw: float,
) -> Dict[str, Any]:

    mode = "charge" if p_mw > 0 else "discharge"

    return {
        "mechanism": "bess_dispatch",
        "scope": "storage",
        "targets": ["gen_BESS9"],
        "p_mw": float(p_mw),
        "description": (
            f"battery commanded to {mode} at {abs(p_mw):.1f} MW"
        ),
    }


def make_compound_event(
    components: Sequence[Dict[str, Any]],
    label: str,
) -> Dict[str, Any]:

    return {
        "mechanism": "compound",
        "scope": "multiple",
        "label": label,
        "components": list(components),
        "targets": [
            target
            for component in components
            for target in component.get("targets", [])
        ],
        "description": " + ".join(
            component["description"] for component in components
        ),
    }


def make_null_event() -> Dict[str, Any]:

    return {
        "mechanism": "none",
        "scope": "none",
        "targets": [],
        "description": "no disturbance injected",
    }


# -----------------------------------------------------------------------------
# APPLICATION
# -----------------------------------------------------------------------------

def apply_injected_event(
    net,
    event: Dict[str, Any],
) -> None:
    """
    Apply one injected event to a network already in its pre-event state.

    Every mutation this stage performs goes through here. Compound events
    recurse, so a compound is exactly its components applied in order — no
    special case, no separate code path that could diverge.
    """

    mechanism = event["mechanism"]

    if mechanism == "none":

        return

    if mechanism == "compound":

        for component in event["components"]:

            apply_injected_event(net, component)

        return

    if mechanism == "load_change":

        factor = 1.0 + event["magnitude_percent"] / 100.0

        mask = net.load.cid.isin(event["targets"]).values

        net.load.loc[mask, "p_mw"] = (
            net.load.loc[mask, "p_mw"].values * factor
        )

        net.load.loc[mask, "q_mvar"] = (
            net.load.loc[mask, "q_mvar"].values * factor
        )

        return

    if mechanism == "line_outage":

        mask = net.line.cid.isin(event["targets"]).values

        net.line.loc[mask, "in_service"] = False

        return

    if mechanism == "generator_outage":

        mask = net.gen.cid.isin(event["targets"]).values

        net.gen.loc[mask, "in_service"] = False

        return

    if mechanism == "renewable_ramp":

        delta = event["delta_availability"]

        mask = net.sgen.cid.isin(event["targets"]).values

        rated = net.sgen.loc[mask, "rated_mw"].values.astype(float)

        current = net.sgen.loc[mask, "p_mw"].values.astype(float)

        availability = np.clip(current / rated + delta, 0.0, 1.0)

        net.sgen.loc[mask, "p_mw"] = rated * availability

        net.sgen.loc[mask, "availability"] = availability

        return

    if mechanism == "avr_setpoint_shift":

        delta = event["delta_vm_pu"]

        gen_mask = net.gen.cid.isin(event["targets"]).values

        net.gen.loc[gen_mask, "vm_pu"] = (
            net.gen.loc[gen_mask, "vm_pu"].values + delta
        )

        slack_mask = net.ext_grid.cid.isin(event["targets"]).values

        if slack_mask.any():

            net.ext_grid.loc[slack_mask, "vm_pu"] = (
                net.ext_grid.loc[slack_mask, "vm_pu"].values + delta
            )

        return

    if mechanism == "shunt_overcompensation":

        net.shunt["q_mvar"] = (
            net.shunt["q_mvar"].values * event["factor"]
        )

        return

    if mechanism == "bess_dispatch":

        net.storage["p_mw"] = float(event["p_mw"])

        return

    raise ValueError(
        f"Unknown injected-event mechanism: '{mechanism}'."
    )


# =============================================================================
# =============================================================================
# PART 3 — STATE MEASUREMENT AND CONSEQUENCES
# =============================================================================
#
# Consequences are DETECTED from the solved state against the limit columns
# Stage 1 wrote onto the element tables. Nothing here is ever assigned.
#
# =============================================================================

def measure_state(
    net,
) -> Dict[str, Any]:
    """
    Full description of a solved state, including which specific components
    violate which limit.

    The component ID lists are the part that matters downstream: Stage 9
    writes them into ground truth and Stage 23 checks the model's claims
    against them, so "line_9_14 is overloaded" becomes a checkable fact
    rather than an impression.
    """

    voltages = net.res_bus.vm_pu.values

    under_mask = voltages < net.bus.min_vm_pu.values
    over_mask = voltages > net.bus.max_vm_pu.values

    line_loading = net.res_line.loading_percent.values
    trafo_loading = net.res_trafo.loading_percent.values

    line_mask = (
        line_loading > net.line.max_loading_percent.values
    ) & net.line.in_service.values

    trafo_mask = (
        trafo_loading > net.trafo.max_loading_percent.values
    ) & net.trafo.in_service.values

    undervoltage_buses = list(net.bus.cid.values[under_mask])
    overvoltage_buses = list(net.bus.cid.values[over_mask])
    overloaded_lines = list(net.line.cid.values[line_mask])
    overloaded_trafos = list(net.trafo.cid.values[trafo_mask])

    in_service_line = line_loading[net.line.in_service.values]
    in_service_trafo = trafo_loading[net.trafo.in_service.values]

    max_line = (
        float(in_service_line.max()) if len(in_service_line) else 0.0
    )

    max_trafo = (
        float(in_service_trafo.max()) if len(in_service_trafo) else 0.0
    )

    n_violations = (
        len(undervoltage_buses)
        + len(overvoltage_buses)
        + len(overloaded_lines)
        + len(overloaded_trafos)
    )

    return {
        "v_min_pu": float(voltages.min()),
        "v_max_pu": float(voltages.max()),
        "v_min_bus": str(net.bus.cid.values[int(voltages.argmin())]),
        "v_max_bus": str(net.bus.cid.values[int(voltages.argmax())]),
        "max_line_loading_percent": max_line,
        "max_trafo_loading_percent": max_trafo,
        "peak_branch_loading_percent": max(max_line, max_trafo),
        "total_load_mw": float(net.load.p_mw.sum()),
        "renewable_p_mw": float(net.sgen.p_mw.sum()),
        "slack_p_mw": float(net.res_ext_grid.p_mw.sum()),
        "slack_q_mvar": float(net.res_ext_grid.q_mvar.sum()),
        "total_losses_mw": float(
            net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()
        ),
        "undervoltage_buses": undervoltage_buses,
        "overvoltage_buses": overvoltage_buses,
        "overloaded_lines": overloaded_lines,
        "overloaded_transformers": overloaded_trafos,
        "n_undervoltage": len(undervoltage_buses),
        "n_overvoltage": len(overvoltage_buses),
        "n_line_overload": len(overloaded_lines),
        "n_trafo_overload": len(overloaded_trafos),
        "n_violations": n_violations,
        "has_undervoltage": len(undervoltage_buses) > 0,
        "has_overvoltage": len(overvoltage_buses) > 0,
        "has_overload": (
            len(overloaded_lines) + len(overloaded_trafos)
        ) > 0,
    }


def capture_profiles(
    net,
) -> Tuple[np.ndarray, np.ndarray]:

    return (
        net.res_bus.vm_pu.values.copy(),
        np.concatenate(
            [
                net.res_line.loading_percent.values,
                net.res_trafo.loading_percent.values,
            ]
        ),
    )


def compute_consequences(
    pre: Dict[str, Any],
    post: Dict[str, Any],
) -> Dict[str, Any]:
    """
    What CHANGED, expressed as deltas and as newly appearing violations.

    "What changed?" is the question the execution plan wants the LLM to
    answer, so the answer is computed here rather than left implicit in two
    snapshots. A component that was already at its limit before the event
    is not a consequence OF the event, and the `new_*` lists enforce that
    distinction.
    """

    def newly(
        key: str,
    ) -> List[str]:

        return sorted(set(post[key]) - set(pre[key]))

    new_undervoltage = newly("undervoltage_buses")
    new_overvoltage = newly("overvoltage_buses")
    new_line = newly("overloaded_lines")
    new_trafo = newly("overloaded_transformers")

    effects: List[str] = []

    for bus in new_undervoltage:
        effects.append(f"{bus} undervoltage")

    for bus in new_overvoltage:
        effects.append(f"{bus} overvoltage")

    for branch in new_line + new_trafo:
        effects.append(f"{branch} overload")

    return {
        "delta_v_min_pu": post["v_min_pu"] - pre["v_min_pu"],
        "delta_v_max_pu": post["v_max_pu"] - pre["v_max_pu"],
        "delta_peak_loading_percent": (
            post["peak_branch_loading_percent"]
            - pre["peak_branch_loading_percent"]
        ),
        "delta_slack_p_mw": post["slack_p_mw"] - pre["slack_p_mw"],
        "delta_slack_q_mvar": post["slack_q_mvar"] - pre["slack_q_mvar"],
        "delta_losses_mw": (
            post["total_losses_mw"] - pre["total_losses_mw"]
        ),
        "delta_load_mw": post["total_load_mw"] - pre["total_load_mw"],
        "delta_renewable_mw": (
            post["renewable_p_mw"] - pre["renewable_p_mw"]
        ),
        "new_undervoltage_buses": new_undervoltage,
        "new_overvoltage_buses": new_overvoltage,
        "new_overloaded_lines": new_line,
        "new_overloaded_transformers": new_trafo,
        "n_new_violations": (
            len(new_undervoltage)
            + len(new_overvoltage)
            + len(new_line)
            + len(new_trafo)
        ),
        "effects": effects,
        "effect_summary": (
            "; ".join(effects) if effects else "no limit violations"
        ),
    }


# =============================================================================
# =============================================================================
# PART 4 — OPERATING-POINT HANDLING
# =============================================================================

def apply_operating_point(
    net,
    stage3,
    handles: Dict[str, str],
    base_p: np.ndarray,
    base_q: np.ndarray,
    factors: np.ndarray,
    row: pd.Series,
) -> None:
    """
    Put the network into a stored Stage-4 operating point.

    Identical arithmetic to Stage 4's own applier — Stage 4 proved that a
    stored row reconstructs its state exactly, and this stage depends on
    that guarantee for every pre-event condition it claims.
    """

    net.load["p_mw"] = base_p * factors
    net.load["q_mvar"] = base_q * factors

    stage3.set_pv_output(net, handles["pv_id"], float(row.solar_fraction))
    stage3.set_wind_output(net, handles["wind_id"], float(row.wind_fraction))

    stage3.set_bess_power(net, handles["bess_id"], float(row.bess_p_mw))
    stage3.set_bess_soc(net, handles["bess_id"], float(row.bess_soc))


def select_operating_point(
    rng: np.random.Generator,
    points: pd.DataFrame,
    bias: str = "any",
) -> int:
    """
    Choose a pre-event operating point, optionally biased.

    Bias is about making the search efficient, not about faking anything.
    Hunting for overvoltage from a heavily loaded winter evening wastes
    solves; starting from a light-load high-renewable state finds it sooner.
    The disturbance still has to do the work.
    """

    if bias == "high_load":

        pool = points.index[
            points.load_scale >= points.load_scale.quantile(0.70)
        ]

    elif bias == "low_load":

        pool = points.index[
            (points.load_scale <= points.load_scale.quantile(0.30))
            & (points.solar_fraction >= 0.5)
        ]

    elif bias == "low_soc":

        pool = points.index[points.bess_soc <= 0.30]

    else:

        pool = points.index

    if not len(pool):

        pool = points.index

    return int(rng.choice(np.asarray(pool)))


# =============================================================================
# =============================================================================
# PART 5 — EVENT BUILDERS
# =============================================================================
#
# One builder per class. Each returns an injected-event dictionary; none of
# them touches the network or knows anything about outcomes.
#
# =============================================================================

def build_e0(
    rng: np.random.Generator,
    catalogue: Dict[str, Any],
) -> Dict[str, Any]:

    return make_null_event()


def build_load_event(
    rng: np.random.Generator,
    catalogue: Dict[str, Any],
    magnitudes: Sequence[float],
) -> Dict[str, Any]:
    """
    Shared by E1 and E2: the only difference is the sign of the magnitude.
    """

    scope = str(
        rng.choice(["single_bus", "region", "system"], p=[0.5, 0.3, 0.2])
    )

    percent = float(rng.choice(np.asarray(magnitudes)))

    if scope == "single_bus":

        targets = [str(rng.choice(np.asarray(catalogue["load_ids"])))]

    elif scope == "region":

        region = str(rng.choice(np.asarray(list(REGIONS))))

        targets = [
            cid
            for cid, bus in zip(
                catalogue["load_ids"], catalogue["load_buses"]
            )
            if bus in REGIONS[region]
        ]

    else:

        targets = list(catalogue["load_ids"])

    event = make_load_event(targets, percent, scope)

    if scope == "region":

        event["region"] = region

    return event


def build_e3(
    rng: np.random.Generator,
    catalogue: Dict[str, Any],
) -> Dict[str, Any]:

    return make_line_outage_event(
        str(rng.choice(np.asarray(catalogue["line_ids"])))
    )


def build_e4(
    rng: np.random.Generator,
    catalogue: Dict[str, Any],
) -> Dict[str, Any]:

    return make_generator_outage_event(
        str(rng.choice(np.asarray(catalogue["gen_ids"])))
    )


def build_e5(
    rng: np.random.Generator,
    catalogue: Dict[str, Any],
) -> Dict[str, Any]:

    severity = str(rng.choice(np.asarray(list(RAMP_SEVERITY))))

    delta = RAMP_SEVERITY[severity]

    if rng.random() < 0.5:

        delta = -delta

    choice = rng.random()

    if choice < 0.4:
        targets = [catalogue["pv_id"]]

    elif choice < 0.8:
        targets = [catalogue["wind_id"]]

    else:
        targets = [catalogue["pv_id"], catalogue["wind_id"]]

    return make_ramp_event(targets, delta, severity)


# -----------------------------------------------------------------------------
# OUTCOME-CLASS LADDERS
# -----------------------------------------------------------------------------
#
# For E6, E7 and E8 the builder returns an ORDERED LIST of candidate
# disturbances, mildest first. The generator walks the list and stops at the
# first one that actually produces the target outcome. That is what makes
# these classes physically caused rather than assigned: the corpus keeps the
# smallest disturbance that provably did it.
#
# -----------------------------------------------------------------------------

def ladder_e6(
    rng: np.random.Generator,
    catalogue: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Undervoltage. Mechanisms that drain reactive support or lengthen the
    electrical distance to generation.
    """

    ladder: List[Dict[str, Any]] = []

    weak_lines = [
        cid
        for cid in ("line_9_14", "line_6_13", "line_13_14", "line_9_10")
        if cid in catalogue["line_ids"]
    ]

    condensers = [
        cid for cid in catalogue["gen_ids"] if cid != "gen_G2"
    ]

    for percent in (20.0, 30.0, 40.0):

        region = "lv_south" if rng.random() < 0.5 else "lv_north"

        targets = [
            cid
            for cid, bus in zip(
                catalogue["load_ids"], catalogue["load_buses"]
            )
            if bus in REGIONS[region]
        ]

        event = make_load_event(targets, percent, "region")
        event["region"] = region

        ladder.append(event)

    if condensers:

        ladder.append(
            make_generator_outage_event(
                str(rng.choice(np.asarray(condensers)))
            )
        )

    for cid in weak_lines:

        ladder.append(make_line_outage_event(cid))

    # Escalation: an outage on top of a regional surge.

    if weak_lines:

        ladder.append(
            make_compound_event(
                [
                    make_line_outage_event(weak_lines[0]),
                    make_load_event(
                        [
                            cid
                            for cid, bus in zip(
                                catalogue["load_ids"],
                                catalogue["load_buses"],
                            )
                            if bus in REGIONS["lv_south"]
                        ],
                        30.0,
                        "region",
                    ),
                ],
                "undervoltage_escalation",
            )
        )

    return ladder


def ladder_e7(
    rng: np.random.Generator,
    catalogue: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Overvoltage. Reactive mechanisms only — see the header for why light
    load and high DG are not enough on this network.
    """

    ladder: List[Dict[str, Any]] = []

    controllers = list(catalogue["gen_ids"]) + list(catalogue["slack_ids"])

    for shift in AVR_SHIFT_LADDER:

        ladder.append(make_avr_event(controllers, shift))

    for factor in SHUNT_FACTOR_LADDER:

        ladder.append(make_shunt_event(factor))

    # Both together, for the cases where neither alone quite gets there.

    ladder.append(
        make_compound_event(
            [
                make_avr_event(controllers, 0.02),
                make_shunt_event(3.0),
            ],
            "overvoltage_escalation",
        )
    )

    return ladder


def ladder_e8(
    rng: np.random.Generator,
    catalogue: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Thermal overload. Outages on the heavily loaded corridors, and demand
    surges on top of an already loaded system.
    """

    ladder: List[Dict[str, Any]] = []

    corridors = [
        cid
        for cid in (
            "line_2_3",
            "line_1_5",
            "line_2_4",
            "line_4_5",
            "line_1_2",
            "line_3_4",
        )
        if cid in catalogue["line_ids"]
    ]

    for percent in (20.0, 30.0, 40.0):

        ladder.append(
            make_load_event(list(catalogue["load_ids"]), percent, "system")
        )

    for cid in corridors:

        ladder.append(make_line_outage_event(cid))

    if corridors:

        ladder.append(
            make_compound_event(
                [
                    make_line_outage_event(
                        str(rng.choice(np.asarray(corridors)))
                    ),
                    make_load_event(
                        list(catalogue["load_ids"]), 20.0, "system"
                    ),
                ],
                "overload_escalation",
            )
        )

    return ladder


def build_e9(
    rng: np.random.Generator,
    catalogue: Dict[str, Any],
    pattern: str,
) -> Tuple[Dict[str, Any], str]:
    """
    Compound events — the execution plan's three patterns.

    These are meant to be the hardest scenarios in the corpus, because the
    model has to separate two simultaneous causes rather than match one
    signature.

    The pattern is chosen by the CALLER, not here, because pattern C is
    defined as "outage + high load + BESS low SOC" and the low state of
    charge is a property of the PRE-EVENT operating point, not of the
    injection. The caller therefore has to know the pattern before it picks
    the operating point, or pattern C would end up on batteries that are
    perfectly capable of helping — which is the opposite of the intended
    scenario.
    """

    if pattern == "A":

        components = [
            make_line_outage_event(
                str(rng.choice(np.asarray(catalogue["line_ids"])))
            ),
            make_load_event(
                list(catalogue["load_ids"]),
                float(rng.choice(np.asarray([15.0, 20.0, 25.0]))),
                "system",
            ),
        ]

        label = "A_outage_plus_load_increase"

    elif pattern == "B":

        components = [
            make_generator_outage_event(
                str(rng.choice(np.asarray(catalogue["gen_ids"])))
            ),
            make_ramp_event(
                [catalogue["pv_id"], catalogue["wind_id"]],
                -0.60,
                "large",
            ),
        ]

        label = "B_generator_outage_plus_renewable_collapse"

    else:

        components = [
            make_line_outage_event(
                str(rng.choice(np.asarray(catalogue["line_ids"])))
            ),
            make_load_event(
                list(catalogue["load_ids"]),
                20.0,
                "system",
            ),
        ]

        label = "C_outage_plus_high_load_with_depleted_storage"

    return make_compound_event(components, label), pattern


# =============================================================================
# =============================================================================
# PART 6 — SCENARIO GENERATION
# =============================================================================

def build_catalogue(
    net,
) -> Dict[str, Any]:
    """
    The component IDs each builder may choose from.
    """

    return {
        "load_ids": list(net.load.cid),
        "load_buses": [int(b) for b in net.load.bus_ieee],
        "line_ids": list(net.line.cid),
        "trafo_ids": list(net.trafo.cid),
        "gen_ids": list(net.gen.cid),
        "slack_ids": list(net.ext_grid.cid),
        "bus_ids": list(net.bus.cid),
        "pv_id": str(net.sgen.cid.iloc[0]),
        "wind_id": str(net.sgen.cid.iloc[1]),
        "bess_id": str(net.storage.cid.iloc[0]),
    }


def outcome_achieved(
    event_class: str,
    state: Dict[str, Any],
) -> bool:

    if event_class == "E6":
        return bool(state["has_undervoltage"])

    if event_class == "E7":
        return bool(state["has_overvoltage"])

    if event_class == "E8":
        return bool(state["has_overload"])

    return True


def generate_scenarios(
    net,
    stage3,
    catalogue: Dict[str, Any],
    points: pd.DataFrame,
    factors: pd.DataFrame,
    pre_voltages: pd.DataFrame,
    pre_loadings: pd.DataFrame,
    results: Dict[str, bool],
) -> Dict[str, Any]:
    """
    Build the corpus.

    For every scenario: restore the network, load the pre-event operating
    point, take the pre-event measurement, apply the injected event, solve,
    measure the post-event state, and compute what changed.
    """

    print_subheader(
        f"GENERATING {N_PER_CLASS} SCENARIOS PER CLASS "
        f"({N_PER_CLASS * len(EVENT_CLASSES)} total)"
    )

    rng = np.random.default_rng(MASTER_SEED)

    snapshot = snapshot_network(net)

    base_p = net.load.p_mw.values.copy()
    base_q = net.load.q_mvar.values.copy()

    handles = {
        "pv_id": catalogue["pv_id"],
        "wind_id": catalogue["wind_id"],
        "bess_id": catalogue["bess_id"],
    }

    bus_ids = list(net.bus.cid)
    branch_ids = list(net.line.cid) + list(net.trafo.cid)

    factor_matrix = factors.drop(columns=["op_id"]).values
    voltage_matrix = pre_voltages.drop(columns=["op_id"]).values
    loading_matrix = pre_loadings.drop(columns=["op_id"]).values

    bias_by_class = {
        "E6": "high_load",
        "E7": "low_load",
        "E8": "high_load",
        "E9": "any",
    }

    records: List[Dict[str, Any]] = []
    nested: List[Dict[str, Any]] = []

    post_voltage_rows: List[np.ndarray] = []
    post_loading_rows: List[np.ndarray] = []

    search_effort: Dict[str, List[int]] = {
        key: [] for key in OUTCOME_CLASSES
    }

    abandoned: Dict[str, int] = {key: 0 for key in EVENT_CLASSES}

    counter = 0

    start = time.time()

    for event_class in EVENT_CLASSES:

        produced = 0

        guard = 0

        while produced < N_PER_CLASS:

            guard += 1

            if guard > N_PER_CLASS * (MAX_OPERATING_POINT_RETRIES + 4):

                abandoned[event_class] += N_PER_CLASS - produced

                break

            bias = bias_by_class.get(event_class, "any")

            pending_pattern = ""

            if event_class == "E9":

                pending_pattern = str(
                    rng.choice(np.asarray(["A", "B", "C"]))
                )

                # Pattern C is "outage + high load + DEPLETED STORAGE", so
                # it must land on an operating point whose battery is
                # actually depleted. Drawing the pattern first is what makes
                # that possible.

                bias = "low_soc" if pending_pattern == "C" else "any"

            index = select_operating_point(rng, points, bias)

            row = points.iloc[index]

            # ---- pre-event condition ---------------------------------------

            restore_network(net, snapshot)

            apply_operating_point(
                net=net,
                stage3=stage3,
                handles=handles,
                base_p=base_p,
                base_q=base_q,
                factors=factor_matrix[index],
                row=row,
            )

            pre_state = {
                "v_min_pu": float(row.v_min_pu),
                "v_max_pu": float(row.v_max_pu),
                "v_min_bus": str(row.v_min_bus),
                "v_max_bus": "",
                "max_line_loading_percent": float(
                    row.max_line_loading_percent
                ),
                "max_trafo_loading_percent": float(
                    row.max_trafo_loading_percent
                ),
                "peak_branch_loading_percent": float(
                    row.peak_branch_loading_percent
                ),
                "total_load_mw": float(row.total_load_mw),
                "renewable_p_mw": float(row.renewable_p_mw),
                "slack_p_mw": float(row.slack_p_mw),
                "slack_q_mvar": float(row.slack_q_mvar),
                "total_losses_mw": float(row.total_losses_mw),
                "undervoltage_buses": [],
                "overvoltage_buses": [],
                "overloaded_lines": [],
                "overloaded_transformers": [],
                "n_undervoltage": 0,
                "n_overvoltage": 0,
                "n_line_overload": 0,
                "n_trafo_overload": 0,
                "n_violations": 0,
                "has_undervoltage": False,
                "has_overvoltage": False,
                "has_overload": False,
            }

            # ---- injected event --------------------------------------------

            compound_pattern = ""

            if event_class in OUTCOME_CLASSES:

                ladder = {
                    "E6": ladder_e6,
                    "E7": ladder_e7,
                    "E8": ladder_e8,
                }[event_class](rng, catalogue)

                event = None
                post_state = None
                attempts = 0

                for candidate in ladder[:MAX_SEARCH_ATTEMPTS]:

                    attempts += 1

                    restore_network(net, snapshot)

                    apply_operating_point(
                        net=net,
                        stage3=stage3,
                        handles=handles,
                        base_p=base_p,
                        base_q=base_q,
                        factors=factor_matrix[index],
                        row=row,
                    )

                    apply_injected_event(net, candidate)

                    if not solve(net):
                        continue

                    candidate_state = measure_state(net)

                    if outcome_achieved(event_class, candidate_state):

                        event = candidate
                        post_state = candidate_state

                        break

                if event is None:

                    continue

                search_effort[event_class].append(attempts)

            else:

                if event_class == "E0":
                    event = build_e0(rng, catalogue)

                elif event_class == "E1":
                    event = build_load_event(
                        rng, catalogue, LOAD_SURGE_PERCENT
                    )

                elif event_class == "E2":
                    event = build_load_event(
                        rng, catalogue, LOAD_DROP_PERCENT
                    )

                elif event_class == "E3":
                    event = build_e3(rng, catalogue)

                elif event_class == "E4":
                    event = build_e4(rng, catalogue)

                elif event_class == "E5":
                    event = build_e5(rng, catalogue)

                else:
                    event, compound_pattern = build_e9(
                        rng, catalogue, pending_pattern
                    )

                apply_injected_event(net, event)

                if not solve(net):

                    abandoned[event_class] += 1

                    continue

                post_state = measure_state(net)

            # ---- E0 must actually be normal --------------------------------
            #
            # The execution plan is explicit: do not label a state normal,
            # verify it. A point that violates anything is not E0, whatever
            # Stage 4 thought.

            if event_class == "E0" and post_state["n_violations"] > 0:

                abandoned["E0"] += 1

                continue

            post_voltages, post_loadings = capture_profiles(net)

            consequences = compute_consequences(pre_state, post_state)

            counter += 1

            scenario_id = f"{SCENARIO_PREFIX}_{counter:06d}"

            bess_state = stage3.BESS(
                bess_id=catalogue["bess_id"],
                bus=int(stage3.BESS_CONFIG["bus"]),
                p_max_mw=stage3.BESS_CONFIG["p_max_mw"],
                e_max_mwh=stage3.BESS_CONFIG["e_max_mwh"],
                soc=float(row.bess_soc),
                soc_min=stage3.BESS_CONFIG["soc_min"],
                soc_max=stage3.BESS_CONFIG["soc_max"],
                eta_charge=stage3.BESS_CONFIG["eta_charge"],
                eta_discharge=stage3.BESS_CONFIG["eta_discharge"],
            )

            flat = {
                "scenario_id": scenario_id,
                "event_class": event_class,
                "event_name": EVENT_CLASSES[event_class],
                "op_id": str(row.op_id),
                "compound_pattern": compound_pattern,

                "injected_mechanism": event["mechanism"],
                "injected_scope": event["scope"],
                "injected_targets": ";".join(event.get("targets", [])),
                "injected_magnitude_percent": event.get(
                    "magnitude_percent", np.nan
                ),
                "injected_delta_availability": event.get(
                    "delta_availability", np.nan
                ),
                "injected_delta_vm_pu": event.get("delta_vm_pu", np.nan),
                "injected_description": event["description"],

                "pre_load_scale": float(row.load_scale),
                "pre_solar_fraction": float(row.solar_fraction),
                "pre_wind_fraction": float(row.wind_fraction),
                "pre_bess_soc": float(row.bess_soc),
                "pre_bess_p_mw": float(row.bess_p_mw),
                "pre_bess_max_discharge_mw": bess_state.max_discharge_mw(),
                "pre_bess_max_charge_mw": bess_state.max_charge_mw(),
                "pre_v_min_pu": pre_state["v_min_pu"],
                "pre_v_max_pu": pre_state["v_max_pu"],
                "pre_peak_loading_percent": pre_state[
                    "peak_branch_loading_percent"
                ],
                "pre_slack_p_mw": pre_state["slack_p_mw"],
                "pre_total_load_mw": pre_state["total_load_mw"],
                "pre_losses_mw": pre_state["total_losses_mw"],

                "post_v_min_pu": post_state["v_min_pu"],
                "post_v_max_pu": post_state["v_max_pu"],
                "post_v_min_bus": post_state["v_min_bus"],
                "post_v_max_bus": post_state["v_max_bus"],
                "post_peak_loading_percent": post_state[
                    "peak_branch_loading_percent"
                ],
                "post_slack_p_mw": post_state["slack_p_mw"],
                "post_total_load_mw": post_state["total_load_mw"],
                "post_renewable_mw": post_state["renewable_p_mw"],
                "post_losses_mw": post_state["total_losses_mw"],

                "n_undervoltage": post_state["n_undervoltage"],
                "n_overvoltage": post_state["n_overvoltage"],
                "n_line_overload": post_state["n_line_overload"],
                "n_trafo_overload": post_state["n_trafo_overload"],
                "n_violations": post_state["n_violations"],
                "has_undervoltage": post_state["has_undervoltage"],
                "has_overvoltage": post_state["has_overvoltage"],
                "has_overload": post_state["has_overload"],

                "undervoltage_buses": ";".join(
                    post_state["undervoltage_buses"]
                ),
                "overvoltage_buses": ";".join(
                    post_state["overvoltage_buses"]
                ),
                "overloaded_lines": ";".join(
                    post_state["overloaded_lines"]
                ),
                "overloaded_transformers": ";".join(
                    post_state["overloaded_transformers"]
                ),

                "delta_v_min_pu": consequences["delta_v_min_pu"],
                "delta_v_max_pu": consequences["delta_v_max_pu"],
                "delta_peak_loading_percent": consequences[
                    "delta_peak_loading_percent"
                ],
                "delta_slack_p_mw": consequences["delta_slack_p_mw"],
                "delta_losses_mw": consequences["delta_losses_mw"],
                "delta_load_mw": consequences["delta_load_mw"],
                "n_new_violations": consequences["n_new_violations"],
                "effect_summary": consequences["effect_summary"],
            }

            records.append(flat)

            nested.append(
                {
                    "scenario_id": scenario_id,
                    "event_class": event_class,
                    "event_name": EVENT_CLASSES[event_class],
                    "pre_event": {
                        "op_id": str(row.op_id),
                        "load_scale": float(row.load_scale),
                        "solar_fraction": float(row.solar_fraction),
                        "wind_fraction": float(row.wind_fraction),
                        "bess_soc": float(row.bess_soc),
                        "bess_p_mw": float(row.bess_p_mw),
                        "bess_max_discharge_mw": (
                            bess_state.max_discharge_mw()
                        ),
                        "bess_max_charge_mw": bess_state.max_charge_mw(),
                        "state": {
                            key: value
                            for key, value in pre_state.items()
                        },
                    },
                    "injected_event": event,
                    "post_event": {
                        "converged": True,
                        "state": post_state,
                    },
                    "consequences": consequences,
                }
            )

            post_voltage_rows.append(post_voltages)
            post_loading_rows.append(post_loadings)

            produced += 1

            if counter % PROGRESS_EVERY == 0:

                elapsed = time.time() - start

                info(
                    f"  {counter:>5} scenarios   "
                    f"{event_class} {produced:>3}/{N_PER_CLASS}   "
                    f"{1000 * elapsed / counter:>5.1f} ms/scenario"
                )

    elapsed = time.time() - start

    scenarios = pd.DataFrame(records)

    post_voltages_frame = pd.DataFrame(
        np.array(post_voltage_rows),
        columns=bus_ids,
    )

    post_voltages_frame.insert(0, "scenario_id", scenarios.scenario_id)

    post_loading_frame = pd.DataFrame(
        np.array(post_loading_rows),
        columns=branch_ids,
    )

    post_loading_frame.insert(0, "scenario_id", scenarios.scenario_id)

    info(
        f"Generated {len(scenarios)} scenarios in {elapsed:.1f} s "
        f"({1000 * elapsed / max(len(scenarios), 1):.1f} ms each)."
    )

    for event_class in OUTCOME_CLASSES:

        effort = search_effort[event_class]

        if effort:

            info(
                f"  {event_class} search: mean "
                f"{np.mean(effort):.2f} attempts, max {max(effort)} "
                f"(first candidate sufficed "
                f"{100 * np.mean(np.array(effort) == 1):.0f} % of the time)"
            )

    total_abandoned = sum(abandoned.values())

    if total_abandoned:

        info(
            f"  abandoned draws: {total_abandoned} "
            + ", ".join(
                f"{k} {v}" for k, v in abandoned.items() if v
            )
        )

    return {
        "scenarios": scenarios,
        "nested": nested,
        "post_voltages": post_voltages_frame,
        "post_loadings": post_loading_frame,
        "elapsed": elapsed,
        "search_effort": {
            key: (
                {
                    "mean_attempts": float(np.mean(value)),
                    "max_attempts": int(max(value)),
                }
                if value
                else {}
            )
            for key, value in search_effort.items()
        },
        "abandoned": abandoned,
        "snapshot": snapshot,
        "base_p": base_p,
        "base_q": base_q,
        "handles": handles,
        "factor_matrix": factor_matrix,
    }


# =============================================================================
# =============================================================================
# PART 7 — VALIDATION
# =============================================================================

def validate_corpus(
    net,
    stage3,
    bundle: Dict[str, Any],
    points: pd.DataFrame,
    results: Dict[str, bool],
) -> Dict[str, Any]:

    print_subheader(
        "CORPUS VALIDATION"
    )

    scenarios = bundle["scenarios"]

    # ---- structure ---------------------------------------------------------

    check(
        scenarios.scenario_id.is_unique,
        f"All {len(scenarios)} scenario IDs are unique.",
        results,
        "s5_ids_unique",
    )

    well_formed = scenarios.scenario_id.str.match(
        rf"^{SCENARIO_PREFIX}_\d{{6}}$"
    ).all()

    check(
        bool(well_formed),
        f"All scenario IDs match {SCENARIO_PREFIX}_NNNNNN "
        f"(e.g. {scenarios.scenario_id.iloc[0]}).",
        results,
        "s5_id_format",
    )

    check(
        bool(scenarios.op_id.isin(points.op_id).all()),
        "Every scenario references a real Stage-4 operating point.",
        results,
        "s5_op_ids_valid",
    )

    numeric = scenarios.select_dtypes(include=[np.number])

    always_present = [
        column
        for column in numeric.columns
        if not column.startswith("injected_")
    ]

    check(
        not numeric[always_present].isnull().values.any(),
        "No unexpected NaN in the scenario table (injected_* columns are "
        "sparse by design: a line outage has no magnitude_percent).",
        results,
        "s5_no_nan",
    )

    # ---- class coverage ----------------------------------------------------

    counts = scenarios.event_class.value_counts()

    check(
        len(counts) == len(EVENT_CLASSES),
        f"All {len(EVENT_CLASSES)} event classes are represented.",
        results,
        "s5_all_classes_present",
    )

    check(
        int(counts.min()) >= int(0.9 * N_PER_CLASS),
        f"Every class has at least {int(counts.min())} scenarios "
        f"(target {N_PER_CLASS}).",
        results,
        "s5_class_balance",
    )

    # ---- E0 really is normal ----------------------------------------------

    e0 = scenarios[scenarios.event_class == "E0"]

    check(
        bool((e0.n_violations == 0).all()),
        f"All {len(e0)} E0 scenarios were verified normal by power flow, "
        f"not assumed.",
        results,
        "s5_e0_verified_normal",
    )

    check(
        bool((e0.injected_mechanism == "none").all())
        and bool(
            (e0.delta_load_mw.abs() < 1e-9).all()
        ),
        "E0 scenarios contain no injected disturbance at all.",
        results,
        "s5_e0_no_injection",
    )

    # ---- mechanism classes did what they claim ----------------------------

    e1 = scenarios[scenarios.event_class == "E1"]

    check(
        bool((e1.delta_load_mw > 0).all()),
        f"All {len(e1)} E1 scenarios increased total demand "
        f"(+{e1.delta_load_mw.min():.2f} to "
        f"+{e1.delta_load_mw.max():.2f} MW).",
        results,
        "s5_e1_load_increased",
    )

    e2 = scenarios[scenarios.event_class == "E2"]

    check(
        bool((e2.delta_load_mw < 0).all()),
        f"All {len(e2)} E2 scenarios reduced total demand "
        f"({e2.delta_load_mw.min():.2f} to "
        f"{e2.delta_load_mw.max():.2f} MW).",
        results,
        "s5_e2_load_decreased",
    )

    e3 = scenarios[scenarios.event_class == "E3"]

    check(
        bool((e3.injected_mechanism == "line_outage").all())
        and bool(e3.injected_targets.isin(net.line.cid).all()),
        f"All {len(e3)} E3 scenarios outage a real line "
        f"({e3.injected_targets.nunique()} distinct lines used).",
        results,
        "s5_e3_valid_lines",
    )

    e4 = scenarios[scenarios.event_class == "E4"]

    check(
        bool(e4.injected_targets.isin(net.gen.cid).all()),
        f"All {len(e4)} E4 scenarios outage a real generator, never the "
        f"slack ({e4.injected_targets.nunique()} distinct units).",
        results,
        "s5_e4_valid_generators",
    )

    e5 = scenarios[scenarios.event_class == "E5"]

    check(
        bool((e5.injected_mechanism == "renewable_ramp").all()),
        f"All {len(e5)} E5 scenarios ramp a renewable resource "
        f"(delta from {e5.injected_delta_availability.min():+.2f} to "
        f"{e5.injected_delta_availability.max():+.2f} availability).",
        results,
        "s5_e5_valid_ramps",
    )

    # ---- outcome classes were CAUSED, not assigned ------------------------
    #
    # This is the scientific-credibility check the execution plan calls for.
    # Every E6 must genuinely show undervoltage in the solved state, every
    # E7 overvoltage, every E8 an overload. If any of these fail, the corpus
    # contains a fabricated label.

    e6 = scenarios[scenarios.event_class == "E6"]

    check(
        bool(e6.has_undervoltage.all()) and len(e6) > 0,
        f"All {len(e6)} E6 scenarios exhibit detected undervoltage "
        f"(worst {e6.post_v_min_pu.min():.4f} p.u.).",
        results,
        "s5_e6_caused",
    )

    e7 = scenarios[scenarios.event_class == "E7"]

    check(
        bool(e7.has_overvoltage.all()) and len(e7) > 0,
        f"All {len(e7)} E7 scenarios exhibit detected overvoltage "
        f"(worst {e7.post_v_max_pu.max():.4f} p.u.).",
        results,
        "s5_e7_caused",
    )

    e8 = scenarios[scenarios.event_class == "E8"]

    check(
        bool(e8.has_overload.all()) and len(e8) > 0,
        f"All {len(e8)} E8 scenarios exhibit detected thermal overload "
        f"(worst {e8.post_peak_loading_percent.max():.2f} %).",
        results,
        "s5_e8_caused",
    )

    e9 = scenarios[scenarios.event_class == "E9"]

    check(
        bool((e9.injected_mechanism == "compound").all()),
        f"All {len(e9)} E9 scenarios inject two simultaneous mechanisms "
        f"({e9.compound_pattern.nunique()} patterns used).",
        results,
        "s5_e9_compound",
    )

    # ---- separation of injection from consequence -------------------------
    #
    # The injected description must never contain outcome language, and the
    # effect summary must never contain the mechanism. If these ever cross,
    # the corpus has leaked the answer into the question.

    outcome_words = (
        "undervoltage",
        "overvoltage",
        "overload",
        "violation",
    )

    leaked = scenarios.injected_description.str.lower().apply(
        lambda text: any(word in text for word in outcome_words)
    )

    check(
        not bool(leaked.any()),
        "No injected-event description mentions an outcome: the injection "
        "and its consequences stay separate.",
        results,
        "s5_no_outcome_leakage",
    )

    return {
        "counts": counts,
        "e6": e6,
        "e7": e7,
        "e8": e8,
    }


def validate_replay(
    net,
    stage3,
    bundle: Dict[str, Any],
    points: pd.DataFrame,
    results: Dict[str, bool],
    n_checks: int = 40,
) -> None:
    """
    Re-apply stored events to restored pre-event states and confirm they
    reproduce the recorded post-event state exactly.

    This is the strongest check in the stage. It proves the injected event
    is a complete and faithful description of what was done — that nothing
    happened to the network which is not written down in the record. A
    scenario that cannot be replayed is an anecdote, not data.
    """

    print_subheader(
        "EVENT REPLAY"
    )

    scenarios = bundle["scenarios"]
    nested = bundle["nested"]
    post_voltages = bundle["post_voltages"]

    rng = np.random.default_rng(MASTER_SEED + 7)

    indices = rng.choice(
        len(scenarios),
        size=min(n_checks, len(scenarios)),
        replace=False,
    )

    point_index = {
        op_id: i for i, op_id in enumerate(points.op_id)
    }

    worst_v = 0.0
    worst_loading = 0.0
    failures = 0

    for index in indices:

        record = nested[int(index)]

        row = points.iloc[point_index[record["pre_event"]["op_id"]]]

        restore_network(net, bundle["snapshot"])

        apply_operating_point(
            net=net,
            stage3=stage3,
            handles=bundle["handles"],
            base_p=bundle["base_p"],
            base_q=bundle["base_q"],
            factors=bundle["factor_matrix"][
                point_index[record["pre_event"]["op_id"]]
            ],
            row=row,
        )

        apply_injected_event(net, record["injected_event"])

        if not solve(net):

            failures += 1

            continue

        replayed = measure_state(net)

        worst_v = max(
            worst_v,
            abs(
                replayed["v_min_pu"]
                - record["post_event"]["state"]["v_min_pu"]
            ),
        )

        worst_loading = max(
            worst_loading,
            abs(
                replayed["peak_branch_loading_percent"]
                - record["post_event"]["state"][
                    "peak_branch_loading_percent"
                ]
            ),
        )

        profile_gap = float(
            np.abs(
                net.res_bus.vm_pu.values
                - post_voltages.iloc[int(index), 1:].values.astype(float)
            ).max()
        )

        worst_v = max(worst_v, profile_gap)

    check(
        failures == 0 and worst_v <= TOL_REPLAY_PU,
        f"{len(indices)} stored events replayed from their pre-event "
        f"states reproduce the recorded outcome exactly "
        f"(max dV {worst_v:.2e} p.u., max dLoading "
        f"{worst_loading:.2e} %, {failures} failures).",
        results,
        "s5_events_replayable",
    )

    restore_network(net, bundle["snapshot"])


# =============================================================================
# =============================================================================
# PART 8 — REPORTING
# =============================================================================

def build_class_summary(
    scenarios: pd.DataFrame,
) -> pd.DataFrame:

    rows: List[Dict[str, Any]] = []

    for event_class, name in EVENT_CLASSES.items():

        group = scenarios[scenarios.event_class == event_class]

        if not len(group):
            continue

        rows.append(
            {
                "event_class": event_class,
                "event_name": name,
                "n_scenarios": len(group),
                "n_with_violations": int((group.n_violations > 0).sum()),
                "pct_with_violations": round(
                    100.0 * (group.n_violations > 0).mean(), 2
                ),
                "pct_undervoltage": round(
                    100.0 * group.has_undervoltage.mean(), 2
                ),
                "pct_overvoltage": round(
                    100.0 * group.has_overvoltage.mean(), 2
                ),
                "pct_overload": round(
                    100.0 * group.has_overload.mean(), 2
                ),
                "worst_v_min_pu": round(float(group.post_v_min_pu.min()), 4),
                "worst_v_max_pu": round(float(group.post_v_max_pu.max()), 4),
                "worst_loading_percent": round(
                    float(group.post_peak_loading_percent.max()), 2
                ),
                "mean_delta_slack_mw": round(
                    float(group.delta_slack_p_mw.mean()), 3
                ),
            }
        )

    return pd.DataFrame(rows)


def show_class_summary(
    summary: pd.DataFrame,
) -> None:

    print_subheader(
        "SCENARIOS BY EVENT CLASS"
    )

    print(
        f"  {'class':<6} {'name':<26} {'n':>5} {'viol %':>8} "
        f"{'UV %':>7} {'OV %':>7} {'OL %':>7} {'worst V':>9} "
        f"{'worst load %':>13}"
    )

    print("  " + "-" * 94)

    for _, row in summary.iterrows():

        print(
            f"  {row.event_class:<6} {row.event_name:<26} "
            f"{row.n_scenarios:>5} {row.pct_with_violations:>8.1f} "
            f"{row.pct_undervoltage:>7.1f} {row.pct_overvoltage:>7.1f} "
            f"{row.pct_overload:>7.1f} {row.worst_v_min_pu:>9.4f} "
            f"{row.worst_loading_percent:>13.2f}"
        )


def show_overlap(
    scenarios: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Report how often a mechanism class produces an outcome-class symptom.

    This overlap is a property of the taxonomy, not a defect, but it has to
    be measured and stated. A classifier trained on `event_class` is
    learning the injected mechanism; one trained on the consequence flags is
    learning the outcome. They are different tasks and the corpus supports
    both.
    """

    print_subheader(
        "MECHANISM / OUTCOME OVERLAP"
    )

    mechanism = scenarios[scenarios.event_class.isin(MECHANISM_CLASSES)]

    overlap = {
        "n_mechanism_scenarios": int(len(mechanism)),
        "pct_with_any_violation": round(
            100.0 * (mechanism.n_violations > 0).mean(), 2
        ),
        "pct_undervoltage": round(
            100.0 * mechanism.has_undervoltage.mean(), 2
        ),
        "pct_overvoltage": round(
            100.0 * mechanism.has_overvoltage.mean(), 2
        ),
        "pct_overload": round(100.0 * mechanism.has_overload.mean(), 2),
    }

    info(
        f"Of {overlap['n_mechanism_scenarios']} E1-E5 scenarios, "
        f"{overlap['pct_with_any_violation']:.1f} % show at least one "
        f"limit violation:"
    )

    info(
        f"  undervoltage {overlap['pct_undervoltage']:.1f} %   "
        f"overvoltage {overlap['pct_overvoltage']:.1f} %   "
        f"overload {overlap['pct_overload']:.1f} %"
    )

    info(
        "E1-E5 label the INJECTED MECHANISM; E6-E8 label the RESULTING "
        "OUTCOME. They are separate axes and overlap by construction. "
        "State which axis any classification result refers to."
    )

    return overlap


def show_examples(
    scenarios: pd.DataFrame,
) -> None:
    """
    One scenario per class, in the before / injected / after form the
    execution plan asks for.
    """

    print_subheader(
        "EXAMPLE SCENARIOS (one per class)"
    )

    for event_class in EVENT_CLASSES:

        group = scenarios[scenarios.event_class == event_class]

        if not len(group):
            continue

        row = (
            group.sort_values("n_violations", ascending=False).iloc[0]
            if event_class != "E0"
            else group.iloc[0]
        )

        print()

        print(
            f"  {row.scenario_id}   [{row.event_class}] {row.event_name}"
        )

        print(
            f"    PRE       load {100 * row.pre_load_scale:.0f} %, "
            f"solar {100 * row.pre_solar_fraction:.0f} %, "
            f"wind {100 * row.pre_wind_fraction:.0f} %, "
            f"SOC {100 * row.pre_bess_soc:.0f} %"
        )

        print(
            f"              V {row.pre_v_min_pu:.4f} - "
            f"{row.pre_v_max_pu:.4f} p.u., peak loading "
            f"{row.pre_peak_loading_percent:.1f} %"
        )

        print(
            f"    INJECTED  {row.injected_description}"
        )

        print(
            f"    POST      V {row.post_v_min_pu:.4f} - "
            f"{row.post_v_max_pu:.4f} p.u., peak loading "
            f"{row.post_peak_loading_percent:.1f} %"
        )

        print(
            f"    EFFECTS   {row.effect_summary}"
        )


# =============================================================================
# =============================================================================
# PART 9 — SAVE
# =============================================================================

def save_artefacts(
    bundle: Dict[str, Any],
    class_summary: pd.DataFrame,
) -> List[Path]:

    print_subheader(
        "ARTEFACT EXPORT"
    )

    written: List[Path] = []

    bundle["scenarios"].to_csv(SCENARIOS_CSV, index=False)
    written.append(SCENARIOS_CSV)

    with open(SCENARIOS_JSONL, "w", encoding="utf-8") as handle:

        for record in bundle["nested"]:

            handle.write(json.dumps(record) + "\n")

    written.append(SCENARIOS_JSONL)

    bundle["post_voltages"].to_csv(POST_VOLTAGES_FILE, index=False)
    written.append(POST_VOLTAGES_FILE)

    bundle["post_loadings"].to_csv(POST_LOADING_FILE, index=False)
    written.append(POST_LOADING_FILE)

    class_summary.to_csv(CLASS_SUMMARY_FILE, index=False)
    written.append(CLASS_SUMMARY_FILE)

    for path in written:

        info(
            f"Saved: {path}  ({path.stat().st_size:,} bytes)"
        )

    return written


def save_validation_summary(
    results: Dict[str, bool],
) -> None:

    frame = pd.DataFrame(
        {
            "check": list(results.keys()),
            "passed": [bool(v) for v in results.values()],
        }
    )

    frame.to_csv(VALIDATION_FILE, index=False)

    info(
        f"Saved: {VALIDATION_FILE}  "
        f"({int(frame.passed.sum())}/{len(frame)} checks passed)"
    )


def save_stage_metadata(
    bundle: Dict[str, Any],
    class_summary: pd.DataFrame,
    overlap: Dict[str, Any],
    parent_hash: str,
    results: Dict[str, bool],
    elapsed_seconds: float,
) -> Dict[str, Any]:

    scenarios = bundle["scenarios"]

    metadata = {
        "stage": 5,
        "stage_name": STAGE_NAME,
        "network": NETWORK_NAME,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "generation_seconds": round(bundle["elapsed"], 3),

        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandapower": pp.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "numba": NUMBA_AVAILABLE,
        },

        "parent_re_layout_hash_sha256": parent_hash,

        "corpus": {
            "n_scenarios": int(len(scenarios)),
            "n_per_class_target": N_PER_CLASS,
            "seed": MASTER_SEED,
            "id_format": f"{SCENARIO_PREFIX}_NNNNNN",
            "classes": EVENT_CLASSES,
            "class_counts": {
                str(k): int(v)
                for k, v in scenarios.event_class.value_counts().items()
            },
            "distinct_operating_points_used": int(
                scenarios.op_id.nunique()
            ),
        },

        "event_magnitudes": {
            "load_surge_percent": list(LOAD_SURGE_PERCENT),
            "load_drop_percent": list(LOAD_DROP_PERCENT),
            "ramp_severity": RAMP_SEVERITY,
            "regions": REGIONS,
        },

        "outcome_search": bundle["search_effort"],

        "mechanism_outcome_overlap": overlap,

        "class_summary": json.loads(
            class_summary.to_json(orient="records")
        ),

        "design_notes": [
            "The injected event and its consequences are stored in "
            "separate fields and are never merged. Nothing in the "
            "injected_* columns names an outcome.",
            "E6, E7 and E8 are never assigned. A disturbance is applied, "
            "the power flow is solved, and the outcome is detected. The "
            "corpus keeps the mildest disturbance from an escalating "
            "ladder that provably produced the target outcome.",
            "E1-E5 label the injected MECHANISM; E6-E8 label the "
            "resulting OUTCOME. The two axes overlap by construction and "
            "the overlap is measured in mechanism_outcome_overlap.",
            "Low load and high distributed generation cannot produce "
            "overvoltage on this network: the measured ceiling is 1.0347 "
            "p.u. against a 1.05 limit, because four synchronous machines "
            "hold their buses at 1.02-1.03. E7 therefore uses explicit "
            "reactive mechanisms (raised excitation setpoints, capacitor "
            "over-compensation). This is a property of the IEEE 14-bus "
            "system and belongs in the paper's limitations.",
            "No single line outage islands this network; all 15 converge.",
            "Every stored event is replayable: re-applying it to the "
            "restored pre-event state reproduces the recorded post-event "
            "state exactly.",
            "The network is fully restored from a snapshot before every "
            "scenario, so no disturbance can leak into the next one.",
        ],

        "artefacts": {
            "scenarios_csv": str(SCENARIOS_CSV),
            "scenarios_jsonl": str(SCENARIOS_JSONL),
            "post_voltages": str(POST_VOLTAGES_FILE),
            "post_loading": str(POST_LOADING_FILE),
            "class_summary": str(CLASS_SUMMARY_FILE),
        },

        "validation": {k: bool(v) for k, v in results.items()},
    }

    METADATA_OUTPUT_FILE.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    info(f"Saved: {METADATA_OUTPUT_FILE}")

    return metadata


# =============================================================================
# CHECKPOINT
# =============================================================================

def show_checkpoint(
    results: Dict[str, bool],
    parent_hash: str,
    n_scenarios: int,
) -> None:

    print_header(
        "CHECKPOINT 5 — EVENT GENERATOR"
    )

    failures = [k for k, v in results.items() if not v]

    if failures:

        for key in failures:

            failed(f"Failed check: {key}")

        raise RuntimeError(
            f"Checkpoint 5 failed. {len(failures)} check(s) did not pass."
        )

    passed(
        "Stage-3 network and Stage-4 operating-point pool loaded and "
        "verified."
    )

    passed(
        f"{n_scenarios} scenarios generated across all 10 event classes."
    )

    passed(
        "Every scenario stores pre-event condition, injected event, "
        "post-event condition and consequences separately."
    )

    passed(
        "No injected-event description mentions an outcome: cause and "
        "effect never leak into one another."
    )

    passed(
        "E0 scenarios verified normal by power flow rather than assumed."
    )

    passed(
        "E1-E5 mechanisms each verified to have done what they claim."
    )

    passed(
        "E6, E7 and E8 outcomes were physically caused and then detected, "
        "never assigned."
    )

    passed(
        "E9 compound scenarios inject two simultaneous mechanisms."
    )

    passed(
        "Stored events replay exactly from their pre-event states."
    )

    passed(
        "Network fully restored between scenarios; no cross-contamination."
    )

    passed(
        "Corpus, nested records, post-event profiles and class summary "
        "saved."
    )

    print()

    print(f"PARENT RE LAYOUT HASH: {parent_hash}")

    print()

    print("STAGE 5 STATUS: PASSED")

    print()

    print("NEXT STAGE:")

    print(
        "STAGE 6 — SCENARIO DATASET ASSEMBLY "
        "(structure this corpus into the study dataset)"
    )


# =============================================================================
# MAIN
# =============================================================================

def load_stage3_module():

    if not STAGE3_FILE.exists():

        raise FileNotFoundError(
            f"{STAGE3_FILE} not found. Run Stage 5 from the same working "
            f"directory as Stages 1-4. Current: {Path.cwd()}"
        )

    spec = importlib.util.spec_from_file_location(
        STAGE3_MODULE_NAME,
        STAGE3_FILE,
    )

    module = importlib.util.module_from_spec(spec)

    sys.modules[STAGE3_MODULE_NAME] = module

    spec.loader.exec_module(module)

    return module


def main() -> int:

    start = time.time()

    stage_success = False

    results: Dict[str, bool] = {}

    print_header(STAGE_NAME)

    info(f"Started at {datetime.now().isoformat(timespec='seconds')}")

    info(f"Python {platform.python_version()} on {platform.platform()}")

    info(
        f"pandapower {pp.__version__}  pandas {pd.__version__}  "
        f"numpy {np.__version__}  numba "
        f"{'on' if NUMBA_AVAILABLE else 'off'}"
    )

    info(f"Working directory: {Path.cwd()}")

    try:

        print_subheader("INPUTS")

        missing = [
            str(path)
            for path in (
                NET_RE_FILE,
                RE_HASH_FILE,
                POINTS_FILE,
                FACTORS_FILE,
                VOLTAGES_FILE,
                LOADING_FILE,
            )
            if not path.exists()
        ]

        if missing:

            raise FileNotFoundError(
                "Stage 3/4 artefacts are missing: "
                + ", ".join(missing)
                + ". Run Stages 1-4 first."
            )

        stage3 = load_stage3_module()

        net = pp.from_json(NET_RE_FILE)

        parent_hash = RE_HASH_FILE.read_text(encoding="utf-8").strip()

        actual_hash = stage3.compute_re_layout_hash(net)

        check(
            parent_hash == actual_hash,
            "Loaded network matches the Stage-3 renewable fingerprint.",
            results,
            "s5_parent_hash_match",
        )

        require(
            parent_hash == actual_hash,
            "Network fingerprint does not match Stage 3. Re-run Stage 3 "
            "and Stage 4 before Stage 5.",
        )

        points = pd.read_csv(POINTS_FILE)
        factors = pd.read_csv(FACTORS_FILE)
        pre_voltages = pd.read_csv(VOLTAGES_FILE)
        pre_loadings = pd.read_csv(LOADING_FILE)

        check(
            len(points) == len(factors) == len(pre_voltages),
            f"Stage-4 pool loaded: {len(points)} operating points with "
            f"matching factor and profile matrices.",
            results,
            "s5_pool_consistent",
        )

        catalogue = build_catalogue(net)

        info(
            f"Catalogue: {len(catalogue['load_ids'])} loads, "
            f"{len(catalogue['line_ids'])} lines, "
            f"{len(catalogue['gen_ids'])} generators, "
            f"{len(REGIONS)} regions."
        )

        bundle = generate_scenarios(
            net=net,
            stage3=stage3,
            catalogue=catalogue,
            points=points,
            factors=factors,
            pre_voltages=pre_voltages,
            pre_loadings=pre_loadings,
            results=results,
        )

        validate_corpus(
            net=net,
            stage3=stage3,
            bundle=bundle,
            points=points,
            results=results,
        )

        validate_replay(
            net=net,
            stage3=stage3,
            bundle=bundle,
            points=points,
            results=results,
        )

        class_summary = build_class_summary(bundle["scenarios"])

        written = save_artefacts(bundle, class_summary)

        save_validation_summary(results)

        overlap = show_overlap(bundle["scenarios"])

        elapsed = time.time() - start

        save_stage_metadata(
            bundle=bundle,
            class_summary=class_summary,
            overlap=overlap,
            parent_hash=parent_hash,
            results=results,
            elapsed_seconds=elapsed,
        )

        show_class_summary(class_summary)

        show_examples(bundle["scenarios"])

        show_checkpoint(
            results,
            parent_hash,
            len(bundle["scenarios"]),
        )

        stage_success = True

    except Exception as error:  # noqa: BLE001

        print()

        failed(f"{type(error).__name__}: {error}")

        print()

        print("STAGE 5 STATUS: FAILED")

        print()

        print(
            "Do not proceed to Stage 6. This corpus is the ground truth "
            "for every downstream evaluation."
        )

    finally:

        elapsed = time.time() - start

        print_subheader("RUN STATISTICS")

        info(f"Elapsed time: {elapsed:.2f} s")

        if PSUTIL_AVAILABLE:

            process = psutil.Process(os.getpid())

            info(
                f"Peak memory: "
                f"{process.memory_info().rss / (1024 ** 2):.1f} MB"
            )

        else:

            info("Peak memory: psutil not installed, not measured.")

        info(f"Final status: {'PASSED' if stage_success else 'FAILED'}")

        print_header("END OF STAGE 5")

    return 0 if stage_success else 1


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    raise SystemExit(main())
