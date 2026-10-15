"""Numerical results for Question 2.(b) and Question 3.(e)–(f).

Run from the repository root:

    python analyze_q2b_q3ef.py

Writes tables and figures under results/analysis/.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import load_question
from src.model import FlexibleConsumerModel

OUT = Path(__file__).resolve().parent / "results" / "analysis"
TOL = 1e-5


def solve(data):
    return FlexibleConsumerModel(data).build().solve()


def breakdown(data, results) -> dict:
    """Split the minimised objective into procurement cost and preference cost."""
    h = results.hourly
    p_imp = data.energy_price + data.import_tariff
    p_exp = data.energy_price - data.export_tariff
    procurement = float(
        (data.pv_marginal_cost * h["pv"] + p_imp * h["import"] - p_exp * h["export"]).sum()
    )
    ref = data.reference_load
    deviation = h["load"] - ref if ref is not None else None
    if data.linear_disutility is not None:
        disutility = float(data.linear_disutility * deviation.abs().sum())
        utility = 0.0
    elif data.quadratic_disutility is not None:
        disutility = float(data.quadratic_disutility * (deviation ** 2).sum())
        utility = 0.0
    elif data.consumption_utility is not None:
        disutility = 0.0
        utility = float(data.consumption_utility * h["load"].sum())
    else:
        disutility = 0.0
        utility = 0.0
    abs_dev = float(deviation.abs().sum()) if deviation is not None else float("nan")
    # "Deviation bound binding": load is on an hourly bound and off its reference,
    # so the bound is what stops a larger deviation. The epigraph constraints on the
    # auxiliary deviation variable are binding in every hour whenever c^L > 0.
    if deviation is not None:
        on_bound = (h["load"] <= data.load_min_kWh + TOL) | (h["load"] >= data.load_max_kWh - TOL)
        n_dev_bound = int((on_bound & (deviation.abs() > TOL)).sum())
        n_off_ref = int((deviation.abs() > TOL).sum())
        n_above_ref = int((deviation > TOL).sum())
        n_below_ref = int((deviation < -TOL).sum())
    else:
        n_dev_bound = n_off_ref = n_above_ref = n_below_ref = 0
    return {
        "objective": results.objective,
        "procurement": procurement,
        "disutility": disutility,
        "utility": utility,
        "energy": float(h["load"].sum()),
        "import": float(h["import"].sum()),
        "export": float(h["export"].sum()),
        "pv": float(h["pv"].sum()),
        "abs_deviation": abs_dev,
        "n_off_reference": n_off_ref,
        "n_deviation_bound_binding": n_dev_bound,
        "n_above_reference": n_above_ref,
        "n_below_reference": n_below_ref,
        "mu_min_energy": results.duals.get("min_energy", float("nan")),
    }


def effective_prices(data):
    return data.energy_price + data.import_tariff, data.energy_price - data.export_tariff


def linear_sweep(base) -> pd.DataFrame:
    """Sweep c^L across the procurement-cost thresholds of the dataset."""
    p_imp, p_exp = effective_prices(base)
    # Include 0, the base value, PV cost, and every distinct effective price,
    # plus points just on either side so the jump (not the tie) is visible.
    anchors = sorted(set(np.round(np.concatenate([p_imp, p_exp, [base.pv_marginal_cost]]), 6)))
    values = [0.0, 0.2, 0.5]
    for a in anchors:
        values.extend([round(a - 0.02, 4), round(a, 4), round(a + 0.02, 4)])
    values.extend([base.linear_disutility, 3.5, 4.0])
    values = sorted({v for v in values if v >= 0})

    rows = []
    for c in values:
        data = replace(base, linear_disutility=float(c))
        res = solve(data)
        row = {"c_L": c, **breakdown(data, res)}
        rows.append(row)
        res.save(OUT / "q2_linear_sweep", tag=f"cL_{c:.4f}".replace(".", "p"))
    return pd.DataFrame(rows)


def classify_linear_hour(data, load, t) -> str:
    """Analytical regime label used to check the solved base case. See the report."""
    p_imp = data.energy_price[t] + data.import_tariff
    p_exp = data.energy_price[t] - data.export_tariff
    c_pv = data.pv_marginal_cost
    c_L = data.linear_disutility
    ref = float(data.reference_load[t])
    pv = float(data.pv_available[t])
    lo = data.load_min_kWh
    if c_pv <= p_exp:
        region = "I"          # PV fully used, surplus exported; MCP = p_exp below PV, p_imp above
        m_low, m_high = p_exp, p_imp
    elif c_pv <= p_imp:
        region = "II"         # PV used only on-site; MCP = c_pv below PV, p_imp above
        m_low, m_high = c_pv, p_imp
    else:
        region = "III"        # PV never used; MCP = p_imp everywhere
        m_low, m_high = p_imp, p_imp
    # Load never exceeds the reference when every marginal procurement cost is positive.
    if ref <= pv or region == "III":
        # One piece between lo and ref, marginal cost m_low (region III: m_low = p_imp).
        threshold = m_low
        if c_L > threshold + 1e-9:
            expect = ref
        elif c_L < threshold - 1e-9:
            expect = lo
        else:
            expect = None  # whole segment optimal
        candidate = "ref_or_min"
    else:
        # Two pieces: (lo, PV] costs m_low, (PV, ref] costs m_high.
        if c_L > m_high + 1e-9:
            expect = ref
        elif c_L > m_low + 1e-9:
            expect = pv
        elif c_L < m_low - 1e-9:
            expect = lo
        else:
            expect = None
        candidate = "ref_or_pv_or_min"
    return region, expect, candidate, m_low, m_high


def q3_versus_q2(q2_data, q3_data):
    r2 = solve(q2_data)
    r3 = solve(q3_data)
    r2.save(OUT / "q3e")
    # tag the Q3 run distinctly; question name is already in the stem
    r3.save(OUT / "q3e")
    h = pd.DataFrame({
        "price": q3_data.energy_price,
        "p_imp": q3_data.energy_price + q3_data.import_tariff,
        "p_exp": q3_data.energy_price - q3_data.export_tariff,
        "pv_available": q3_data.pv_available,
        "reference": q3_data.reference_load,
        "load_Q2c": r2.hourly["load"].to_numpy(),
        "pv_Q2c": r2.hourly["pv"].to_numpy(),
        "import_Q2c": r2.hourly["import"].to_numpy(),
        "export_Q2c": r2.hourly["export"].to_numpy(),
        "load_Q3": r3.hourly["load"].to_numpy(),
        "pv_Q3": r3.hourly["pv"].to_numpy(),
        "import_Q3": r3.hourly["import"].to_numpy(),
        "export_Q3": r3.hourly["export"].to_numpy(),
    })
    h["delta_load"] = h["load_Q3"] - h["load_Q2c"]
    h["above_ref_Q3"] = h["load_Q3"] - h["reference"]
    return r2, r3, h


def sweep_emin(base, values) -> pd.DataFrame:
    rows = []
    profiles = {}
    for e in values:
        data = replace(base, min_daily_energy_kWh=float(e))
        res = solve(data)
        rows.append({"E_min": e, **breakdown(data, res)})
        profiles[e] = res.hourly["load"].to_numpy()
        res.save(OUT / "q3f_emin", tag=f"E_{e:.1f}".replace(".", "p"))
    return pd.DataFrame(rows), profiles


def sweep_cq(base, values) -> pd.DataFrame:
    rows = []
    profiles = {}
    for c in values:
        data = replace(base, quadratic_disutility=float(c))
        res = solve(data)
        rows.append({"c_Q": c, **breakdown(data, res)})
        profiles[c] = res.hourly["load"].to_numpy()
        res.save(OUT / "q3f_cq", tag=f"cQ_{c:.2f}".replace(".", "p"))
    return pd.DataFrame(rows), profiles


def plot_linear_sweep(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(df["c_L"], df["energy"], "o-", ms=3.5, label="daily energy")
    ax.plot(df["c_L"], df["abs_deviation"], "s-", ms=3.5, label="absolute deviation")
    ax.set(xlabel=r"linear disutility coefficient $c^{L}$ [DKK/kWh]", ylabel="kWh/day")
    ax.legend()
    ax.set_title("Q2(b): daily energy and deviation versus $c^{L}$")
    fig.tight_layout()
    fig.savefig(OUT / "q2b_sweep_energy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(df["c_L"], df["procurement"], "o-", ms=3.5, label="procurement cost")
    ax.plot(df["c_L"], df["disutility"], "s-", ms=3.5, label="disutility")
    ax.plot(df["c_L"], df["objective"], "k.-", ms=3.5, label="total (objective)")
    ax.set(xlabel=r"$c^{L}$ [DKK/kWh]", ylabel="DKK/day")
    ax.legend()
    ax.set_title("Q2(b): cost breakdown versus $c^{L}$")
    fig.tight_layout()
    fig.savefig(OUT / "q2b_sweep_cost.png", dpi=150)
    plt.close(fig)


def plot_q3_comparison(h: pd.DataFrame) -> None:
    hours = np.arange(len(h))
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.step(hours, h["reference"], where="mid", color="0.4", ls=":", label="reference")
    ax.step(hours, h["load_Q2c"], where="mid", color="C0", label="Q2(c) unconstrained")
    ax.step(hours, h["load_Q3"], where="mid", color="C3", label="Q3 base case")
    ax.set(xlabel="hour", ylabel="kWh/h", title="Q3(e): load against the reference and against Q2(c)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "q3e_load_comparison.png", dpi=150)
    plt.close(fig)


def plot_emin_profiles(profiles: dict, reference: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    hours = np.arange(len(reference))
    ax.step(hours, reference, where="mid", color="k", ls=":", label="reference")
    for e, load in profiles.items():
        ax.step(hours, load, where="mid", label=f"Emin={e:.0f}")
    ax.set(xlabel="hour", ylabel="kWh/h", title="Q3(f): where the extra energy is consumed")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "q3f_emin_profiles.png", dpi=150)
    plt.close(fig)


def plot_cq_profiles(profiles: dict, reference: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    hours = np.arange(len(reference))
    ax.step(hours, reference, where="mid", color="k", ls=":", label="reference")
    for c, load in profiles.items():
        ax.step(hours, load, where="mid", label=f"cQ={c:g}")
    ax.set(xlabel="hour", ylabel="kWh/h", title="Q3(f): effect of the quadratic penalty, Emin fixed at 40 kWh")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "q3f_cq_profiles.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    q2 = load_question("Q2_linear")
    q2q = load_question("Q2_quadratic")
    q3 = load_question("Q3")

    print("=== Q2 linear base case ===")
    base_res = solve(q2)
    base_res.save(OUT / "q2b_base")
    base_m = breakdown(q2, base_res)
    print(base_m)
    hourly = base_res.hourly.copy()
    hourly["p_imp"] = q2.energy_price + q2.import_tariff
    hourly["p_exp"] = q2.energy_price - q2.export_tariff
    checks = []
    for t in range(q2.n_hours):
        region, expect, candidate, m_low, m_high = classify_linear_hour(q2, hourly["load"], t)
        got = float(hourly.loc[t, "load"])
        ok = True if expect is None else abs(got - expect) < 1e-4
        checks.append({
            "hour": t, "region": region, "ref": float(q2.reference_load[t]),
            "pv": float(q2.pv_available[t]), "p_imp": float(hourly.loc[t, "p_imp"]),
            "p_exp": float(hourly.loc[t, "p_exp"]),
            "m_low": m_low, "m_high": m_high,
            "load": got, "expect": expect if expect is not None else "tie",
            "import": float(hourly.loc[t, "import"]),
            "export": float(hourly.loc[t, "export"]),
            "pv_used": float(hourly.loc[t, "pv"]),
            "match": ok,
        })
    check_df = pd.DataFrame(checks)
    check_df.to_csv(OUT / "q2b_base_hourly_check.csv", index=False)
    print(check_df.to_string(index=False))
    print("all analytical labels match:", bool(check_df["match"].all()))

    print("\n=== Q2 linear sweep (compressed to rows where the schedule metrics change) ===")
    sweep = linear_sweep(q2)
    sweep.to_csv(OUT / "q2b_sweep_full.csv", index=False)
    # Keep a readable subset: the base case and the first row of each distinct energy level.
    key_cols = ["c_L", "procurement", "disutility", "objective", "energy", "abs_deviation",
                "n_off_reference", "n_deviation_bound_binding", "import", "export", "pv"]
    changed = sweep.loc[(sweep["energy"].round(4).diff().abs().fillna(1) > 1e-3)]
    show = pd.concat([changed, sweep.loc[sweep["c_L"].sub(q2.linear_disutility).abs() < 1e-9]]).drop_duplicates()
    show = show.sort_values("c_L")
    show[key_cols].to_csv(OUT / "q2b_sweep_changes.csv", index=False)
    print(show[key_cols].round(3).to_string(index=False))
    plot_linear_sweep(sweep)

    print("\n=== Q3(e) base case vs Q2(c) ===")
    r2, r3, h = q3_versus_q2(q2q, q3)
    h.to_csv(OUT / "q3e_hourly.csv", index_label="hour")
    m2, m3 = breakdown(q2q, r2), breakdown(q3, r3)
    print("Q2c", m2)
    print("Q3 ", m3)
    print("reference daily energy", float(q3.reference_load.sum()))
    print(h.round(3).to_string())
    plot_q3_comparison(h)

    # Finite-difference check that mu = d(objective)/d(Emin).
    eps = 0.05
    up = solve(replace(q3, min_daily_energy_kWh=q3.min_daily_energy_kWh + eps))
    print("mu from solver", m3["mu_min_energy"], "finite difference", (up.objective - r3.objective) / eps)

    print("\n=== Q3(f) sweep Emin ===")
    emin_values = [0, 10, 20, 25, 32.34, 40, 50, 60, 80, 100, 120]
    emin_df, emin_profiles = sweep_emin(q3, emin_values)
    emin_df.to_csv(OUT / "q3f_emin.csv", index=False)
    print(emin_df.round(3).to_string(index=False))
    # Profiles at the qualitatively different points.
    plot_emin_profiles({e: emin_profiles[e] for e in (20, 40, 60, 100)}, q3.reference_load)

    print("\n=== Q3(f) sweep cQ, Emin = 40 ===")
    cq_values = [0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0]
    cq_df, cq_profiles = sweep_cq(q3, cq_values)
    cq_df.to_csv(OUT / "q3f_cq.csv", index=False)
    print(cq_df.round(3).to_string(index=False))
    plot_cq_profiles({c: cq_profiles[c] for c in (0.05, 1.0, 20.0)}, q3.reference_load)

    # Also the unconstrained consumer's total, so the slack/binding threshold is explicit.
    print("\nUnconstrained Q2(c) energy", m2["energy"], "objective", m2["objective"])
    print("DONE", OUT)


if __name__ == "__main__":
    main()
