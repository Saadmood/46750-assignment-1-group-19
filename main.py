"""Entry point: load one question's data, build and solve the model, save results and figures.

    python main.py                          # base case of Q1_caseA
    python main.py --question Q2_linear     # another case
    python main.py --scenarios              # also run the example sensitivity scenarios

Results (CSV, TXT, PNG) are written to ``results/<question>/``. Extend ``run_scenarios``
with your own scenarios, or add a new function per question, as your analysis grows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

from src.data_loader import load_question, list_questions
from src.model import FlexibleConsumerModel, Results
from src.plotting import plot_duals, plot_inputs, plot_scenario_comparison, plot_schedule
from src.scenarios import scale_prices, scale_pv, set_tariffs

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_base_case(question: str, out: Path, show: bool) -> Results | None:
    data = load_question(question)
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    model = FlexibleConsumerModel(data).build()
    try:
        results = model.solve()
    except NotImplementedError as e:
        print(f"[skipped] {e}")
        return None

    u_L = data.consumption_utility or 0.0
    total_utility = (u_L * results.hourly["load"]).sum()
    procurement_cost = total_utility - results.objective

    print(results, "\n")
    print(f"Total Consumption Utility : {total_utility:.2f} DKK")
    print(f"Total Procurement Cost    : {procurement_cost:.2f} DKK")
    print(f"Net Utility (Objective)   : {results.objective:.2f} DKK")

    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")
    plot_duals(results, data, save_to=out / "duals.png")
    if show:
        matplotlib.pyplot.show()
    return results


def run_scenarios(question: str, out: Path) -> dict[str, Results]:
    """Example sensitivity analysis. Replace with the scenarios you design in Question 1.g."""
    base = load_question(question)
    scenarios = {
        "base": base,
        "flat_prices": scale_prices(base, factor=0.0, keep_mean=True),
        "double_spread": scale_prices(base, factor=2.0, keep_mean=True),
        "no_tariffs": set_tariffs(base, import_tariff=0.0, export_tariff=0.0),
        "no_pv": scale_pv(base, factor=0.0),
    }
    runs: dict[str, Results] = {}
    for name, data in scenarios.items():
        results = FlexibleConsumerModel(data).build().solve()
        results.save(out, tag=name)
        runs[name] = results
        print(f"{name:>14}: cost {results.objective:8.2f} DKK | import {results.hourly['import'].sum():5.1f} kWh"
              f" | export {results.hourly['export'].sum():5.1f} kWh")
    plot_scenario_comparison(runs, "objective", save_to=out / "scenarios_cost.png")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="Q1_caseA", choices=list_questions(), help="data case to use")
    parser.add_argument("--scenarios", action="store_true", help="also run the example sensitivity scenarios")
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    args = parser.parse_args()

    out = RESULTS_DIR / args.question
    out.mkdir(parents=True, exist_ok=True)
    if not args.show:
        matplotlib.use("Agg")

    base = run_base_case(args.question, out, args.show)
    if args.scenarios and base is not None:
        run_scenarios(args.question, out)
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt

price_factors = [0.5, 1.0, 1.5, 2.0]
values_price = [1.20, 2.63, 5.15, 8.05]

capacities = [2.0, 4.0, 6.0, 8.0]
values_cap = [1.30, 2.63, 3.16, 3.24]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

ax1.plot(price_factors, values_price, 'o-', color='C0', lw=2)
ax1.set(xlabel="Price volatility multiplier", ylabel="Battery value [DKK/day]", title="Impact of market price spread")
ax1.grid(True, ls='--', alpha=0.5)

ax2.plot(capacities, values_cap, 's-', color='C3', lw=2)
ax2.set(xlabel="Battery capacity [kWh]", ylabel="Battery value [DKK/day]", title="Impact of storage capacity")
ax2.grid(True, ls='--', alpha=0.5)

plt.tight_layout()
plt.savefig("results/Q3_battery/battery_sensitivity.png", dpi=150)