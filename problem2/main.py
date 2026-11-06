import argparse

from .evolution_strategies import ALGS
from .portfolio_experiments import cases, run_one, run_All
from .plot_results import plot_convergence


def main():
    """Start the selected experiment"""

    p = argparse.ArgumentParser(
        description="Optimize a portfolio with Evolution Strategies"
    )

    p.add_argument(
        "--all",
        action="store_true",
        help="Run all 450 experiments"
    )

    p.add_argument(
        "--plot",
        action="store_true",
        help="Plot saved convergence results"
    )

    p.add_argument(
        "--case",
        choices=cases(),
        default="port1_alpha025"
    )

    p.add_argument(
        "--alg",
        choices=ALGS,
        default="one"
    )

    p.add_argument(
        "--seed",
        type=int,
        default=0
    )

    p.add_argument(
        "--budget",
        type=int,
        default=30000
    )

    args = p.parse_args()

    if args.all:

        # Run all datasets, algorithms and seeds
        run_All(args.budget)

        print(
            f"Convergence plot saved: {plot_convergence()}"
        )

    elif args.plot:

        # Generate a plot from previous experiment results
        print(
            f"Convergence plot saved: {plot_convergence()}"
        )

    else:

        # Run a single experiment
        row, _ = run_one(
            args.case,
            args.alg,
            args.seed,
            args.budget
        )

        results = [
            ("Case", row["case"]),
            ("Algorithm", row["alg"]),
            ("Seed", row["seed"]),
            ("Target return", f"{row['target']:.10f}"),
            ("Best variance", f"{row['variance']:.10f}"),
            ("Expected return", f"{row['return']:.10f}"),
            ("Feasible", row["valid"]),
            ("Evaluations", row["evals"]),
            ("Runtime", f"{row['runtime_s']:.3f} seconds"),
            ("Best portfolio weights", row["weights"])
        ]

        for label, value in results:
            print(f"{label}: {value}")


if __name__ == "__main__":
    main()