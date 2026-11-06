import numpy as np
import matplotlib.pyplot as plt

from .portfolio_experiments import OUT, SEEDS
from .evolution_strategies import ALGS


def plot_convergence():
    """Plot convergence for port5_alpha050"""

    case = "port5_alpha050"

    for alg in ALGS:

        hs = []

        # Read convergence history from each seed
        for seed in SEEDS:

            path = OUT / "logs" / (
                f"{case}_{alg}_seed{seed}.npz"
            )

            with np.load(path) as data:
                hs.append(data["hist"])

        # Calculate mean convergence across 10 runs
        avg = np.mean(
            np.stack(hs),
            axis=0
        )

        # Plot the algorithm's average history
        plt.plot(
            np.arange(1, len(avg) + 1),
            avg,
            label=alg
        )

    plt.xlabel("Objective evaluations")

    plt.ylabel(
        "Mean archive best portfolio variance"
    )

    plt.title(
        "Convergence: port5_alpha050 (10 seeds)"
    )

    plt.legend(title="ES variant")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    # Save figure.
    path = OUT / "port5_alpha050_convergence.png"

    plt.savefig(
        path,
        dpi=180
    )

    plt.close()

    return path