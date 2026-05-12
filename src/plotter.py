"""
plotter.py — Optional matplotlib energy/momentum graphs.

Plots KE, PE, and total energy vs time in a separate window.
Call plot_energy() to open a blocking figure, or plot_energy_live()
for a non-blocking update (requires plt.ion() to have been called).
"""

from __future__ import annotations

try:
    import matplotlib
    matplotlib.use("TkAgg")   # non-blocking backend; falls back gracefully
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from src.world import World


def plot_energy(world: World) -> None:
    """
    Open a static matplotlib figure showing energy history.
    Blocks until the window is closed.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[plotter] matplotlib not installed — skipping energy plot.")
        return

    t  = world.time_history
    ke = world.ke_history
    pe = world.pe_history
    te = world.te_history

    if len(t) < 2:
        print("[plotter] Not enough data to plot yet.")
        return

    fig = plt.figure(figsize=(10, 6), facecolor="#0c0e14")
    fig.suptitle(
        "Physics Sandbox — Energy vs Time",
        color="#c8d2e6", fontsize=14, fontweight="bold"
    )

    gs = gridspec.GridSpec(2, 1, hspace=0.4)

    # ── top panel: KE, PE, TE ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#0f1218")
    ax1.plot(t, ke, color="#64c8ff", linewidth=1.2, label="Kinetic Energy")
    ax1.plot(t, pe, color="#ff9640", linewidth=1.2, label="Potential Energy")
    ax1.plot(t, te, color="#a0ff78", linewidth=1.5, linestyle="--", label="Total Energy")
    ax1.set_xlabel("Time (s)", color="#9aa5be")
    ax1.set_ylabel("Energy (J)", color="#9aa5be")
    ax1.legend(facecolor="#1a1e2a", edgecolor="#3a4060", labelcolor="#c8d2e6")
    ax1.tick_params(colors="#9aa5be")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#3a4060")
    ax1.set_title("Energy Components", color="#9aa5be", fontsize=11)

    # ── bottom panel: energy conservation check ────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#0f1218")

    if te[0] > 1e-6:
        deviation = [(e - te[0]) / te[0] * 100 for e in te]
    else:
        deviation = [0.0] * len(te)

    ax2.plot(t, deviation, color="#ff5050", linewidth=1.0)
    ax2.axhline(0, color="#3a4060", linewidth=0.8)
    ax2.fill_between(t, deviation, alpha=0.15, color="#ff5050")
    ax2.set_xlabel("Time (s)", color="#9aa5be")
    ax2.set_ylabel("ΔE / E₀ (%)", color="#9aa5be")
    ax2.tick_params(colors="#9aa5be")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#3a4060")
    ax2.set_title(
        "Energy Conservation Error (numerical drift)",
        color="#9aa5be", fontsize=11
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def save_energy_plot(world: World, path: str = "graphs/energy_plot.png") -> bool:
    """
    Save energy plot to a PNG file without opening a GUI window.
    Returns True on success.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[plotter] matplotlib not installed.")
        return False

    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend
    import matplotlib.pyplot as plt2
    import matplotlib.gridspec as gs2

    t  = world.time_history
    ke = world.ke_history
    pe = world.pe_history
    te = world.te_history

    if len(t) < 2:
        return False

    fig = plt2.figure(figsize=(10, 6), facecolor="#0c0e14")
    fig.suptitle("Physics Sandbox — Energy vs Time", color="#c8d2e6", fontsize=14)
    spec = gs2.GridSpec(2, 1, hspace=0.4)

    ax1 = fig.add_subplot(spec[0])
    ax1.set_facecolor("#0f1218")
    ax1.plot(t, ke, "#64c8ff", label="KE")
    ax1.plot(t, pe, "#ff9640", label="PE")
    ax1.plot(t, te, "#a0ff78", linestyle="--", label="Total")
    ax1.legend(facecolor="#1a1e2a", labelcolor="#c8d2e6")
    ax1.set_ylabel("Energy (J)", color="#9aa5be")
    ax1.tick_params(colors="#9aa5be")

    ax2 = fig.add_subplot(spec[1])
    ax2.set_facecolor("#0f1218")
    if te[0] > 1e-6:
        dev = [(e - te[0]) / te[0] * 100 for e in te]
        ax2.plot(t, dev, "#ff5050")
    ax2.axhline(0, color="#3a4060")
    ax2.set_xlabel("Time (s)", color="#9aa5be")
    ax2.set_ylabel("ΔE/E₀ (%)", color="#9aa5be")
    ax2.tick_params(colors="#9aa5be")

    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt2.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt2.close(fig)
    print(f"[plotter] Saved energy plot → {path}")
    return True
