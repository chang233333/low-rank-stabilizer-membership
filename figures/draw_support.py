"""Draw the two coordinate axes supporting the Pauli obstruction for k=2."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "cm",
                         "font.size": 10, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.1, 2.46), layout="constrained")
    labels = ["00", "01", "10", "11"]
    for x in range(4):
        for y in range(4):
            on_axis = x == 0 or y == 0
            ax.scatter(x, y, s=68 if on_axis else 35, marker="s" if on_axis else "o",
                       facecolor="#2f6373" if on_axis else "white",
                       edgecolor="#234952" if on_axis else "#a9b2b4", linewidth=.75, zorder=2)
    ax.plot([2, 0], [0, 3], color="#a25d32", linestyle="--", linewidth=1.2, zorder=1)
    ax.scatter([2, 0], [0, 3], s=116, facecolors="none", edgecolors="#8e4c26",
               linewidth=1.1, marker="s", zorder=3)
    ax.set(xlim=(-.4, 3.4), ylim=(-.4, 3.4), aspect="equal", xlabel="First label",
           ylabel="Second label")
    ax.set_xticks(range(4), labels)
    ax.set_yticks(range(4), labels)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#899698")
    ax.tick_params(length=2.5, width=.6)
    fig.savefig(ROOT / "paper" / "support-obstruction.pdf", bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
