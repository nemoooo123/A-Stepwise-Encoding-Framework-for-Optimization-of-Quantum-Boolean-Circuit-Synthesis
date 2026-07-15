import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Gate encoding (matches utils/init_state.py assemble_reversible_circuit):
#   0 -> negative control (white/open circle)
#   1 -> positive control (black/filled circle)
#   3 -> target bit (white circle with a cross through it)


def draw_circuit(circuit, radius=0.12, save_path=None):
    """
    Draw a reversible/quantum boolean circuit diagram with matplotlib.

    circuit: list of gates, each gate is a list of length n (n = number of qubits/lines).
             Each entry is 0 (open control), 1 (filled control) or 3 (target).
    """
    n_lines = len(circuit[0])
    n_gates = len(circuit)

    fig, ax = plt.subplots(figsize=(max(4, n_gates * 1.0), max(2, n_lines * 0.8)))

    # Horizontal wires, one per qubit line. y = 0 is the top line, increasing downward.
    for row in range(n_lines):
        ax.plot([0, n_gates - 1], [row, row], color="black", linewidth=1, zorder=1)
        ax.text(-0.5, row, f"q{row}", ha="right", va="center", fontsize=11)

    # Each column is one multi-controlled gate: every line participates
    # (as a 0/1 control or the 3 target), so the connector spans the full column.
    for col, gate in enumerate(circuit):
        top = 0
        bottom = n_lines - 1
        ax.plot([col, col], [top, bottom], color="black", linewidth=1.5, zorder=2)

        for row, value in enumerate(gate):
            if value == 1:
                circle = Circle((col, row), radius, facecolor="black", edgecolor="black", zorder=3)
                ax.add_patch(circle)
            elif value == 0:
                circle = Circle((col, row), radius, facecolor="white", edgecolor="black", zorder=3)
                ax.add_patch(circle)
            elif value == 3:
                circle = Circle((col, row), radius, facecolor="white", edgecolor="black", zorder=3)
                ax.add_patch(circle)
                ax.plot([col - radius, col + radius], [row, row], color="black", linewidth=1.5, zorder=4)
                ax.plot([col, col], [row - radius, row + radius], color="black", linewidth=1.5, zorder=4)
            else:
                raise ValueError(f"Unknown gate value {value} at column {col}, row {row}")

    ax.set_xlim(-1, n_gates)
    ax.set_ylim(bottom=n_lines - 0.5, top=-0.5)  # row 0 at the top
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
    plt.show()


if __name__ == "__main__":
    # 3-qubit example circuit
    circuit = [
        [0, 0, 3],
        [0, 0, 3],
        [0, 1, 3],
        [0, 3, 0],
        [3, 1, 1],
    ]
    draw_circuit(circuit)
