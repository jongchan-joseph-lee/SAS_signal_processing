"""English, publication-style figures for README.md (docs/figures).

Every figure is drawn from files under results/ and references/, except the
first four bars of the interpolation/window comparison: those were computed
from results/v2/native_ablation.npz, which is not distributed, and are copied
from report_v2 (development points 1/2/4, inner 25 x 25 ROI). Font: Times New
Roman where installed, otherwise the metric-compatible Liberation Serif.

Usage: python src/readme_figures.py
"""
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"

# Publication palette: black and grey first, one red and one blue for contrast.
K, G, LG, R, B = "#000000", "#7a7a7a", "#bdbdbd", "#b2182b", "#2166ac"
METHODS = ["uncorrected", "local_gpga", "per_ping_entropy"]
METHOD_LABELS = ["Uncorrected", "Local GPGA", "Per-ping entropy"]
METHOD_COLORS = [G, B, K]


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
    })


def read(rel):
    return json.loads((ROOT / rel).read_text())


def db(z, peak=None):
    a = np.abs(z)
    peak = a.max() if peak is None else peak
    return 20 * np.log10(np.maximum(a / (peak + 1e-30), 1e-4))


def panel(ax, letter):
    ax.text(-0.13, 1.02, f"({letter})", transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)


def scene_comparison():
    ref = np.load(ROOT / "results/v3/detection_example.npz")
    bp = np.load(ROOT / "results/v3/raw_bridge_examples.npz")
    x, y = ref["x"], ref["y"]
    panels = [(ref["power"], "Provided ASASIN SLC"), (bp["clean_bp_power"], "Own backprojection"),
              (bp["corrupted_bp_power"], "Own backprojection,\n2 mm RMS sway error")]
    fig, axs = plt.subplots(1, 3, figsize=(10, 5.4), layout="constrained", sharex=True, sharey=True)
    for i, (ax, (power, title)) in enumerate(zip(axs, panels)):
        norm = np.median(power, axis=1, keepdims=True) + 1e-25
        im = ax.imshow(10 * np.log10(np.maximum(power / norm, 1e-5)), origin="lower", cmap="gray", vmin=-12, vmax=20,
                       extent=[x[0], x[-1], y[0], y[-1]], aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("Along-track (m)")
        panel(ax, "abc"[i])
    axs[0].set_ylabel("Cross-track (m)")
    fig.colorbar(im, ax=axs, shrink=0.7, label="Power re. range-wise median (dB, display only)")
    save(fig, "scene_comparison.png")


def motion_compensation():
    p = read("results/possm/possm_results.json")
    rows = p["heldout_points"]["rows"]
    stop = np.array([r["stop_and_hop"]["localization_m"] * 1000 for r in rows])
    moving = np.array([r["continuous_receiver"]["localization_m"] * 1000 for r in rows])
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
    n = np.arange(1, len(rows) + 1)
    axs[0].plot(n, stop, "o-", color=G, mfc="white", label="Stop-and-hop")
    axs[0].plot(n, moving, "s-", color=K, label="Receiver motion included")
    axs[0].set(xlabel="Held-out scatterer index", ylabel="Localisation error (mm)", xlim=(0, 25), ylim=(0, None))
    axs[0].legend(loc="upper right")
    d = np.load(ROOT / "results/possm/point_examples.npz")
    z, s, x = d["moving"][0], d["stop"][0], d["x"][0]
    iy = np.abs(z).max(1).argmax()
    truth = rows[0]["truth"][0]
    axs[1].plot((x - truth) * 1000, db(s[iy]), color=G, label="Stop-and-hop")
    axs[1].plot((x - truth) * 1000, db(z[iy]), color=K, label="Receiver motion included")
    axs[1].axvline(0, color=R, ls=":", lw=1, label="True position")
    axs[1].set(xlim=(-80, 80), ylim=(-45, 2), xlabel="Along-track offset from truth (mm)", ylabel="Normalised amplitude (dB)")
    axs[1].legend(loc="lower center", fontsize=8)
    panel(axs[0], "a"); panel(axs[1], "b")
    save(fig, "motion_compensation.png")


def window_ablation_dev():
    rows = read("results/v2/taper_ablation.json")["rows"]
    hann = [np.mean([r["complex_correlation"] for r in rows if r["taper"] == "hann60" and r["beam"] == b]) for b in (0, 1)]
    labels = ["4×", "8×", "16×", "Gaussian", "Hann", "Hann +\nGaussian"]
    scores = [0.505, 0.495, 0.493, 0.525, *hann]  # first four: report_v2, see module docstring
    fig, ax = plt.subplots(figsize=(6.4, 3.0), layout="constrained")
    bars = ax.bar(labels, scores, color=[LG, LG, LG, G, G, K], edgecolor=K, linewidth=0.6, width=0.62)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set(ylim=(0, 1), ylabel="Mean complex NCC (3 development points)")
    ax.text(1, -0.24, "Interpolation only", transform=ax.get_xaxis_transform(), ha="center", fontsize=9, color=G)
    ax.text(4, -0.24, "Range-spectrum / along-track weighting", transform=ax.get_xaxis_transform(), ha="center", fontsize=9, color=G)
    save(fig, "window_ablation_dev.png")


def window_test_points():
    v = read("results/v2/validation_results.json")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained", sharey=True)
    n = np.arange(1, len(v["point_rows"]) + 1)
    for ax, metric, ylab in zip(axs, ["complex_correlation", "amplitude_correlation"], ["Complex NCC", "Amplitude correlation (Pearson)"]):
        ax.plot(n, [r["native_legacy"][metric] for r in v["point_rows"]], "o-", color=G, mfc="white", label="First pass")
        ax.plot(n, [r["native_v2"][metric] for r in v["point_rows"]], "s-", color=K, label="Hann + Gaussian")
        ax.set(xlabel="Held-out point index", ylabel=ylab, ylim=(0, 1.03), xlim=(0, 25))
        ax.legend(loc="lower left")
    axs[1].axhline(0.9, color=R, ls=":", lw=1)
    axs[1].yaxis.set_tick_params(labelleft=True)
    panel(axs[0], "a"); panel(axs[1], "b")
    save(fig, "window_test_points.png")


def psf_window():
    v = read("results/v2/validation_results.json")
    e = np.load(ROOT / "results/v2/point_validation_examples.npz")
    pts = np.loadtxt(ROOT / "references/Scatterer_Locations_Scene00.csv", delimiter=",", skiprows=1)
    truth = pts[v["frozen_configuration"]["test_point_indices"][0]]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained", sharey=True)
    for name, label, color in [("moving", "First pass", G), ("v2", "Hann + Gaussian", K)]:
        z = e[name + "_dense"]
        iy, ix = np.unravel_index(np.abs(z).argmax(), z.shape)
        axs[0].plot((e["dense_x"] - truth[0]) * 1000, db(z[iy]), color=color, label=label)
        axs[1].plot((e["dense_y"] - truth[1]) * 1000, db(z[:, ix]), color=color, label=label)
    for ax, t in zip(axs, ["Along-track cut", "Range cut"]):
        ax.set(xlim=(-80, 80), ylim=(-55, 2), xlabel="Offset from true position (mm)", title=t)
        ax.axhline(-3, color=R, ls=":", lw=0.9)
    axs[0].set_ylabel("Normalised amplitude (dB)")
    axs[0].legend(loc="lower left")
    axs[1].yaxis.set_tick_params(labelleft=True)
    panel(axs[0], "a"); panel(axs[1], "b")
    save(fig, "psf_window.png")


def navigation_sensitivity():
    s = read("results/v2/stress_results.json")["summary"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
    for kind, label, color, marker in [("surge", "Surge (along-track)", G, "o"), ("sway", "Sway (cross-track)", K, "s"), ("joint", "Surge + sway", R, "^")]:
        rows = [r for r in s if r["kind"] == kind]
        x = [r["amount"] * 1000 for r in rows]
        axs[0].plot(x, [r["complex_correlation"]["mean"] for r in rows], marker + "-", color=color, mfc="white" if kind == "surge" else color, label=label)
        axs[0].fill_between(x, [r["complex_correlation"]["q25"] for r in rows], [r["complex_correlation"]["q75"] for r in rows], color=color, alpha=0.06, lw=0)
    rows = [r for r in s if r["kind"] == "sound_speed_bias"]
    x = [r["amount"] for r in rows]
    axs[1].plot(x, [r["complex_correlation"]["mean"] for r in rows], "s-", color=K)
    axs[1].fill_between(x, [r["complex_correlation"]["q25"] for r in rows], [r["complex_correlation"]["q75"] for r in rows], color=K, alpha=0.1, lw=0)
    axs[0].set(xlabel="Track error RMS (mm)", ylabel="Complex NCC vs error-free BP", ylim=(0, 1.04))
    axs[1].set(xlabel="Constant sound-speed bias (m/s)", ylabel="Complex NCC vs error-free BP", ylim=(0, 1.04))
    axs[0].legend(loc="lower left")
    panel(axs[0], "a"); panel(axs[1], "b")
    save(fig, "navigation_sensitivity.png")


def autofocus_by_range():
    af = read("results/v3/autofocus_results.json")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for i, (m, label, color) in enumerate(zip(METHODS, METHOD_LABELS, METHOD_COLORS)):
        for b in range(3):
            vals = [r["complex_correlation"] for r in af["rows"] if r["amount"] and r["method"] == m and r["range_bin"] == b]
            bp = axs[0].boxplot(vals, positions=[b * 4 + i], widths=0.7, patch_artist=True,
                                boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 1.1},
                                medianprops={"color": color, "linewidth": 1.6}, whiskerprops={"color": color},
                                capprops={"color": color}, flierprops={"markersize": 2, "markeredgecolor": color})
        axs[0].plot([], [], color=color, lw=4, label=label)
        loss = [1 - r["complex_correlation"] for r in af["rows"] if not r["amount"] and r["method"] == m]
        axs[1].scatter(np.arange(1, len(loss) + 1), loss, color=color, s=14, marker="os^"[i], label=label,
                       facecolors="white" if i == 0 else color)
    axs[0].axhline(0.9, color=R, ls="--", lw=1, label="Success criterion (0.90)")
    axs[0].set(xticks=[1, 5, 9], xticklabels=["15–35 m", "35–55 m", "55–80 m"], ylim=(-0.03, 1.05),
               xlabel="Range band", ylabel="Complex NCC vs error-free BP")
    axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=8)
    axs[1].axhline(0.01, color=R, ls="--", lw=1, label="Maximum allowed loss (0.01)")
    axs[1].set(xlabel="Held-out point index", ylabel="NCC loss on error-free input", xlim=(0, 25))
    axs[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=8)
    panel(axs[0], "a"); panel(axs[1], "b")
    save(fig, "autofocus_by_range.png")


def detection_froc():
    det = read("results/v3/detection_results.json")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    curves = [("froc_proposal", "Contrast proposals only", LG), ("froc_rf", "Proposals + RF (location)", K), ("froc_class_aware", "Proposals + RF (location and class)", B)]
    for key, label, color in curves:
        xx = [r["fp_per_scene"] for r in det[key]]
        yy = [r["recall"] for r in det[key]]
        for ax in axs:
            ax.step(xx, yy, where="post", color=color, label=label)
    op = det["operating_point_rf"]
    lo, hi = op["wilson95"]
    axs[0].errorbar(op["fp_per_scene"], op["recall"], yerr=[[op["recall"] - lo], [hi - op["recall"]]], fmt="o", color=R, capsize=3, label="Frozen operating point (Wilson 95%)")
    axs[0].set(xlim=(0, 30), ylim=(-0.01, 1.03), xlabel="False positives per scene", ylabel="Recall (of 36 objects)")
    axs[1].set(xscale="log", xlim=(0.1, 650), ylim=(-0.01, 1.03), xlabel="False positives per scene (log)", ylabel="Recall (of 36 objects)")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=8.5)
    panel(axs[0], "a"); panel(axs[1], "b")
    save(fig, "detection_froc.png")


def detection_map():
    det = read("results/v3/detection_results.json")
    sel = read("results/v3/detection_selection.json")
    ex = np.load(ROOT / "results/v3/detection_example.npz")
    power = ex["power"]
    normal = np.median(power, axis=1, keepdims=True) + 1e-25
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 5.0), layout="constrained", gridspec_kw={"width_ratios": [1, 1.05]})
    ax = axs[0]
    ax.imshow(10 * np.log10(np.maximum(power / normal, 1e-5)), origin="lower", cmap="gray", vmin=-12, vmax=20,
              extent=[ex["x"][0], ex["x"][-1], ex["y"][0], ex["y"][-1]], aspect="equal")
    g = ex["truth"]
    ax.scatter(g[:, 0], g[:, 1], s=80, facecolors="none", edgecolors="white", linewidths=1.2, label="Inserted object (truth)")
    keep = 1 - ex["probability"][:, 0] >= sel["rf_threshold"]
    p = ex["positions"][keep]
    ax.scatter(p[:, 0], p[:, 1], s=40, marker="+", c=R, linewidths=1.4, label="Detection (frozen threshold)")
    ax.set(xlabel="Along-track (m)", ylabel="Cross-track (m)", title="Test scene 04")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="none", marker="o", ms=8, mfc="none", mec=K, label="Inserted object (truth)"),
               Line2D([], [], ls="none", marker="+", ms=8, mew=1.4, color=R, label="Detection (frozen threshold)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=8.5)
    cm = np.array(det["operating_point_rf"]["confusion_rows_truth_columns_background_trap_rock_cylinder"])
    ax = axs[1]
    ax.imshow(cm, cmap="Greys", vmin=0, vmax=max(14, cm.max()))
    for iy in range(cm.shape[0]):
        for ix in range(cm.shape[1]):
            ax.text(ix, iy, str(cm[iy, ix]), ha="center", va="center", color="white" if cm[iy, ix] > 7 else K)
    ax.set(xticks=range(4), xticklabels=["Missed", "Trap", "Rock", "Cylinder"], yticks=range(3),
           yticklabels=["Trap", "Rock", "Cylinder"], xlabel="Prediction", ylabel="Truth",
           title="Class result after 1 m location match\n(4 test scenes, 36 objects)")
    ax.tick_params(top=False, right=False)
    panel(axs[0], "a"); panel(axs[1], "b")
    save(fig, "detection_map.png")


def main():
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (scene_comparison, motion_compensation, window_ablation_dev, window_test_points, psf_window,
              navigation_sensitivity, autofocus_by_range, detection_froc, detection_map):
        f()
        print("wrote", f.__name__)


if __name__ == "__main__":
    main()
