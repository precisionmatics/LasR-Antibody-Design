"""
Regenerates all multi-panel manuscript figures with:
  - Panel labels A, B, C, D, E (bold, 14 pt)
  - Larger axis/title fonts (13-14 pt body, 15 pt titles)
  - White background

Produces revised PNG/PDF in Revision/figures/
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

# ── paths ─────────────────────────────────────────────────────────────────────
BASE   = "/home/stalin/Desktop/LasR_Antibody_Design"
MDDIR  = f"{BASE}/05_MD_Simulation/NbLasR-2"
BA     = f"{BASE}/06_Binding_Affinity"
OUTDIR = f"{BASE}/Revision/figures"
os.makedirs(OUTDIR, exist_ok=True)

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         13,
    "axes.titlesize":    14,
    "axes.labelsize":    13,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "legend.fontsize":   11,
    "axes.linewidth":    1.3,
    "figure.dpi":        150,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#fafafa",
})

def label_panel(ax, letter, x=-0.12, y=1.03, fontsize=15):
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top")

# ── load data ─────────────────────────────────────────────────────────────────
rmsd_all  = np.load(f"{MDDIR}/analysis/rmsd_complex.npy")
rmsd_lasr = np.load(f"{MDDIR}/analysis/rmsd_lasr.npy")
rmsd_nb   = np.load(f"{MDDIR}/analysis/rmsd_nb.npy")
time_ns   = np.load(f"{MDDIR}/analysis/time_ns.npy")
rg        = np.load(f"{MDDIR}/analysis/rg.npy")
rmsf_ca   = np.load(f"{MDDIR}/analysis/rmsf_ca.npy")
sasa      = np.load(f"{MDDIR}/analysis/sasa.npy")

dg        = np.load(f"{BA}/dg_per_frame.npy")
d_sasa    = np.load(f"{BA}/delta_sasa.npy")
cumul     = np.load(f"{BA}/cumul_mean_dg.npy")
with open(f"{BA}/mmgbsa_results.json") as f:
    mmr = json.load(f)

dg_mean = mmr["dg_bind_kcal_mol"]
dg_std  = mmr["dg_std_kcal_mol"]
ci_lo   = mmr["ci_95_lo"]
ci_hi   = mmr["ci_95_hi"]
blk_sem = mmr["block_sem_kcal_mol"]
hotspots = mmr["hotspots"]
n = len(dg)
time_mm = np.linspace(50, 100, n)

CLR_BLU = "#2196F3"
CLR_RED = "#F44336"
CLR_GRN = "#4CAF50"
CLR_PUR = "#9C27B0"
CLR_ORG = "#FF9800"

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Backbone RMSD vs time  (single panel, but needs larger fonts)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(time_ns, rmsd_all,  color=CLR_BLU, lw=1.2, label="Complex (backbone)", alpha=0.9)
ax.plot(time_ns, rmsd_lasr, color=CLR_GRN, lw=1.2, label="LasR LBD", alpha=0.9)
ax.plot(time_ns, rmsd_nb,   color=CLR_ORG, lw=1.2, label="NbLasR-2", alpha=0.9)
ax.set_xlabel("Time (ns)")
ax.set_ylabel("RMSD (Å)")
ax.set_title("Backbone RMSD — NbLasR-2 : LasR LBD Complex (100 ns MD)",
             fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/Fig9_RMSD_time.{fmt}", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: Fig9_RMSD_time")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 10 — RMSD distribution (3 panels, labels A B C)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
datasets = [(rmsd_all, "Complex (all backbone)", CLR_BLU),
            (rmsd_lasr, "LasR LBD", CLR_GRN),
            (rmsd_nb, "NbLasR-2", CLR_ORG)]
letters  = ["A", "B", "C"]

for ax, (data, lbl, col), letter in zip(axes, datasets, letters):
    ax.hist(data, bins=60, color=col, edgecolor="black", lw=0.4, alpha=0.82)
    ax.axvline(data.mean(), color="red", lw=2.5, ls="--",
               label=f"Mean = {data.mean():.2f} Å\nSD = {data.std():.2f} Å")
    ax.set_xlabel("RMSD (Å)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"RMSD Distribution — {lbl}", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    label_panel(ax, letter)

plt.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/Fig10_RMSD_distribution.{fmt}", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: Fig10_RMSD_distribution")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 11 — MM-GBSA (5 panels, labels A B C D E)
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

# (A) ΔG time series
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(time_mm, dg, color=CLR_BLU, lw=0.7, alpha=0.55, label="Per-frame MM-GBSA")
ax1.axhline(dg_mean, color=CLR_RED, lw=2.5,
            label=f"Mean = {dg_mean:.1f} kcal/mol")
ax1.fill_between(time_mm, ci_lo, ci_hi, alpha=0.15, color=CLR_RED,
                 label=f"95% CI [{ci_lo:.1f}, {ci_hi:.1f}]")
ax1.set_xlabel("Simulation Time (ns)")
ax1.set_ylabel("ΔG$_{bind}$ (kcal/mol)")
ax1.set_title("MM-GBSA Binding Free Energy — NbLasR-2 : LasR LBD", fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
info = (f"ΔG$_{{bind}}$ = {dg_mean:.1f} ± {dg_std:.1f} kcal/mol\n"
        f"95% CI: [{ci_lo:.1f}, {ci_hi:.1f}] kcal/mol\n"
        f"Method: ST-MM-GBSA | OBC2 | AMBER ff14SB\n"
        f"Frames: {n} (50–100 ns, Δt = 100 ps)")
ax1.text(0.02, 0.04, info, transform=ax1.transAxes, fontsize=9,
         va="bottom", bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.85))
label_panel(ax1, "A")

# (B) KDE distribution
ax2 = fig.add_subplot(gs[0, 2])
kde = gaussian_kde(dg, bw_method=0.3)
xr  = np.linspace(dg.min() - 10, dg.max() + 10, 400)
ax2.fill_between(xr, kde(xr), alpha=0.35, color=CLR_BLU)
ax2.plot(xr, kde(xr), color="#1565C0", lw=2.5)
ax2.axvline(dg_mean, color=CLR_RED, lw=2, ls="--", label=f"μ = {dg_mean:.1f}")
ax2.axvline(ci_lo, color=CLR_ORG, lw=1.5, ls=":", label="95% CI")
ax2.axvline(ci_hi, color=CLR_ORG, lw=1.5, ls=":")
ax2.set_xlabel("ΔG$_{bind}$ (kcal/mol)")
ax2.set_ylabel("Density")
ax2.set_title("Distribution of ΔG$_{bind}$", fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3)
label_panel(ax2, "B")

# (C) Convergence
ax3 = fig.add_subplot(gs[1, :2])
ax3.plot(time_mm, cumul, color=CLR_GRN, lw=2.5, label="Cumulative mean")
ax3.axhline(dg_mean, color=CLR_RED, lw=1.8, ls="--", alpha=0.7,
            label=f"Final mean {dg_mean:.1f} kcal/mol")
ax3.fill_between(time_mm, dg_mean - blk_sem, dg_mean + blk_sem,
                 alpha=0.20, color=CLR_RED, label=f"±SEM$_{{block}}$ = {blk_sem:.1f}")
ax3.set_xlabel("Simulation Time (ns)")
ax3.set_ylabel("Cumulative ΔG$_{bind}$ (kcal/mol)")
ax3.set_title("Convergence of MM-GBSA Mean", fontweight="bold")
ax3.legend()
ax3.grid(True, alpha=0.3)
label_panel(ax3, "C")

# (D) ΔSASA
ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(time_mm, d_sasa, color=CLR_PUR, lw=0.8, alpha=0.55)
ax4.axhline(d_sasa.mean(), color="#4A148C", lw=2.5, ls="--",
            label=f"Mean {d_sasa.mean():.0f} Å²")
ax4.set_xlabel("Time (ns)")
ax4.set_ylabel("ΔSASA (Å²)")
ax4.set_title("Interface SASA Burial (ΔSASA)", fontweight="bold")
ax4.legend()
ax4.grid(True, alpha=0.3)
label_panel(ax4, "D")

# (E) Hot-spot bar chart
ax5 = fig.add_subplot(gs[2, :])
top_hs  = hotspots[:20]
colors5 = [CLR_BLU if r["chain"] == "NbLasR-2" else CLR_GRN for r in top_hs]
labels5 = [f"{r['resname']}{r['resid']}\n({r['chain'][:2]})" for r in top_hs]
vals5   = [r["burial_A2"] for r in top_hs]
ax5.bar(range(len(top_hs)), vals5, color=colors5, edgecolor="white", lw=0.6)
ax5.set_xticks(range(len(top_hs)))
ax5.set_xticklabels(labels5, fontsize=10)
ax5.set_ylabel("SASA Burial (Å²)")
ax5.set_title("Top Interface Hot-Spot Residues by SASA Burial", fontweight="bold")
ax5.grid(True, alpha=0.3, axis="y")
ax5.legend(handles=[Patch(facecolor=CLR_BLU, label="NbLasR-2"),
                    Patch(facecolor=CLR_GRN, label="LasR LBD")],
           loc="upper right")
label_panel(ax5, "E")

plt.suptitle("Figure 11 — MM-GBSA Binding Affinity Analysis: NbLasR-2 : LasR LBD",
             fontsize=15, fontweight="bold", y=1.01)
for fmt in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/Fig11_MMGBSA.{fmt}", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: Fig11_MMGBSA")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 9 (combined 3-panel) — RMSD + Rg + SASA side by side (alternative)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

ax = axes[0]
ax.plot(time_ns, rmsd_all,  color=CLR_BLU, lw=1.2, label="Complex", alpha=0.9)
ax.plot(time_ns, rmsd_lasr, color=CLR_GRN, lw=1.2, label="LasR LBD", alpha=0.9)
ax.plot(time_ns, rmsd_nb,   color=CLR_ORG, lw=1.2, label="NbLasR-2", alpha=0.9)
ax.set_xlabel("Time (ns)")
ax.set_ylabel("RMSD (Å)")
ax.set_title("Backbone RMSD", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
label_panel(ax, "A")

ax = axes[1]
ax.plot(time_ns, rg, color=CLR_RED, lw=1.2, alpha=0.85)
ax.axhline(rg.mean(), color="black", lw=1.8, ls="--",
           label=f"Mean = {rg.mean():.2f} Å")
ax.fill_between(time_ns, rg.mean()-rg.std(), rg.mean()+rg.std(),
                alpha=0.15, color=CLR_RED, label=f"±1σ = {rg.std():.2f} Å")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Radius of Gyration (Å)")
ax.set_title("Radius of Gyration", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
label_panel(ax, "B")

ax = axes[2]
ax.plot(range(len(rmsf_ca)), rmsf_ca, color=CLR_PUR, lw=1.2, alpha=0.85)
ax.fill_between(range(len(rmsf_ca)), rmsf_ca, alpha=0.15, color=CLR_PUR)
ax.set_xlabel("Residue Index")
ax.set_ylabel("Cα RMSF (Å)")
ax.set_title("Per-Residue Flexibility (RMSF)", fontweight="bold")
ax.grid(alpha=0.3)
label_panel(ax, "C")

plt.suptitle("Figure 9 — 100 ns MD Trajectory Analysis: NbLasR-2 : LasR LBD",
             fontsize=15, fontweight="bold")
plt.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/Fig9_MD_analysis.{fmt}", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: Fig9_MD_analysis (RMSD+Rg+RMSF combined)")

print(f"\n  All revised figures saved to: {OUTDIR}/")
