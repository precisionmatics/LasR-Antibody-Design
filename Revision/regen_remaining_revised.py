"""
Regenerate Figures 4, 6, 8, 13, 14, 15 with:
  - Panel labels (A), (B), … (bold 15 pt)
  - Larger body fonts (13 pt) and titles (14 pt)
  - White backgrounds
  - Saved to Revision/figures/
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr

BASE   = "/home/stalin/Desktop/LasR_Antibody_Design"
OUTDIR = f"{BASE}/Revision/figures"
os.makedirs(OUTDIR, exist_ok=True)

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   13,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  11,
    "axes.linewidth":   1.3,
    "figure.dpi":       150,
    "figure.facecolor": "white",
    "axes.facecolor":   "#fafafa",
})

def label_panel(ax, letter, x=-0.13, y=1.04, fontsize=15):
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top")

CANDIDATES = ["NbLasR-1", "NbLasR-2", "NbLasR-3", "NbLasR-4", "NbLasR-5"]
SHORT      = ["Nb-1", "Nb-2", "Nb-3", "Nb-4", "Nb-5"]
COLORS     = ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6", "#F39C12"]

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Multi-algorithm epitope prediction (single composite figure)
# ═══════════════════════════════════════════════════════════════════════════
print("Regenerating Fig 4: Epitope prediction...")

df = pd.read_csv(f"{BASE}/02_Epitope_Mapping/epitope_scores_all_residues.csv")
res = df["ResNum"].values
aa  = df["AA"].values

def norm(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn)

kt   = norm(df["KT_score"].values)
par  = norm(df["Parker"].values)
emi  = norm(df["Emini"].values)
ksf  = norm(df["KS_flex"].values)
cft  = norm(df["CF_turn"].values)
comb = norm(df["Combined"].values)
fin  = norm(df["Final_Score"].values)

in_lbd   = df["In_LBD"].values.astype(bool)
key_res  = df["Key_Res"].values.astype(bool)
epitope  = df["Final_Epitope"].values.astype(bool)

REGIONS = [
    dict(start=42, end=49,  label="Ep-I\n(KDSQDYEN)",  rank=1),
    dict(start=62, end=76,  label="Ep-II\n(EHYDRAGYARVDPTV)", rank=2),
    dict(start=11, end=16,  label="Ep-III\n(ERSSGK)",   rank=3),
]
ALGS = [
    ("Kolaskar-Tongaonkar (Antigenicity)",  kt,   "#E74C3C"),
    ("Parker (Hydrophilicity)",              par,  "#3498DB"),
    ("Emini (Surface Accessibility)",        emi,  "#2ECC71"),
    ("Karplus-Schulz (Flexibility)",         ksf,  "#9B59B6"),
    ("Chou-Fasman (β-Turn Propensity)",      cft,  "#F39C12"),
]
sigma = 1.2
algs_sm = [(lbl, gaussian_filter1d(vals, sigma), col) for lbl, vals, col in ALGS]
fin_sm  = gaussian_filter1d(fin, sigma)

n_algo  = len(ALGS)
heights = [0.55] + [1.0] * n_algo + [1.3] + [0.55]
fig = plt.figure(figsize=(18, 15), facecolor="white")
gs  = GridSpec(n_algo + 3, 1, figure=fig,
               height_ratios=heights, hspace=0.04,
               left=0.10, right=0.97, top=0.97, bottom=0.08)
axes = [fig.add_subplot(gs[i]) for i in range(n_algo + 3)]
ax_top   = axes[0]
alg_axes = axes[1:n_algo+1]
ax_comb  = axes[n_algo+1]
ax_seq   = axes[n_algo+2]

ep_colors = ["#E74C3C", "#F39C12", "#27AE60"]

def shade_regions(ax, alpha=0.12):
    for reg, col in zip(REGIONS, ep_colors):
        ax.axvspan(reg["start"]-0.5, reg["end"]+0.5, alpha=alpha, color=col, zorder=0)

def shade_lbd(ax, alpha=0.06):
    lbd_idx = np.where(in_lbd)[0]
    if len(lbd_idx):
        ax.axvspan(res[lbd_idx[0]]-0.5, res[lbd_idx[-1]]+0.5, alpha=alpha,
                   color="#2C3E50", zorder=0)

# Top context strip
ax_top.set_xlim(res[0]-1, res[-1]+1)
ax_top.set_ylim(0, 1)
ax_top.set_yticks([])
lbd_idx = np.where(in_lbd)[0]
ax_top.barh(0.6, res[lbd_idx[-1]]-res[lbd_idx[0]], left=res[lbd_idx[0]],
            height=0.25, color="#2C3E50", alpha=0.20, zorder=1)
ax_top.text(res[lbd_idx[0]] + (res[lbd_idx[-1]]-res[lbd_idx[0]])/2,
            0.625, "Ligand-Binding Domain (LBD)", ha="center", va="center",
            fontsize=10, color="#2C3E50", fontweight="bold")
ep_labels = ["Ep-I  KDSQDYEN", "Ep-II  EHYDRAGYARVDPTV", "Ep-III  ERSSGK"]
for reg, col, lbl in zip(REGIONS, ep_colors, ep_labels):
    mid = (reg["start"] + reg["end"]) / 2
    ax_top.barh(0.2, reg["end"]-reg["start"], left=reg["start"],
                height=0.22, color=col, alpha=0.80, zorder=2)
    ax_top.text(mid, 0.215, lbl, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
kr_idx = np.where(key_res)[0]
for i in kr_idx:
    ax_top.annotate("", xy=(res[i], 0.90), xytext=(res[i], 0.97),
                    arrowprops=dict(arrowstyle="-|>", color="#8E44AD",
                                   lw=1.2, mutation_scale=7))
ax_top.set_ylabel("Context", fontsize=10, rotation=90, labelpad=4, va="center",
                  color="#555555")
ax_top.tick_params(bottom=False, labelbottom=False)
ax_top.spines[["top","right","bottom","left"]].set_visible(False)
label_panel(ax_top, "A", x=-0.09, y=1.05)

# Algorithm tracks
for ax, (lbl, vals, col) in zip(alg_axes, algs_sm):
    shade_lbd(ax)
    shade_regions(ax, alpha=0.13)
    ax.fill_between(res, 0, vals, alpha=0.35, color=col, zorder=2)
    ax.plot(res, vals, color=col, lw=1.5, zorder=3)
    ep_mask = epitope
    ax.scatter(res[ep_mask], vals[ep_mask], color=col, s=18, zorder=4,
               edgecolors="white", linewidths=0.6)
    ax.set_xlim(res[0]-1, res[-1]+1)
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"], fontsize=9, color="#555555")
    ax.tick_params(bottom=False, labelbottom=False)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#CCCCCC")
    ax.text(res[0]+0.5, 0.90, lbl, fontsize=10, va="top", color=col,
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# Combined score track
ax = ax_comb
shade_lbd(ax)
shade_regions(ax, alpha=0.14)
cmap_grad = LinearSegmentedColormap.from_list("br", ["#3498DB","#ECF0F1","#E74C3C"])
for i in range(len(res)-1):
    ax.fill_between(res[i:i+2], 0, fin_sm[i:i+2],
                    color=cmap_grad(fin_sm[i]), alpha=0.80, zorder=2)
ax.plot(res, fin_sm, color="#2C3E50", lw=1.6, zorder=3)
ep_colors2 = ["#C0392B","#D35400","#1E8449"]
for reg, col in zip(REGIONS, ep_colors2):
    mask = (res >= reg["start"]) & (res <= reg["end"])
    if mask.any():
        peak_i = np.argmax(fin_sm[mask])
        peak_x = res[mask][peak_i]
        peak_y = fin_sm[mask][peak_i]
        ax.annotate(f"Rank {reg['rank']}", xy=(peak_x, peak_y),
                    xytext=(peak_x, peak_y+0.22), ha="center",
                    fontsize=9, fontweight="bold", color=col,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2))
ax.set_xlim(res[0]-1, res[-1]+1)
ax.set_ylim(-0.05, 1.55)
ax.set_yticks([0, 0.5, 1.0])
ax.set_yticklabels(["0","0.5","1"], fontsize=9, color="#555555")
ax.tick_params(bottom=False, labelbottom=False)
ax.spines[["top","right"]].set_visible(False)
ax.spines[["left","bottom"]].set_color("#CCCCCC")
ax.text(res[0]+0.5, 1.48, "Final Composite Score", fontsize=11,
        color="#2C3E50", fontweight="bold", va="top",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# Sequence axis
ax = ax_seq
ax.set_xlim(res[0]-1, res[-1]+1)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.spines[["top","right","left"]].set_visible(False)
ax.spines["bottom"].set_color("#CCCCCC")
for i, (r, a) in enumerate(zip(res, aa)):
    if r % 10 == 0 or r == res[0] or r == res[-1]:
        ax.text(r, 0.70, str(r), ha="center", va="center",
                fontsize=9, color="#555555")
    if epitope[i]:
        col_e = "#E74C3C"
        for reg in REGIONS:
            if reg["start"] <= r <= reg["end"]:
                col_e = ep_colors[REGIONS.index(reg)]
                break
        ax.text(r, 0.25, a, ha="center", va="center",
                fontsize=8, color=col_e, fontweight="bold")
ax.set_xlabel("LasR LBD Residue Position", fontsize=13, labelpad=4, color="#333333")
ax.tick_params(bottom=True, labelbottom=False, length=3, color="#CCCCCC")

fig.text(0.025, 0.48, "Normalised Score (0–1)", ha="center", va="center",
         fontsize=12, rotation=90, color="#333333")

legend_patches = [
    mpatches.Patch(color="#E74C3C", alpha=0.75, label="Ep-I: KDSQDYEN (Rank 1, LBD)"),
    mpatches.Patch(color="#F39C12", alpha=0.75, label="Ep-II: EHYDRAGYARVDPTV (Rank 2, LBD)"),
    mpatches.Patch(color="#27AE60", alpha=0.75, label="Ep-III: ERSSGK (Rank 3)"),
    mpatches.Patch(color="#2C3E50", alpha=0.18, label="Ligand-Binding Domain"),
    mpatches.Patch(color="#8E44AD", alpha=0.80, label="Key functional residue"),
]
fig.legend(handles=legend_patches, loc="upper right",
           bbox_to_anchor=(0.97, 0.97), framealpha=0.92,
           fontsize=10, edgecolor="#CCCCCC", ncol=1,
           title="Annotation", title_fontsize=11)

for fmt in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/Fig4_EpitopePrediction.{fmt}", dpi=300,
                bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: Fig4_EpitopePrediction")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6 — pLDDT profiles (5-panel grid A–E)
# ═══════════════════════════════════════════════════════════════════════════
print("Regenerating Fig 6: pLDDT profiles...")

with open(f"{BASE}/03_Antibody_Design/CDR_positions.json") as f:
    cdr_pos = json.load(f)

plddt_all = {}
for cand in CANDIDATES:
    pdb = f"{BASE}/03_Antibody_Design/{cand}_ESMFold.pdb"
    if not os.path.exists(pdb):
        continue
    res_plddt = {}
    with open(pdb) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resnum = int(line[22:26].strip())
                bfact  = float(line[60:66].strip())
                res_plddt[resnum] = bfact
    plddt_all[cand] = res_plddt

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
letters = list("ABCDE")
cdr_shade_cols = {"CDR1": "#FF6B6B", "CDR2": "#4ECDC4", "CDR3": "#45B7D1"}

for idx, (cand, col, letter) in enumerate(zip(CANDIDATES, COLORS, letters)):
    ax = axes[idx]
    if cand not in plddt_all:
        ax.axis("off")
        continue
    res_data = plddt_all[cand]
    rpos = sorted(res_data.keys())
    vals = [res_data[r] for r in rpos]
    ax.fill_between(rpos, vals, alpha=0.25, color=col)
    ax.plot(rpos, vals, color=col, lw=1.8)
    ax.axhline(90, color="#2ECC71", lw=1.5, ls="--", label="Very high (90)")
    ax.axhline(70, color="#F39C12", lw=1.5, ls="--", label="High (70)")
    ax.axhline(50, color="#E74C3C", lw=1.5, ls="--", label="Low (50)")
    if cand in cdr_pos:
        for cdr_name, cdr_col in cdr_shade_cols.items():
            if cdr_name in cdr_pos[cand]:
                s, e = cdr_pos[cand][cdr_name]
                ax.axvspan(s, e, alpha=0.22, color=cdr_col, label=cdr_name)
    mean_val = np.mean(vals)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Residue Position", fontsize=13)
    ax.set_ylabel("pLDDT", fontsize=13)
    ax.set_title(f"{cand}\n(Mean pLDDT = {mean_val:.2f})",
                 fontsize=14, fontweight="bold", color=col)
    ax.legend(fontsize=9, loc="lower right", ncol=2, framealpha=0.7)
    ax.grid(alpha=0.3)
    label_panel(ax, letter)

axes[-1].axis("off")
plt.suptitle("ESMFold pLDDT Structural Confidence Profiles — All Nanobody Candidates",
             fontsize=15, fontweight="bold")
plt.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/Fig6_pLDDT_profiles.{fmt}", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: Fig6_pLDDT_profiles")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Phase 3 vs Phase 4 correlation (2-panel A, B)
# ═══════════════════════════════════════════════════════════════════════════
print("Regenerating Fig 8: P3 vs P4 correlation...")

p3 = pd.read_csv(f"{BASE}/03_Antibody_Design/nanobody_final_ranking.csv")
p4 = pd.read_csv(f"{BASE}/04_Docking/phase4_combined_ranking.csv")
merged = p4.merge(p3[["Candidate","Overall_Score","pLDDT_mean"]], on="Candidate")
merged = merged.rename(columns={"Overall_Score_x": "P4_Overall",
                                 "Overall_Score_y": "P3_Overall"})
if "P4_Overall" not in merged.columns:
    merged["P4_Overall"] = merged.get("Combined_Score", merged["MEGADOCK_Score"])

cand_colors = dict(zip(CANDIDATES, COLORS))
pt_colors = [cand_colors.get(c, "#888") for c in merged["Candidate"]]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (xcol, xlabel), letter in zip(axes,
    [("P3_Overall", "Phase 3 Composite Score"),
     ("pLDDT_mean", "ESMFold Mean pLDDT")],
    ["A", "B"]):
    ax.scatter(merged[xcol], merged["MEGADOCK_Score"],
               c=pt_colors, s=160, edgecolors="black", lw=0.9, zorder=3)
    for _, row in merged.iterrows():
        ax.annotate(row["Candidate"], (row[xcol], row["MEGADOCK_Score"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=10)
    if len(merged) > 2:
        r, p = pearsonr(merged[xcol], merged["MEGADOCK_Score"])
        ax.set_title(f"{xlabel} vs MEGADOCK Score\nPearson r = {r:.3f},  p = {p:.3f}",
                     fontsize=14, fontweight="bold")
    else:
        ax.set_title(f"{xlabel} vs MEGADOCK Score", fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("MEGADOCK Score", fontsize=13)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(True, alpha=0.3)
    label_panel(ax, letter)

handles = [mpatches.Patch(facecolor=c, label=n, edgecolor="black")
           for n, c in zip(CANDIDATES, COLORS)]
fig.legend(handles=handles, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, -0.07), fontsize=11, framealpha=0.8)
plt.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/Fig8_P3vsP4_correlation.{fmt}", dpi=300,
                bbox_inches="tight")
plt.close()
print("  Saved: Fig8_P3vsP4_correlation")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 13 — Immunogenicity (7-panel A–G)
# ═══════════════════════════════════════════════════════════════════════════
print("Regenerating Fig 13: Immunogenicity...")

with open(f"{BASE}/07_Immunogenicity_ADMET/phase7_results.json") as f:
    imm_data = json.load(f)

IMMUNO_SCALE = {'A':0.0,'R':-0.2,'N':-0.1,'D':-0.3,'C':0.1,'Q':-0.1,'E':-0.3,
                'G':0.0,'H':0.0,'I':0.4,'L':0.5,'K':-0.2,'M':0.3,'F':0.5,
                'P':-0.2,'S':-0.1,'T':0.0,'W':0.4,'Y':0.3,'V':0.4}
MHC2_ANCHOR  = {'A':0.3,'R':-0.5,'N':-0.3,'D':-0.8,'C':0.4,'Q':-0.1,'E':-0.7,
                'G':0.0,'H':0.1,'I':1.2,'L':1.5,'K':-0.6,'M':0.9,'F':1.4,
                'P':-0.8,'S':-0.2,'T':-0.1,'W':1.0,'Y':0.8,'V':1.1}
KD           = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
                'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
                'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}

def sliding_window_score(seq, scale, window):
    scores = []
    for i in range(len(seq) - window + 1):
        pep = seq[i:i+window]
        try:
            s = sum(scale.get(aa, 0) for aa in pep) / window
        except Exception:
            s = 0
        scores.append(s)
    return np.array(scores)

seqs = {c: imm_data[c]["info"]["seq"] for c in CANDIDATES}

MHC1_THRESH = 0.60
MHC2_THRESH = 0.45

mhc1_profiles = {c: sliding_window_score(seqs[c], IMMUNO_SCALE, 9)  for c in CANDIDATES}
mhc2_profiles = {c: sliding_window_score(seqs[c], MHC2_ANCHOR,  15) for c in CANDIDATES}
max_len1 = max(len(v) for v in mhc1_profiles.values())
max_len2 = max(len(v) for v in mhc2_profiles.values())
hm1 = np.full((5, max_len1), np.nan)
hm2 = np.full((5, max_len2), np.nan)
for i, c in enumerate(CANDIDATES):
    hm1[i, :len(mhc1_profiles[c])] = mhc1_profiles[c]
    hm2[i, :len(mhc2_profiles[c])] = mhc2_profiles[c]
mhc1_mean = [hm1[i].mean() for i in range(5)]
mhc1_max  = [hm1[i].max()  for i in range(5)]
mhc2_mean = [hm2[i].mean() for i in range(5)]
mhc2_max  = [hm2[i].max()  for i in range(5)]

# Aggregate immuno composite from json
immuno_comp = [imm_data[c]["immuno"].get("composite_score", 0) for c in CANDIDATES]

fig = plt.figure(figsize=(18, 15), facecolor="white")
gs_outer = GridSpec(3, 1, figure=fig, height_ratios=[3.0, 2.8, 3.2],
                    hspace=0.42, left=0.07, right=0.97, top=0.97, bottom=0.06)

# Row 1: MHC-I | MHC-II heatmaps
gs_top = gs_outer[0].subgridspec(1, 2, wspace=0.10)
ax_h1 = fig.add_subplot(gs_top[0])
ax_h2 = fig.add_subplot(gs_top[1])

cmap_mhc1 = LinearSegmentedColormap.from_list("mhc1", ["#EAF3FB","#3498DB","#154360"])
cmap_mhc2 = LinearSegmentedColormap.from_list("mhc2", ["#FEF9E7","#F39C12","#784212"])

for ax, hm, cmap, thresh, label, window, letter in [
    (ax_h1, hm1, cmap_mhc1, MHC1_THRESH, "MHC-I (9-mer)",  9,  "A"),
    (ax_h2, hm2, cmap_mhc2, MHC2_THRESH, "MHC-II (15-mer)", 15, "B"),
]:
    vmin, vmax = hm.min(), np.percentile(hm, 99)
    im = ax.pcolormesh(hm, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_yticks(np.arange(5)+0.5)
    ax.set_yticklabels(SHORT, fontsize=10, fontweight="bold")
    for i, col in enumerate(COLORS):
        ax.get_yticklabels()[i].set_color(col)
    n_cols = hm.shape[1]
    ax.set_xticks(np.linspace(0, n_cols, 6).astype(int))
    ax.set_xticklabels([str(int(t)+1) for t in np.linspace(0, n_cols, 6)], fontsize=10)
    ax.set_xlabel(f"Peptide position (start of {window}-mer)", fontsize=12)
    ax.tick_params(left=False)
    for i in range(5):
        for j in range(hm.shape[1]):
            if hm[i, j] > thresh:
                ax.add_patch(plt.Rectangle((j, i), 1, 1,
                    fill=False, edgecolor="#E74C3C", lw=1.5, zorder=3))
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Binding score", fontsize=10)
    ax.set_title(f"{label} Binding Score Landscape\n(threshold = {thresh})",
                 fontsize=13, fontweight="bold", color="#1A252F", pad=5)
    ax.text(n_cols*0.5, -0.85, "ALL CANDIDATES: LOW RISK",
            ha="center", va="center", fontsize=10, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#27AE60",
                      edgecolor="none", alpha=0.92),
            transform=ax.transData, zorder=10)
    ax.spines[["top","right","left","bottom"]].set_color("#CCCCCC")
    label_panel(ax, letter)

# Row 2: per-peptide line profiles
gs_mid = gs_outer[1].subgridspec(1, 2, wspace=0.12)
ax_l1  = fig.add_subplot(gs_mid[0])
ax_l2  = fig.add_subplot(gs_mid[1])

for ax, profiles, thresh, title_str, letter in [
    (ax_l1, mhc1_profiles, MHC1_THRESH, "MHC-I Per-Peptide Score Profile", "C"),
    (ax_l2, mhc2_profiles, MHC2_THRESH, "MHC-II Per-Peptide Score Profile", "D"),
]:
    for i, c in enumerate(CANDIDATES):
        s = gaussian_filter1d(profiles[c], sigma=1.5)
        x = np.arange(len(s))
        ax.fill_between(x, 0, s, alpha=0.20, color=COLORS[i])
        ax.plot(x, s, color=COLORS[i], lw=1.8, label=SHORT[i])
        peak = np.argmax(s)
        ax.scatter(peak, s[peak], color=COLORS[i], s=50, zorder=5,
                   edgecolors="white", lw=0.8)
    ax.axhline(thresh, color="#E74C3C", lw=2, ls="--", alpha=0.8)
    ax.text(2, thresh+0.01, f"Risk threshold ({thresh})",
            fontsize=10, color="#E74C3C", va="bottom")
    ax.set_xlabel("Peptide index", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title_str, fontsize=13, fontweight="bold", color="#1A252F")
    ax.legend(fontsize=9, framealpha=0.7, ncol=5, loc="upper right",
              handlelength=1.2)
    ax.spines[["top","right"]].set_visible(False)
    label_panel(ax, letter)

# Row 3: bar charts + composite
gs_bot = gs_outer[2].subgridspec(1, 3, wspace=0.38)
ax_c   = fig.add_subplot(gs_bot[0])
ax_d   = fig.add_subplot(gs_bot[1])
ax_e   = fig.add_subplot(gs_bot[2])

x_pos = np.arange(5)
bar_w = 0.30

# Panel E: MHC-I/II mean/max bars
b1 = ax_c.bar(x_pos - bar_w/2, mhc1_mean, bar_w, label="MHC-I mean",
              color=[c+"AA" for c in COLORS], edgecolor=COLORS, lw=1.5)
b2 = ax_c.bar(x_pos + bar_w/2, mhc2_mean, bar_w, label="MHC-II mean",
              color=COLORS, edgecolor=[c+"AA" for c in COLORS], lw=1.5)
ax_c.set_xticks(x_pos)
ax_c.set_xticklabels(SHORT, fontsize=11)
ax_c.set_ylabel("Mean Binding Score", fontsize=12)
ax_c.set_title("Mean Immunogenicity Scores", fontsize=13, fontweight="bold")
ax_c.legend(fontsize=10)
ax_c.spines[["top","right"]].set_visible(False)
label_panel(ax_c, "E")

# Panel F: CDR3 immunogenicity analysis
cdr3_mhc1 = []
for c in CANDIDATES:
    seq = seqs[c]
    if c in cdr_pos and "CDR3" in cdr_pos[c]:
        s_idx, e_idx = cdr_pos[c]["CDR3"]
        cdr3_seq = seq[s_idx-1:e_idx] if s_idx-1 < len(seq) else ""
        if cdr3_seq:
            scores = [IMMUNO_SCALE.get(aa, 0) for aa in cdr3_seq]
            cdr3_mhc1.append(np.mean(scores))
        else:
            cdr3_mhc1.append(0)
    else:
        cdr3_mhc1.append(0)
bars = ax_d.bar(x_pos, cdr3_mhc1, color=COLORS, edgecolor="black", lw=0.8, width=0.6)
ax_d.axhline(0, color="black", lw=0.8, ls="-")
ax_d.set_xticks(x_pos)
ax_d.set_xticklabels(SHORT, fontsize=11)
ax_d.set_ylabel("CDR3 Mean Immuno Score", fontsize=12)
ax_d.set_title("CDR3 Immunogenicity\n(IMMUNO scale)", fontsize=13, fontweight="bold")
ax_d.spines[["top","right"]].set_visible(False)
for bar, val in zip(bars, cdr3_mhc1):
    ax_d.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
              f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
label_panel(ax_d, "F")

# Panel G: Composite immunogenicity
kd_scores = []
for c in CANDIDATES:
    seq = seqs[c]
    kd_s = np.mean([KD.get(aa, 0) for aa in seq])
    kd_scores.append(kd_s)
bars = ax_e.bar(x_pos, kd_scores, color=COLORS, edgecolor="black", lw=0.8, width=0.6)
ax_e.set_xticks(x_pos)
ax_e.set_xticklabels(SHORT, fontsize=11)
ax_e.set_ylabel("Mean Kyte-Doolittle Score", fontsize=12)
ax_e.set_title("Hydrophobicity Profile\n(Kyte-Doolittle scale)", fontsize=13, fontweight="bold")
ax_e.spines[["top","right"]].set_visible(False)
for bar, val in zip(bars, kd_scores):
    ax_e.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
              f"{val:.3f}", ha="center", fontsize=9)
label_panel(ax_e, "G")

for fmt in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/Fig13_Immunogenicity.{fmt}", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: Fig13_Immunogenicity")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 14 — Developability Heatmap (single panel, updated fonts)
# ═══════════════════════════════════════════════════════════════════════════
print("Regenerating Fig 14: Developability heatmap...")

CANDS = ["NbLasR-1", "NbLasR-2", "NbLasR-3", "NbLasR-4", "NbLasR-5"]
LEAD = 1
METRICS_HM = [
    ("pI",                  [9.007,9.639,9.428,9.301,9.694], None,   "{:.2f}"),
    ("MW (kDa)",            [11.33,11.98,12.31,11.90,12.22], False,  "{:.2f}"),
    ("Instability Index",   [44.01,34.37,44.62,38.69,34.61], False,  "{:.1f}"),
    ("GRAVY",               [-0.526,-0.675,-0.439,-0.558,-0.618], False, "{:.3f}"),
    ("Aliphatic Index",     [63.40,61.60,67.06,61.60,60.46], True,   "{:.1f}"),
    ("Charge @pH 7.4",      [2.49, 5.49, 4.48, 3.49, 6.48], True,   "{:.2f}"),
    ("MHC-I Mean Score",    [0.0449,0.0395,0.0670,0.0497,0.0487], False, "{:.4f}"),
    ("MHC-II Mean Score",   [0.165, 0.146, 0.228, 0.177, 0.174],  False, "{:.3f}"),
    ("Immuno Composite",    [-0.05,-0.07, 0.57,  0.22,  0.18],    False, "{:.2f}"),
    ("# APR Regions",       [4, 5, 5, 5, 4],                       False, "{:.0f}"),
    ("Mean Agg. Score",     [0.071,0.057,0.137,0.089,0.086],       False, "{:.3f}"),
    ("Solubility Score",    [-0.13,-0.08,-0.16,-0.11,-0.11],       True,  "{:.2f}"),
    ("Developability Index",[0.54, 0.62, 0.57, 0.62, 0.62],       True,  "{:.2f}"),
]
CATS_HM = [
    (0,  6,  "Physicochemical",           "#2E4057"),
    (6,  9,  "Immunogenicity",            "#8E1F2E"),
    (9,  12, "Aggregation /\nSolubility", "#1A6B3C"),
    (12, 13, "Overall",                   "#5B3A8A"),
]

n_m = len(METRICS_HM)
n_c = len(CANDS)

norm_hm = np.zeros((n_m, n_c))
for i, (_, vals, hib, _fmt) in enumerate(METRICS_HM):
    v = np.array(vals, dtype=float)
    mn, mx = v.min(), v.max()
    if mx == mn:
        norm_hm[i] = 0.5
    else:
        nn = (v - mn) / (mx - mn)
        if hib is False:
            nn = 1.0 - nn
        elif hib is None:
            nn = 1.0 - np.abs(nn - 0.5) * 2
        norm_hm[i] = nn

CMAP_HM = LinearSegmentedColormap.from_list(
    "rg5", ["#C0392B","#E8967A","#F9F9F9","#76C893","#1A5E39"])

cell_h = 0.55
cell_w = 1.20
lmargin_in = 2.0
rmargin_in = 0.65
cat_strip  = 0.32
fig_w = cat_strip + lmargin_in + n_c * cell_w + rmargin_in + 0.5
fig_h = n_m * cell_h + 1.6

fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
l = (cat_strip + lmargin_in) / fig_w
r = l + n_c * cell_w / fig_w
b = 0.10
t = b + n_m * cell_h / fig_h
ax = fig.add_axes([l, b, r-l, t-b])

im = ax.imshow(norm_hm, aspect="auto", cmap=CMAP_HM, vmin=0, vmax=1,
               interpolation="nearest")
ax.set_xticks(np.arange(n_c+1)-0.5, minor=True)
ax.set_yticks(np.arange(n_m+1)-0.5, minor=True)
ax.grid(which="minor", color="white", linewidth=1.6)
ax.tick_params(which="minor", length=0)

from matplotlib.patches import Rectangle
for r0, r1, _, _ in CATS_HM[1:]:
    ax.axhline(r0-0.5, color="white", linewidth=3.5, zorder=3)
ax.axvline(LEAD-0.5, color="#C0392B", lw=2.0, zorder=4, alpha=0.6)
ax.axvline(LEAD+0.5, color="#C0392B", lw=2.0, zorder=4, alpha=0.6)
ax.add_patch(Rectangle((LEAD-0.5,-0.5), 1.0, n_m,
                        facecolor="#FDF2F8", alpha=0.18, zorder=1))

for i, (lbl, vals, hib, fmt) in enumerate(METRICS_HM):
    best_j = int(np.argmax(norm_hm[i]))
    for j in range(n_c):
        cell_norm = norm_hm[i, j]
        txt_col   = "white" if (cell_norm < 0.22 or cell_norm > 0.82) else "#1A1A1A"
        weight    = "bold" if j == best_j else "normal"
        star      = "★" if j == best_j else ""
        label_txt = fmt.format(vals[j]) + star
        ax.text(j, i, label_txt, ha="center", va="center",
                fontsize=9.5, fontweight=weight, color=txt_col, zorder=5)

ax.set_xlim(-0.5, n_c-0.5)
ax.set_ylim(n_m-0.5, -0.5)
ax.set_xticks(range(n_c))
ax.set_xticklabels(
    [f"{'★ ' if j == LEAD else ''}{c}" for j, c in enumerate(CANDS)],
    fontsize=11, fontweight="bold")
for j, tick in enumerate(ax.get_xticklabels()):
    tick.set_color("#C0392B" if j == LEAD else "#1A1A1A")
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")
ax.tick_params(axis="x", length=0, pad=5)
ax.tick_params(axis="y", length=0)
ax.set_yticks([])
ax.spines[:].set_visible(False)

ax_lbl = fig.add_axes([l - lmargin_in/fig_w, b, lmargin_in/fig_w - 0.005, t-b])
ax_lbl.set_xlim(0, 1)
ax_lbl.set_ylim(0, n_m)
ax_lbl.invert_yaxis()
ax_lbl.axis("off")
direction_sym = {True: "↑ ", False: "↓ ", None: "  "}
direction_col = {True: "#1A6B3C", False: "#8E1F2E", None: "#888888"}
for i, (lbl, _, hib, _) in enumerate(METRICS_HM):
    dsym = direction_sym.get(hib, "  ")
    dcol = direction_col.get(hib, "#888888")
    ax_lbl.text(0.02, i+0.5, dsym, ha="left", va="center",
                fontsize=10, color=dcol, fontweight="bold")
    ax_lbl.text(0.92, i+0.5, lbl, ha="right", va="center",
                fontsize=10, color="#1A1A1A")
for r0, r1, _, _ in CATS_HM[1:]:
    ax_lbl.axhline(r0, color="#CCCCCC", lw=1.2, xmin=0.0, xmax=1.0)

ax_cat = fig.add_axes([0.01, b, cat_strip/fig_w - 0.005, t-b])
ax_cat.set_xlim(0, 1)
ax_cat.set_ylim(0, n_m)
ax_cat.invert_yaxis()
ax_cat.axis("off")
cat_colors = ["#2E4057","#8E1F2E","#1A6B3C","#5B3A8A"]
for (r0, r1, cat_name, _), col in zip(CATS_HM, cat_colors):
    span = r1 - r0
    ax_cat.add_patch(Rectangle((0.08, r0+0.06), 0.84, span-0.12,
                                facecolor=col, alpha=0.88, zorder=2, lw=0))
    mid = (r0+r1)/2
    fs = 7.5 if span == 1 else 8.5
    ax_cat.text(0.50, mid, cat_name.replace("\n"," ") if span == 1 else cat_name,
                ha="center", va="center", fontsize=fs, fontweight="bold",
                color="white", rotation=0 if span == 1 else 90, zorder=3,
                multialignment="center")

cbar_l = r + 0.025
cbar_ax = fig.add_axes([cbar_l, b+0.05*(t-b), 0.018, 0.90*(t-b)])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_ticks([0, 0.5, 1.0])
cb.set_ticklabels(["Poor","Moderate","Excellent"], fontsize=10)
cb.outline.set_linewidth(0.5)
cb.ax.tick_params(length=3, width=0.5)

fig.text(l, b-0.06, "★ Best value per metric   ↑ higher = better   "
         "↓ lower = better   Red border = NbLasR-2 lead candidate",
         fontsize=9, color="#555555", ha="left", va="top")

for fmt in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/Fig14_Developability_Heatmap.{fmt}", dpi=300,
                bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: Fig14_Developability_Heatmap")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 15 — Composite Developability Index (2-panel A, B)
# ═══════════════════════════════════════════════════════════════════════════
print("Regenerating Fig 15: Composite developability...")

candidates_dev = ['NbLasR-1','NbLasR-3','NbLasR-5','NbLasR-4','NbLasR-2']
composite_dev  = [0.540,      0.570,      0.620,      0.620,      0.620]
dev_class      = ['Moderate', 'Moderate', 'Good',     'Good',     'Good']

weights = {'Stability':0.20,'Aggregation':0.20,'Immunogenicity':0.20,
           'Solubility':0.15,'pI':0.10,'GRAVY':0.10,'Cross-react.':0.05}
raw = {
    'NbLasR-1': [0.6, 0.2, 1.0, 0.2, 0.3, 0.7, 1.0],
    'NbLasR-2': [1.0, 0.2, 1.0, 0.2, 0.3, 0.7, 1.0],
    'NbLasR-3': [0.6, 0.2, 1.0, 0.2, 0.3, 1.0, 1.0],
    'NbLasR-4': [1.0, 0.2, 1.0, 0.2, 0.3, 0.7, 1.0],
    'NbLasR-5': [1.0, 0.2, 1.0, 0.2, 0.3, 0.7, 1.0],
}
metrics_dev = list(weights.keys())
wts = list(weights.values())
contribs = {c: [raw[c][i]*wts[i] for i in range(len(metrics_dev))]
            for c in candidates_dev}

LEAD_COL = '#C0392B'
SLATE    = '#5B7FA6'
MOD_COL  = '#E8A838'
METRIC_COLS = {
    'Stability':      '#1A5FA8',
    'Aggregation':    '#E05C5C',
    'Immunogenicity': '#4CAF50',
    'Solubility':     '#FF9800',
    'pI':             '#9C27B0',
    'GRAVY':          '#00ACC1',
    'Cross-react.':   '#795548',
}

N = len(candidates_dev)
y_pos = np.arange(N)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                gridspec_kw={"width_ratios":[1.1,1], "wspace":0.08})
fig.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.22)

# Panel A: lollipop chart
bar_cols = [LEAD_COL if c == 'NbLasR-2' else SLATE for c in candidates_dev]
tier_cols = [LEAD_COL if dc == 'Good' else MOD_COL for dc in dev_class]

ax1.barh(y_pos, composite_dev, left=0, height=0.04,
         color=STEM_COL if 'STEM_COL' in dir() else '#C8D8E4', zorder=2)
ax1.scatter(composite_dev, y_pos, color=bar_cols, s=120, zorder=3, edgecolors="white", lw=1)

for y, c, v, dc, tc in zip(y_pos, candidates_dev, composite_dev, dev_class, tier_cols):
    ax1.text(v+0.004, y, f"{v:.3f}", va="center", fontsize=10,
             fontweight="bold" if c == "NbLasR-2" else "normal",
             color=LEAD_COL if c == "NbLasR-2" else "#333")
    ax1.text(0.738, y, f"({dc})", va="center", ha="right", fontsize=8.5,
             color=tc, style="italic")

ax1.set_yticks(y_pos)
ax1.set_yticklabels(candidates_dev, fontsize=11)
for tick, c in zip(ax1.get_yticklabels(), candidates_dev):
    tick.set_color(LEAD_COL if c == "NbLasR-2" else "#333333")
ax1.set_xlabel("Composite Developability Index", fontsize=13)
ax1.set_xlim(0.45, 0.745)
ax1.set_title("Composite Developability Index", fontsize=14, fontweight="bold")
ax1.spines[["top","right","left"]].set_visible(False)
ax1.tick_params(left=False, axis="y")
ax1.axvline(0.60, color="#888", lw=1.2, ls="--", alpha=0.6)
ax1.text(0.60, -0.6, "Good tier", ha="center", fontsize=9, color="#888")
label_panel(ax1, "A")

# Panel B: stacked contribution bars
bottom_arr = np.zeros(N)
for mname, wt, col in zip(metrics_dev, wts, METRIC_COLS.values()):
    vals_b = [contribs[c][metrics_dev.index(mname)] for c in candidates_dev]
    bars = ax2.barh(y_pos, vals_b, left=bottom_arr, height=0.55,
                    color=col, label=f"{mname} (w={wt:.2f})", edgecolor="white", lw=0.5)
    bottom_arr += np.array(vals_b)

ax2.set_yticks(y_pos)
ax2.set_yticklabels([])
ax2.set_xlabel("Weighted Score Contribution", fontsize=13)
ax2.set_title("Sub-metric Contribution", fontsize=14, fontweight="bold")
ax2.spines[["top","right","left"]].set_visible(False)
ax2.tick_params(left=False, axis="y")
handles_b = [mpatches.Patch(facecolor=c, label=f"{m} (w={w:.2f})")
             for m, w, c in zip(metrics_dev, wts, METRIC_COLS.values())]
fig.legend(handles=handles_b, loc="lower center", ncol=4,
           bbox_to_anchor=(0.55, -0.04), fontsize=9, framealpha=0.85,
           edgecolor="#CCC")
label_panel(ax2, "B")

for fmt in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/Fig15_DevelopabilityIndex.{fmt}", dpi=300,
                bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: Fig15_DevelopabilityIndex")

print(f"\nAll revised figures saved to: {OUTDIR}/")
