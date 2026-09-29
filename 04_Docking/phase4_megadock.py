#!/usr/bin/env python3
"""
Phase 4: Protein-Protein Docking with MEGADOCK-GPU
LasR receptor (2UV0 chain E) vs NbLasR-1..5 (ESMFold structures)

MEGADOCK reference:
  Ohue M et al. "MEGADOCK 4.0: an ultra-high-performance protein-protein
  docking software for heterogeneous supercomputers." Bioinformatics 2014.
"""

import os, sys, subprocess, json, time, shutil, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE = Path("/home/stalin/Desktop/LasR_Antibody_Design")
DOCK_DIR = BASE / "04_Docking"
NB_DIR   = BASE / "03_Antibody_Design"
FIG_DIR  = BASE / "08_Results_Figures/Phase4"
LOG_DIR  = BASE / "logs"
MEGADOCK = BASE / "tools/MEGADOCK"

RECEPTOR = DOCK_DIR / "LasR_chainE_clean.pdb"
MEGADOCK_BIN  = MEGADOCK / "megadock-gpu"
DECOYGEN_BIN  = MEGADOCK / "decoygen"
PPISCORE_BIN  = MEGADOCK / "ppiscore"

FIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = ["NbLasR-1", "NbLasR-2", "NbLasR-3", "NbLasR-4", "NbLasR-5"]
N_ROTATIONS = 10800   # MEGADOCK default full-rotation sampling
N_TOP_POSES = 10      # top poses to extract per candidate

LOG_FILE = LOG_DIR / f"phase4_megadock_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def run_cmd(cmd, cwd=None):
    """Run shell command, return (stdout, stderr, returncode)."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    return result.stdout, result.stderr, result.returncode

# ─── Step 1: Prepare working directory ───────────────────────────────────────
log("=" * 70)
log("PHASE 4 — MEGADOCK-GPU Protein-Protein Docking")
log("=" * 70)

WORK_DIR = DOCK_DIR / "megadock_runs"
WORK_DIR.mkdir(exist_ok=True)

# Copy binaries to work dir for convenience
for b in [MEGADOCK_BIN, DECOYGEN_BIN, PPISCORE_BIN]:
    dest = WORK_DIR / b.name
    if not dest.exists():
        shutil.copy(b, dest)
        os.chmod(dest, 0o755)

log(f"Working directory: {WORK_DIR}")
log(f"Receptor: {RECEPTOR.name}  ({sum(1 for l in open(RECEPTOR) if l.startswith('ATOM'))} ATOM records)")
log(f"Candidates: {CANDIDATES}")
log(f"N rotations: {N_ROTATIONS}")

# ─── Step 2: Run MEGADOCK for each candidate ─────────────────────────────────
results = {}

for nb in CANDIDATES:
    ligand = NB_DIR / f"{nb}_ESMFold.pdb"
    if not ligand.exists():
        log(f"  [SKIP] {nb}: ligand PDB not found at {ligand}")
        continue

    nb_dir = WORK_DIR / nb
    nb_dir.mkdir(exist_ok=True)

    # Copy PDBs to run dir
    rec_local = nb_dir / "receptor.pdb"
    lig_local = nb_dir / "ligand.pdb"
    shutil.copy(RECEPTOR, rec_local)
    shutil.copy(ligand, lig_local)

    out_file = nb_dir / "dock.out"
    log(f"\n--- Running MEGADOCK for {nb} ---")
    log(f"  Ligand atoms: {sum(1 for l in open(lig_local) if l.startswith('ATOM'))}")

    t_start = time.time()
    cmd = (f"{WORK_DIR}/megadock-gpu "
           f"-R {rec_local} "
           f"-L {lig_local} "
           f"-N {N_ROTATIONS} "
           f"-t 1 "
           f"-o {out_file}")

    log(f"  CMD: {cmd}")
    stdout, stderr, rc = run_cmd(cmd, cwd=str(nb_dir))
    elapsed = time.time() - t_start

    # Save raw output
    with open(nb_dir / "megadock_stdout.txt", "w") as f:
        f.write(stdout)
    with open(nb_dir / "megadock_stderr.txt", "w") as f:
        f.write(stderr)

    if rc != 0 or not out_file.exists():
        log(f"  [ERROR] MEGADOCK failed (rc={rc})")
        log(f"  STDERR: {stderr[:500]}")
        results[nb] = {"status": "failed", "error": stderr[:200]}
        continue

    n_lines = sum(1 for _ in open(out_file))
    log(f"  MEGADOCK completed in {elapsed:.1f}s, output lines: {n_lines}")

    # ─── Step 3: Extract top poses with decoygen ─────────────────────────────
    log(f"  Extracting top {N_TOP_POSES} poses...")
    scores = []
    for rank in range(1, N_TOP_POSES + 1):
        pose_pdb = nb_dir / f"pose_{rank:02d}.pdb"
        cmd_dec = (f"{WORK_DIR}/decoygen {pose_pdb} {lig_local} {out_file} {rank}")
        stdout_d, stderr_d, rc_d = run_cmd(cmd_dec, cwd=str(nb_dir))
        if rc_d != 0:
            log(f"    decoygen rank {rank} failed: {stderr_d[:100]}")

    # ─── Step 4: Parse MEGADOCK scores from output file ──────────────────────
    # dock.out format:
    #   Line 1: grid_size  voxel_spacing
    #   Line 2: origin coords (3 floats)
    #   Line 3: receptor path + center
    #   Line 4: ligand path + center
    #   Lines 5+: angle1 angle2 angle3 tx ty tz SCORE  (7 columns)
    score_lines = []
    with open(out_file) as f:
        for i, line in enumerate(f):
            if i < 4:
                continue  # skip 4-line header
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 7:
                try:
                    score_lines.append(float(parts[6]))  # col 7 = MEGADOCK score
                except ValueError:
                    pass

    if score_lines:
        top_score = max(score_lines)
        top10_mean = np.mean(sorted(score_lines, reverse=True)[:10])
        log(f"  Best MEGADOCK score: {top_score:.4f}")
        log(f"  Top-10 mean score:   {top10_mean:.4f}")
    else:
        top_score = 0.0
        top10_mean = 0.0
        log("  [WARNING] Could not parse scores from output file")

    # ─── Step 5: PPI E-score ─────────────────────────────────────────────────
    # ppiscore uses "./dock.out" — must run from nb_dir with just filename
    log(f"  Computing PPI E-score...")
    cmd_ppi = f"perl {WORK_DIR}/ppiscore dock.out {N_ROTATIONS}"
    stdout_ppi, stderr_ppi, rc_ppi = run_cmd(cmd_ppi, cwd=str(nb_dir))
    with open(nb_dir / "ppiscore_output.txt", "w") as f:
        f.write(stdout_ppi)
        f.write(stderr_ppi)

    # Parse E-score: ppiscore output format: "dock.out, E = 2.7342, 10800 decoys"
    ppi_escore = None
    for line in stdout_ppi.split("\n") + stderr_ppi.split("\n"):
        line = line.strip()
        if "E =" in line:
            try:
                # Extract the value after "E = "
                val_str = line.split("E =")[1].strip().split(",")[0].strip()
                ppi_escore = float(val_str)
                break
            except (ValueError, IndexError):
                pass

    if ppi_escore is None:
        # Try to extract from full output
        log(f"  PPI stdout: {stdout_ppi[:200]}")
        ppi_escore = 0.0

    log(f"  PPI E-score: {ppi_escore:.3f}")

    # ─── Step 6: Build best complex PDB ──────────────────────────────────────
    best_pose = nb_dir / "pose_01.pdb"
    if best_pose.exists():
        complex_out = DOCK_DIR / f"{nb}_MEGADOCK_complex.pdb"
        with open(complex_out, "w") as fout:
            # Receptor chains
            with open(rec_local) as fr:
                for line in fr:
                    if line.startswith("ATOM") or line.startswith("TER"):
                        fout.write(line)
            fout.write("TER\n")
            # Best pose ligand
            with open(best_pose) as fl:
                for line in fl:
                    if line.startswith("ATOM") or line.startswith("TER"):
                        fout.write(line)
            fout.write("END\n")
        log(f"  Complex saved: {complex_out.name}")

    # ─── Step 7: Interface analysis of best pose ─────────────────────────────
    interface_residues = []
    contact_dist = 8.0  # Angstrom cutoff for interface

    if best_pose.exists():
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)

        rec_struct = parser.get_structure("rec", str(rec_local))
        lig_struct = parser.get_structure("lig", str(best_pose))

        rec_atoms = list(rec_struct.get_atoms())
        lig_atoms = list(lig_struct.get_atoms())

        rec_coords = np.array([a.get_vector().get_array() for a in rec_atoms])
        lig_coords = np.array([a.get_vector().get_array() for a in lig_atoms])

        # Find interface residues on receptor
        contacted = set()
        for i, ra in enumerate(rec_atoms):
            dists = np.linalg.norm(lig_coords - rec_coords[i], axis=1)
            if dists.min() <= contact_dist:
                res = ra.get_parent()
                contacted.add((res.get_resname(), res.get_id()[1]))

        interface_residues = sorted(contacted, key=lambda x: x[1])
        n_contacts = len(interface_residues)
        log(f"  Interface residues (receptor, ≤{contact_dist}Å): {n_contacts}")
        for rname, rnum in interface_residues[:8]:
            log(f"    {rname}{rnum}")

        # Buried surface area approximation (SASA reduction estimate)
        # Simple count-based: each interfacial atom ~10 Å²
        bsa_approx = n_contacts * 15.0  # rough: 15 Å² per residue
    else:
        n_contacts = 0
        bsa_approx = 0.0

    results[nb] = {
        "status": "success",
        "top_score": top_score,
        "top10_mean": top10_mean,
        "ppi_escore": ppi_escore,
        "n_interface_residues": n_contacts,
        "bsa_approx": bsa_approx,
        "interface_residues": [f"{r[0]}{r[1]}" for r in interface_residues],
        "elapsed_s": round(elapsed, 1)
    }
    log(f"  Done: {nb} — score={top_score:.4f}, E={ppi_escore:.3f}, BSA≈{bsa_approx:.0f}Å²")

# ─── Step 8: Save results ─────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("RESULTS SUMMARY")
log("=" * 70)

results_file = DOCK_DIR / "phase4_megadock_results.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)
log(f"Results JSON: {results_file}")

# Build results dataframe
rows = []
for nb, r in results.items():
    if r["status"] == "success":
        rows.append({
            "Candidate": nb,
            "MEGADOCK_Score": r["top_score"],
            "Top10_Mean": r["top10_mean"],
            "PPI_Escore": r["ppi_escore"],
            "N_Interface_Res": r["n_interface_residues"],
            "BSA_approx_A2": r["bsa_approx"],
            "Runtime_s": r["elapsed_s"]
        })
        log(f"  {nb}: MEGADOCK={r['top_score']:.4f}, E-score={r['ppi_escore']:.3f}, "
            f"Interface={r['n_interface_residues']} res, BSA≈{r['bsa_approx']:.0f}Å²")
    else:
        log(f"  {nb}: FAILED — {r.get('error','?')}")

if not rows:
    log("ERROR: No successful docking runs. Exiting.")
    sys.exit(1)

df = pd.DataFrame(rows)
df = df.sort_values("MEGADOCK_Score", ascending=False).reset_index(drop=True)
df["MEGADOCK_Rank"] = range(1, len(df) + 1)
df.to_csv(DOCK_DIR / "phase4_megadock_ranking.csv", index=False)
log(f"\nRanking CSV saved.")

# ─── Step 9: Figures ──────────────────────────────────────────────────────────
log("\nGenerating figures...")

PHASE_COLORS = {
    "NbLasR-1": "#1f77b4",
    "NbLasR-2": "#ff7f0e",
    "NbLasR-3": "#2ca02c",
    "NbLasR-4": "#d62728",
    "NbLasR-5": "#9467bd"
}

def colors_for(df_col):
    return [PHASE_COLORS.get(c, "#888") for c in df_col]

# ── Fig 25: MEGADOCK Scores bar chart ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(df["Candidate"], df["MEGADOCK_Score"],
              color=colors_for(df["Candidate"]), edgecolor="black", linewidth=0.8)
for bar, score in zip(bars, df["MEGADOCK_Score"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005 * df["MEGADOCK_Score"].max(),
            f"{score:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xlabel("Nanobody Candidate", fontsize=12)
ax.set_ylabel("MEGADOCK Score", fontsize=12)
ax.set_title("MEGADOCK-GPU Docking Scores\nNbLasR Candidates vs LasR (2UV0-E)", fontsize=13, fontweight="bold")
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(FIG_DIR / "Fig25_MEGADOCK_scores.png", dpi=150, bbox_inches="tight")
plt.close()
log("  Saved Fig25_MEGADOCK_scores.png")

# ── Fig 26: PPI E-score bar chart ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
colors_ppi = [PHASE_COLORS.get(c, "#888") for c in df["Candidate"]]
bars = ax.bar(df["Candidate"], df["PPI_Escore"],
              color=colors_ppi, edgecolor="black", linewidth=0.8)
for bar, val in zip(bars, df["PPI_Escore"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
# Reference lines
for thresh, label, color in [(8, "PPI Positive (>8, PPV~10%)", "#e67e22"),
                              (10, "PPV~50% (>10)", "#27ae60"),
                              (12, "PPV~80% (>12)", "#2980b9")]:
    ax.axhline(thresh, color=color, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(len(df)-0.4, thresh + 0.1, label, fontsize=7.5, color=color)
ax.set_xlabel("Nanobody Candidate", fontsize=12)
ax.set_ylabel("PPI E-score", fontsize=12)
ax.set_title("Protein-Protein Interaction E-scores (MEGADOCK)\nNbLasR Candidates vs LasR", fontsize=13, fontweight="bold")
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(FIG_DIR / "Fig26_PPI_Escore.png", dpi=150, bbox_inches="tight")
plt.close()
log("  Saved Fig26_PPI_Escore.png")

# ── Fig 27: Interface residues bar chart ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(df["Candidate"], df["N_Interface_Res"],
              color=colors_for(df["Candidate"]), edgecolor="black", linewidth=0.8)
for bar, val in zip(bars, df["N_Interface_Res"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(int(val)), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xlabel("Nanobody Candidate", fontsize=12)
ax.set_ylabel("Number of Interface Residues", fontsize=12)
ax.set_title("Interface Residue Count (≤8 Å contact)\nNbLasR Candidates vs LasR (2UV0-E)", fontsize=13, fontweight="bold")
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(FIG_DIR / "Fig27_interface_residues.png", dpi=150, bbox_inches="tight")
plt.close()
log("  Saved Fig27_interface_residues.png")

# ── Fig 28: Radar chart — multi-criteria comparison ──────────────────────────
from matplotlib.patches import FancyArrowPatch

categories = ["MEGADOCK\nScore", "PPI\nE-score", "Interface\nResidues", "BSA\n(×100 Å²)"]
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)

# Normalize each metric 0-1
def norm_col(col):
    mn, mx = col.min(), col.max()
    if mx == mn:
        return np.ones(len(col)) * 0.5
    return (col - mn) / (mx - mn)

norm_df = pd.DataFrame({
    "MEGADOCK\nScore": norm_col(df["MEGADOCK_Score"]),
    "PPI\nE-score": norm_col(df["PPI_Escore"]),
    "Interface\nResidues": norm_col(df["N_Interface_Res"]),
    "BSA\n(×100 Å²)": norm_col(df["BSA_approx_A2"]),
})
norm_df.index = df["Candidate"]

for nb in df["Candidate"]:
    vals = norm_df.loc[nb].tolist()
    vals += vals[:1]
    ax.plot(angles, vals, linewidth=2, label=nb, color=PHASE_COLORS.get(nb, "#888"))
    ax.fill(angles, vals, alpha=0.1, color=PHASE_COLORS.get(nb, "#888"))

ax.set_ylim(0, 1)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
ax.set_title("Multi-criteria Docking Performance\n(Normalized, all metrics: higher = better)",
             fontsize=12, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(FIG_DIR / "Fig28_radar_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
log("  Saved Fig28_radar_comparison.png")

# ── Fig 29: Score correlation with Phase 3 ranking ───────────────────────────
phase3_csv = BASE / "03_Antibody_Design/nanobody_final_ranking.csv"
if phase3_csv.exists():
    p3 = pd.read_csv(phase3_csv)
    merged = df.merge(p3[["Candidate", "Overall_Score", "pLDDT_mean"]], on="Candidate")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, xcol, xlabel in zip(axes,
                                 ["Overall_Score", "pLDDT_mean"],
                                 ["Phase 3 Overall Score", "ESMFold pLDDT (mean)"]):
        ax.scatter(merged[xcol], merged["MEGADOCK_Score"],
                   c=[PHASE_COLORS.get(c, "#888") for c in merged["Candidate"]],
                   s=120, edgecolors="black", linewidth=0.8, zorder=3)
        for _, row in merged.iterrows():
            ax.annotate(row["Candidate"], (row[xcol], row["MEGADOCK_Score"]),
                        textcoords="offset points", xytext=(6, 3), fontsize=8)
        from scipy.stats import pearsonr
        if len(merged) > 2:
            r, p = pearsonr(merged[xcol], merged["MEGADOCK_Score"])
            ax.set_title(f"{xlabel} vs MEGADOCK Score\nr={r:.3f}, p={p:.3f}", fontsize=11)
        else:
            ax.set_title(f"{xlabel} vs MEGADOCK Score", fontsize=11)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("MEGADOCK Score", fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Phase 3 Design Quality vs Phase 4 Docking Score", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig29_P3vsP4_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("  Saved Fig29_P3vsP4_correlation.png")

# ── Fig 30: Combined ranking (Phase 3 + Phase 4) ─────────────────────────────
if phase3_csv.exists():
    merged = df.merge(p3[["Candidate", "Overall_Score", "Final_Rank"]], on="Candidate")
    # Normalize and combine
    p4_norm = norm_col(merged["MEGADOCK_Score"])
    p3_norm = norm_col(merged["Overall_Score"])
    merged["Combined_Score"] = 0.4 * p3_norm + 0.6 * p4_norm
    merged = merged.sort_values("Combined_Score", ascending=False).reset_index(drop=True)
    merged["Combined_Rank"] = range(1, len(merged) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(merged))
    w = 0.3
    bars1 = ax.bar(x - w, p3_norm[merged.index], w, label="Phase 3 Design Score (40%)",
                   color="#3498db", edgecolor="black", linewidth=0.7, alpha=0.85)
    bars2 = ax.bar(x, p4_norm[merged.index], w, label="MEGADOCK Score (60%)",
                   color="#e74c3c", edgecolor="black", linewidth=0.7, alpha=0.85)
    bars3 = ax.bar(x + w, merged["Combined_Score"], w, label="Combined Score",
                   color="#27ae60", edgecolor="black", linewidth=0.7, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(merged["Candidate"], fontsize=10)
    ax.set_ylabel("Normalized Score", fontsize=11)
    ax.set_title("Combined Phase 3 + Phase 4 Ranking\n(Design Quality 40% + Docking Score 60%)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

    # Rank labels
    for i, (_, row) in enumerate(merged.iterrows()):
        ax.text(i + w, row["Combined_Score"] + 0.01,
                f"#{int(row['Combined_Rank'])}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#27ae60")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig30_combined_ranking.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("  Saved Fig30_combined_ranking.png")

    # Save combined ranking CSV
    combined_csv = DOCK_DIR / "phase4_combined_ranking.csv"
    merged[["Candidate", "MEGADOCK_Score", "PPI_Escore", "N_Interface_Res",
            "Overall_Score", "Combined_Score", "Combined_Rank"]].to_csv(combined_csv, index=False)
    log(f"  Combined ranking CSV: {combined_csv.name}")

# ─── Step 10: Write Phase 4 Report ───────────────────────────────────────────
log("\nWriting Phase 4 report...")
report_file = DOCK_DIR / "PHASE4_MEGADOCK_REPORT.txt"
with open(report_file, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("PHASE 4 REPORT — MEGADOCK-GPU Protein-Protein Docking\n")
    f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 70 + "\n\n")

    f.write("1. DOCKING SETUP\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Receptor:       LasR LBD (2UV0, chain E, 163 residues)\n")
    f.write(f"  Method:         MEGADOCK-GPU (FFT-based rigid-body docking)\n")
    f.write(f"  Rotations:      {N_ROTATIONS} (full spherical sampling)\n")
    f.write(f"  GPU:            NVIDIA RTX 5060 Ti (CUDA)\n")
    f.write(f"  Ligands:        {len(CANDIDATES)} nanobody candidates (ESMFold structures)\n\n")

    f.write("2. DOCKING RESULTS\n")
    f.write("-" * 40 + "\n")
    f.write(f"{'Candidate':<14} {'MEGADOCK':>10} {'PPI-E':>8} {'Interface':>10} {'BSA(Å²)':>10} {'Time(s)':>8}\n")
    f.write(f"{'':14} {'Score':>10} {'Score':>8} {'Residues':>10} {'(approx)':>10}\n")
    f.write("-" * 62 + "\n")
    for _, row in df.iterrows():
        f.write(f"{row['Candidate']:<14} {row['MEGADOCK_Score']:>10.4f} "
                f"{row['PPI_Escore']:>8.3f} {row['N_Interface_Res']:>10.0f} "
                f"{row['BSA_approx_A2']:>10.0f} {row['Runtime_s']:>8.1f}\n")

    f.write("\n3. PPI E-SCORE INTERPRETATION\n")
    f.write("-" * 40 + "\n")
    for _, row in df.iterrows():
        e = row["PPI_Escore"]
        if e >= 12:
            interp = "PPI POSITIVE (PPV ~80%)"
        elif e >= 10:
            interp = "PPI POSITIVE (PPV ~50%)"
        elif e >= 8:
            interp = "PPI POSITIVE (PPV ~10%)"
        else:
            interp = "PPI Negative (<8)"
        f.write(f"  {row['Candidate']}: E={e:.3f} → {interp}\n")

    f.write("\n4. TOP CANDIDATE\n")
    f.write("-" * 40 + "\n")
    best = df.iloc[0]
    f.write(f"  {best['Candidate']} — MEGADOCK Score: {best['MEGADOCK_Score']:.4f}\n")
    f.write(f"  PPI E-score: {best['PPI_Escore']:.3f}\n")
    f.write(f"  Interface residues: {int(best['N_Interface_Res'])}\n")
    f.write(f"  Approx BSA: {best['BSA_approx_A2']:.0f} Å²\n")

    if results[best["Candidate"]].get("interface_residues"):
        f.write(f"  Key interface residues (receptor):\n")
        for res in results[best["Candidate"]]["interface_residues"][:10]:
            f.write(f"    {res}\n")

    f.write("\n5. OUTPUT FILES\n")
    f.write("-" * 40 + "\n")
    f.write(f"  JSON:  phase4_megadock_results.json\n")
    f.write(f"  CSV:   phase4_megadock_ranking.csv\n")
    f.write(f"  CSV:   phase4_combined_ranking.csv\n")
    f.write(f"  PDB:   NbLasR-X_MEGADOCK_complex.pdb (for each candidate)\n")
    f.write(f"  Figs:  Fig25–Fig30 in 08_Results_Figures/Phase4/\n")
    f.write(f"  Logs:  {LOG_FILE.name}\n")
    f.write(f"  Runs:  04_Docking/megadock_runs/NbLasR-X/ (dock.out, poses, ppiscore)\n")

    f.write("\n6. NEXT STEP → PHASE 5: Molecular Dynamics Simulation\n")
    f.write("   OpenMM GPU-accelerated MD on best docked complex\n")
    f.write("=" * 70 + "\n")

log(f"\nPhase 4 Report: {report_file}")
log("\n✓ Phase 4 MEGADOCK-GPU docking COMPLETE")
log(f"  Log file: {LOG_FILE}")
