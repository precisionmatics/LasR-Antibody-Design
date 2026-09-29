"""
PHASE 6 — Binding Affinity: Single-Trajectory MM-GBSA
NbLasR-2 : LasR LBD complex

  ΔG_bind ≈ <E_MM+GB+SA(complex)> - <E_MM+GB+SA(LasR)> - <E_MM+GB+SA(NbLasR-2)>

Chain assignment (from PDBFixer rename of MEGADOCK complex):
  chainid 0 → solvated chain A → LasR LBD    (159 residues, orig chain E)
  chainid 1 → solvated chain B → NbLasR-2    (106 residues, orig chain A)

Force field : AMBER ff14SB + OBC2 implicit solvent (NoCutoff, no water)
Frames      : last 50 ns, Δt = 100 ps → 500 frames
Environment : unomd (mdtraj 1.10.3, openmm 8.0, parmed 4.3.0)
"""

import os, sys, time, json, logging, warnings
import numpy as np
warnings.filterwarnings("ignore")

BASE   = "/home/stalin/Desktop/LasR_Antibody_Design"
MDDIR  = f"{BASE}/05_MD_Simulation/NbLasR-2"
OUTDIR = f"{BASE}/06_Binding_Affinity"
FIGDIR = f"{OUTDIR}/figures"
LOGFILE = f"{BASE}/logs/phase6_mmgbsa_{time.strftime('%Y%m%d_%H%M%S')}.log"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(f"{BASE}/logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOGFILE), logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ── Parameters ─────────────────────────────────────────────────────────────────
DCD_PATH   = f"{MDDIR}/production/trajectory.dcd"
TOP_PATH   = f"{MDDIR}/prep/system_solvated.pdb"
N_EQUIL    = 5000   # skip first 5000 frames (0–50 ns equilibration)
STRIDE     = 10     # take every 10th frame → 500 frames over 50–100 ns
TEMP_K     = 300.0
CHAIN_LASR = 0      # mdtraj chainid 0 = LasR LBD
CHAIN_NB   = 1      # mdtraj chainid 1 = NbLasR-2
PROBE_R    = 0.14   # nm, solvent probe radius for SASA (1.4 Å)

log.info("=" * 60)
log.info("PHASE 6 — MM-GBSA Binding Affinity  (NbLasR-2 : LasR LBD)")
log.info("=" * 60)

# ── STEP 1: Load trajectory (protein only, last 50 ns) ─────────────────────────
log.info("STEP 1: Loading trajectory (protein only, last 50 ns)...")
import mdtraj as md

# Build protein atom index from topology PDB
top_ref = md.load(TOP_PATH)
prot_idx = top_ref.topology.select("protein")
del top_ref
log.info(f"  Protein atom indices: {len(prot_idx)} atoms")

# Iterload — only protein atoms, skip first 50 ns, stride 100 ps
chunks = []
for chunk in md.iterload(DCD_PATH, top=TOP_PATH, chunk=500,
                          skip=N_EQUIL, stride=STRIDE,
                          atom_indices=prot_idx):
    chunks.append(chunk)
    log.info(f"  Loaded chunk: {chunk.n_frames} frames")

traj = md.join(chunks)
del chunks
log.info(f"  Total: {traj.n_frames} frames | {traj.n_atoms} protein atoms")

# Verify chains
for c in traj.topology.chains:
    resnames = [r.name for r in c.residues]
    log.info(f"  chainid {c.index}: {c.n_residues} residues, "
             f"{c.n_atoms} atoms | {resnames[0]}..{resnames[-1]}")

lasr_idx = traj.topology.select(f"chainid {CHAIN_LASR}")
nb_idx   = traj.topology.select(f"chainid {CHAIN_NB}")
log.info(f"  LasR LBD   (chainid {CHAIN_LASR}): {len(lasr_idx)} atoms")
log.info(f"  NbLasR-2   (chainid {CHAIN_NB}):   {len(nb_idx)} atoms")

traj_lasr = traj.atom_slice(lasr_idx)
traj_nb   = traj.atom_slice(nb_idx)

# ── STEP 2: Save reference PDB files for OpenMM topology ──────────────────────
log.info("STEP 2: Saving reference PDB files...")
ref_i = traj.n_frames // 2   # middle frame as reference
traj[ref_i].save_pdb(f"{OUTDIR}/complex_ref.pdb")
traj_lasr[ref_i].save_pdb(f"{OUTDIR}/lasr_ref.pdb")
traj_nb[ref_i].save_pdb(f"{OUTDIR}/nb_ref.pdb")
log.info("  Saved: complex_ref.pdb, lasr_ref.pdb, nb_ref.pdb")

# ── STEP 3: Build OpenMM MM-GBSA contexts ──────────────────────────────────────
log.info("STEP 3: Building OpenMM OBC2 contexts...")
import openmm as mm
import openmm.app as app
import openmm.unit as unit

for i in range(mm.Platform.getNumPlatforms()):
    log.info(f"  Platform {i}: {mm.Platform.getPlatform(i).getName()}")

platform = mm.Platform.getPlatformByName("CPU")

def build_context(pdb_path, label):
    pdb = app.PDBFile(pdb_path)
    ff  = app.ForceField("amber14/protein.ff14SB.xml", "implicit/obc2.xml")
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        hydrogenMass=None,
        soluteDielectric=1.0,
        solventDielectric=78.5,
    )
    integrator = mm.VerletIntegrator(0.001 * unit.picosecond)
    ctx = mm.Context(system, integrator, platform)
    log.info(f"  {label}: {system.getNumParticles()} particles | OBC2")
    return ctx

ctx_cmplx = build_context(f"{OUTDIR}/complex_ref.pdb", "Complex  ")
ctx_lasr  = build_context(f"{OUTDIR}/lasr_ref.pdb",    "LasR LBD ")
ctx_nb    = build_context(f"{OUTDIR}/nb_ref.pdb",      "NbLasR-2 ")

def energy_kcal(ctx, pos_nm):
    ctx.setPositions(pos_nm * unit.nanometer)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilocalories_per_mole)

# ── STEP 4: Per-frame MM-GBSA energies ─────────────────────────────────────────
log.info("STEP 4: Computing MM-GBSA energies per frame...")
n = traj.n_frames
e_cmplx = np.zeros(n)
e_lasr  = np.zeros(n)
e_nb    = np.zeros(n)

t0 = time.time()
for i in range(n):
    e_cmplx[i] = energy_kcal(ctx_cmplx, traj.xyz[i])
    e_lasr[i]  = energy_kcal(ctx_lasr,  traj_lasr.xyz[i])
    e_nb[i]    = energy_kcal(ctx_nb,    traj_nb.xyz[i])
    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        eta = (n - i - 1) / ((i + 1) / elapsed)
        log.info(f"  {i+1}/{n} frames | {(i+1)/elapsed:.1f} fr/s | ETA {eta:.0f}s")

dg = e_cmplx - e_lasr - e_nb

# ── STEP 5: SASA interface burial ──────────────────────────────────────────────
log.info("STEP 5: Computing SASA for interface burial analysis...")
sasa_cmplx = md.shrake_rupley(traj,      probe_radius=PROBE_R, mode="atom")   # (n, n_atoms)
sasa_lasr  = md.shrake_rupley(traj_lasr, probe_radius=PROBE_R, mode="atom")   # (n, n_lasr)
sasa_nb    = md.shrake_rupley(traj_nb,   probe_radius=PROBE_R, mode="atom")   # (n, n_nb)

nm2_to_A2 = 100.0
sasa_c_A2 = sasa_cmplx.sum(axis=1) * nm2_to_A2
sasa_l_A2 = sasa_lasr.sum(axis=1)  * nm2_to_A2
sasa_n_A2 = sasa_nb.sum(axis=1)    * nm2_to_A2
d_sasa    = sasa_c_A2 - sasa_l_A2 - sasa_n_A2   # negative = buried

log.info(f"  Mean ΔSASA = {d_sasa.mean():.1f} ± {d_sasa.std():.1f} Å²")

# ── STEP 6: Statistics ──────────────────────────────────────────────────────────
log.info("STEP 6: Computing statistics...")
dg_mean = dg.mean()
dg_std  = dg.std()
dg_sem  = dg_std / np.sqrt(n)

rng = np.random.default_rng(42)
boot_means = np.array([rng.choice(dg, size=n, replace=True).mean() for _ in range(2000)])
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

# Block averaging (10 blocks)
n_blk   = 10
blk_sz  = n // n_blk
blk_m   = [dg[i*blk_sz:(i+1)*blk_sz].mean() for i in range(n_blk)]
blk_sem = np.std(blk_m) / np.sqrt(n_blk)

cumul_mean = np.cumsum(dg) / np.arange(1, n + 1)

log.info("=" * 60)
log.info(f"  ΔG_bind (MM-GBSA) = {dg_mean:.2f} ± {dg_std:.2f} kcal/mol")
log.info(f"  SEM               = {dg_sem:.2f} kcal/mol")
log.info(f"  95% CI            = [{ci_lo:.2f}, {ci_hi:.2f}] kcal/mol")
log.info(f"  Block-avg SEM     = {blk_sem:.2f} kcal/mol")
log.info(f"  Mean ΔSASA        = {d_sasa.mean():.0f} Å²")
log.info("=" * 60)

# ── STEP 7: Per-residue SASA burial (hot-spots) ────────────────────────────────
log.info("STEP 7: Per-residue SASA burial analysis...")

# NbLasR-2 side: sasa_cmplx indexed via nb_idx vs sasa_nb in isolated space
sasa_nb_in_cmplx_mean = sasa_cmplx[:, nb_idx].mean(axis=0) * nm2_to_A2  # (n_nb,)
sasa_nb_isolated_mean = sasa_nb.mean(axis=0)               * nm2_to_A2  # (n_nb,)

# LasR LBD side
sasa_l_in_cmplx_mean  = sasa_cmplx[:, lasr_idx].mean(axis=0) * nm2_to_A2
sasa_l_isolated_mean  = sasa_lasr.mean(axis=0)                * nm2_to_A2

hotspots = []

for res in traj_nb.topology.residues:
    aidx = [a.index for a in res.atoms]
    burial = (sasa_nb_isolated_mean[aidx] - sasa_nb_in_cmplx_mean[aidx]).sum()
    if burial > 0.5:  # only interface residues
        hotspots.append({"resid": res.resSeq, "resname": res.name,
                         "chain": "NbLasR-2", "burial_A2": float(burial)})

for res in traj_lasr.topology.residues:
    aidx = [a.index for a in res.atoms]
    burial = (sasa_l_isolated_mean[aidx] - sasa_l_in_cmplx_mean[aidx]).sum()
    if burial > 0.5:
        hotspots.append({"resid": res.resSeq, "resname": res.name,
                         "chain": "LasR", "burial_A2": float(burial)})

hotspots.sort(key=lambda x: x["burial_A2"], reverse=True)
log.info(f"  Interface residues: {len(hotspots)}")
log.info("  Top 10 hot-spots by SASA burial:")
for r in hotspots[:10]:
    log.info(f"    {r['chain']:8s}  {r['resname']}{r['resid']:4d}  {r['burial_A2']:.1f} Å²")

# ── STEP 8: Save results ────────────────────────────────────────────────────────
log.info("STEP 8: Saving results...")

results = {
    "system":             "NbLasR-2 : LasR LBD",
    "method":             "ST-MM-GBSA | AMBER ff14SB | OBC2 | NoCutoff",
    "n_frames":           int(n),
    "time_window_ns":     "50-100",
    "dg_bind_kcal_mol":   float(dg_mean),
    "dg_std_kcal_mol":    float(dg_std),
    "dg_sem_kcal_mol":    float(dg_sem),
    "ci_95_lo":           float(ci_lo),
    "ci_95_hi":           float(ci_hi),
    "block_sem_kcal_mol": float(blk_sem),
    "delta_sasa_A2_mean": float(d_sasa.mean()),
    "delta_sasa_A2_std":  float(d_sasa.std()),
    "e_complex_mean":     float(e_cmplx.mean()),
    "e_lasr_mean":        float(e_lasr.mean()),
    "e_nb_mean":          float(e_nb.mean()),
    "hotspots":           hotspots,
}

with open(f"{OUTDIR}/mmgbsa_results.json", "w") as f:
    json.dump(results, f, indent=2)

np.save(f"{OUTDIR}/dg_per_frame.npy",    dg)
np.save(f"{OUTDIR}/delta_sasa.npy",      d_sasa)
np.save(f"{OUTDIR}/cumul_mean_dg.npy",   cumul_mean)
log.info(f"  Saved: mmgbsa_results.json + .npy arrays")

# ── STEP 9: Figures ─────────────────────────────────────────────────────────────
log.info("STEP 9: Generating figures...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":    11,
    "axes.linewidth": 1.2,
    "figure.dpi":   150,
})

time_axis = np.linspace(50, 100, n)   # ns
CLR_BLU = "#2196F3"
CLR_RED = "#F44336"
CLR_GRN = "#4CAF50"
CLR_PUR = "#9C27B0"
CLR_ORG = "#FF9800"

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)

# — Panel 1: ΔG time series —
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(time_axis, dg, color=CLR_BLU, lw=0.6, alpha=0.55, label="Per-frame MM-GBSA")
ax1.axhline(dg_mean, color=CLR_RED, lw=2.2,
            label=f"Mean = {dg_mean:.1f} kcal/mol")
ax1.fill_between(time_axis, ci_lo, ci_hi, alpha=0.15, color=CLR_RED,
                 label=f"95% CI [{ci_lo:.1f}, {ci_hi:.1f}]")
ax1.set_xlabel("Simulation Time (ns)")
ax1.set_ylabel("ΔG$_{bind}$ (kcal/mol)")
ax1.set_title("MM-GBSA Binding Free Energy — NbLasR-2 : LasR LBD", fontweight="bold")
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(True, alpha=0.3)

info_txt = (f"ΔG$_{{bind}}$ = {dg_mean:.1f} ± {dg_std:.1f} kcal/mol\n"
            f"95% CI: [{ci_lo:.1f}, {ci_hi:.1f}] kcal/mol\n"
            f"Method: ST-MM-GBSA | OBC2 | AMBER ff14SB\n"
            f"Frames: {n} (50–100 ns, Δt = 100 ps)")
ax1.text(0.02, 0.04, info_txt, transform=ax1.transAxes, fontsize=8.5,
         va="bottom", bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.85))

# — Panel 2: KDE distribution —
ax2 = fig.add_subplot(gs[0, 2])
kde = gaussian_kde(dg, bw_method=0.3)
xr  = np.linspace(dg.min() - 10, dg.max() + 10, 400)
ax2.fill_between(xr, kde(xr), alpha=0.35, color=CLR_BLU)
ax2.plot(xr, kde(xr), color="#1565C0", lw=2)
ax2.axvline(dg_mean, color=CLR_RED, lw=2, ls="--", label=f"μ = {dg_mean:.1f}")
ax2.axvline(ci_lo,  color=CLR_ORG, lw=1.5, ls=":", label="95% CI")
ax2.axvline(ci_hi,  color=CLR_ORG, lw=1.5, ls=":")
ax2.set_xlabel("ΔG$_{bind}$ (kcal/mol)")
ax2.set_ylabel("Density")
ax2.set_title("Distribution of ΔG$_{bind}$", fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# — Panel 3: Convergence —
ax3 = fig.add_subplot(gs[1, :2])
ax3.plot(time_axis, cumul_mean, color=CLR_GRN, lw=2.5, label="Cumulative mean")
ax3.axhline(dg_mean, color=CLR_RED, lw=1.5, ls="--", alpha=0.7,
            label=f"Final mean {dg_mean:.1f} kcal/mol")
ax3.fill_between(time_axis, dg_mean - blk_sem, dg_mean + blk_sem,
                 alpha=0.2, color=CLR_RED, label=f"±SEM$_{{block}}$ {blk_sem:.1f}")
ax3.set_xlabel("Simulation Time (ns)")
ax3.set_ylabel("Cumulative ΔG$_{bind}$ (kcal/mol)")
ax3.set_title("Convergence of MM-GBSA", fontweight="bold")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# — Panel 4: ΔSASA time series —
ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(time_axis, d_sasa, color=CLR_PUR, lw=0.7, alpha=0.55)
ax4.axhline(d_sasa.mean(), color="#4A148C", lw=2, ls="--",
            label=f"Mean {d_sasa.mean():.0f} Å²")
ax4.set_xlabel("Time (ns)")
ax4.set_ylabel("ΔSASA (Å²)")
ax4.set_title("Interface Burial (ΔSASA)", fontweight="bold")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# — Panel 5: Hot-spot bar chart —
ax5 = fig.add_subplot(gs[2, :])
top_hs  = hotspots[:20]
colors5 = [CLR_BLU if r["chain"] == "NbLasR-2" else CLR_GRN for r in top_hs]
labels5 = [f"{r['resname']}{r['resid']}\n({r['chain'][:2]})" for r in top_hs]
vals5   = [r["burial_A2"] for r in top_hs]

ax5.bar(range(len(top_hs)), vals5, color=colors5, edgecolor="white", linewidth=0.5)
ax5.set_xticks(range(len(top_hs)))
ax5.set_xticklabels(labels5, fontsize=8.5)
ax5.set_ylabel("SASA Burial (Å²)")
ax5.set_title("Top Interface Hot-Spot Residues (SASA Burial)", fontweight="bold")
ax5.grid(True, alpha=0.3, axis="y")
ax5.legend(handles=[Patch(facecolor=CLR_BLU, label="NbLasR-2"),
                     Patch(facecolor=CLR_GRN, label="LasR LBD")],
           loc="upper right")

plt.suptitle("Phase 6 — Binding Affinity: NbLasR-2 Nanobody vs. LasR LBD (MM-GBSA)",
             fontsize=14, fontweight="bold", y=1.01)

for fmt in ("png", "pdf"):
    fig.savefig(f"{FIGDIR}/mmgbsa_binding_affinity.{fmt}",
                dpi=200, bbox_inches="tight")
plt.close()
log.info(f"  Figure saved: {FIGDIR}/mmgbsa_binding_affinity.png/pdf")

# ── DONE ────────────────────────────────────────────────────────────────────────
elapsed_total = time.time() - t0
log.info("=" * 60)
log.info("PHASE 6 COMPLETE")
log.info(f"  ΔG_bind = {dg_mean:.2f} ± {dg_std:.2f} kcal/mol")
log.info(f"  95% CI  = [{ci_lo:.2f}, {ci_hi:.2f}] kcal/mol")
log.info(f"  ΔSASA   = {d_sasa.mean():.0f} Å²  (interface burial)")
log.info(f"  Output  : {OUTDIR}/")
log.info(f"  Total time: {elapsed_total/60:.1f} min")
log.info("=" * 60)
