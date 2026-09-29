"""
Parameterized ST-MM-GBSA for any NbLasR candidate.
Usage:  python phase6_mmgbsa_multicand.py NbLasR-1

Identical protocol to phase6_binding_affinity.py (NbLasR-2):
  AMBER ff14SB + OBC2 implicit solvent, last 50 ns, 500 frames (Δt=100 ps).
Results saved to 06_Binding_Affinity/<candidate>/
"""

import os, sys, time, json, logging, warnings
import numpy as np
warnings.filterwarnings("ignore")

if len(sys.argv) < 2:
    print("Usage: python phase6_mmgbsa_multicand.py <candidate>")
    sys.exit(1)

CANDIDATE = sys.argv[1]

BASE    = "/home/stalin/Desktop/LasR_Antibody_Design"
MDDIR   = f"{BASE}/05_MD_Simulation/{CANDIDATE}"
OUTDIR  = f"{BASE}/06_Binding_Affinity/{CANDIDATE}"
FIGDIR  = f"{OUTDIR}/figures"
LOGFILE = f"{BASE}/logs/phase6_mmgbsa_{CANDIDATE}_{time.strftime('%Y%m%d_%H%M%S')}.log"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(f"{BASE}/logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOGFILE), logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

DCD_PATH   = f"{MDDIR}/production/trajectory.dcd"
TOP_PATH   = f"{MDDIR}/prep/system_solvated.pdb"
N_EQUIL    = 5000    # skip first 50 ns (5000 frames × 10 ps)
STRIDE     = 10      # every 10th frame → 500 frames over 50–100 ns
TEMP_K     = 300.0
CHAIN_LASR = 0       # chainid 0 = LasR LBD (chain E appears first in all complexes)
CHAIN_NB   = 1       # chainid 1 = nanobody (chain A appears second)
PROBE_R    = 0.14

log.info("=" * 60)
log.info(f"PHASE 6 — MM-GBSA   {CANDIDATE} : LasR LBD")
log.info("=" * 60)

import mdtraj as md

top_ref  = md.load(TOP_PATH)
prot_idx = top_ref.topology.select("protein")
del top_ref
log.info(f"  Protein atoms: {len(prot_idx)}")

chunks = []
for chunk in md.iterload(DCD_PATH, top=TOP_PATH, chunk=500,
                          skip=N_EQUIL, stride=STRIDE,
                          atom_indices=prot_idx):
    chunks.append(chunk)
    log.info(f"  Loaded chunk: {chunk.n_frames} frames")

traj = md.join(chunks)
del chunks
log.info(f"  Total: {traj.n_frames} frames | {traj.n_atoms} protein atoms")

for c in traj.topology.chains:
    resnames = [r.name for r in c.residues]
    log.info(f"  chainid {c.index}: {c.n_residues} res | "
             f"{resnames[0]}..{resnames[-1]}")

lasr_idx  = traj.topology.select(f"chainid {CHAIN_LASR}")
nb_idx    = traj.topology.select(f"chainid {CHAIN_NB}")
traj_lasr = traj.atom_slice(lasr_idx)
traj_nb   = traj.atom_slice(nb_idx)
log.info(f"  LasR LBD atoms: {len(lasr_idx)} | {CANDIDATE} atoms: {len(nb_idx)}")

# Save reference PDB files
log.info("STEP 2: Saving reference PDB files...")
ref_i = traj.n_frames // 2
traj[ref_i].save_pdb(f"{OUTDIR}/complex_ref.pdb")
traj_lasr[ref_i].save_pdb(f"{OUTDIR}/lasr_ref.pdb")
traj_nb[ref_i].save_pdb(f"{OUTDIR}/nb_ref.pdb")

# Build OBC2 contexts
log.info("STEP 3: Building OBC2 contexts...")
import openmm as mm
import openmm.app as app
import openmm.unit as unit

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

ctx_cmplx = build_context(f"{OUTDIR}/complex_ref.pdb", "Complex")
ctx_lasr  = build_context(f"{OUTDIR}/lasr_ref.pdb",    "LasR   ")
ctx_nb    = build_context(f"{OUTDIR}/nb_ref.pdb",      CANDIDATE)

def energy_kcal(ctx, pos_nm):
    ctx.setPositions(pos_nm * unit.nanometer)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilocalories_per_mole)

# Per-frame MM-GBSA
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
        log.info(f"  {i+1}/{n} | {(i+1)/elapsed:.1f} fr/s | ETA {eta:.0f}s")

dg = e_cmplx - e_lasr - e_nb

# SASA interface burial
log.info("STEP 5: SASA interface burial analysis...")
sasa_cmplx = md.shrake_rupley(traj,      probe_radius=PROBE_R, mode="atom")
sasa_lasr  = md.shrake_rupley(traj_lasr, probe_radius=PROBE_R, mode="atom")
sasa_nb    = md.shrake_rupley(traj_nb,   probe_radius=PROBE_R, mode="atom")
nm2_A2     = 100.0
d_sasa     = (sasa_cmplx.sum(axis=1) - sasa_lasr.sum(axis=1) - sasa_nb.sum(axis=1)) * nm2_A2
log.info(f"  Mean ΔSASA = {d_sasa.mean():.1f} ± {d_sasa.std():.1f} Å²")

# Statistics
log.info("STEP 6: Statistics...")
dg_mean = dg.mean()
dg_std  = dg.std()
dg_sem  = dg_std / np.sqrt(n)
rng     = np.random.default_rng(42)
boots   = np.array([rng.choice(dg, size=n, replace=True).mean() for _ in range(2000)])
ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
n_blk   = 10
blk_sz  = n // n_blk
blk_m   = [dg[i*blk_sz:(i+1)*blk_sz].mean() for i in range(n_blk)]
blk_sem = np.std(blk_m) / np.sqrt(n_blk)
cumul   = np.cumsum(dg) / np.arange(1, n + 1)

log.info("=" * 60)
log.info(f"  ΔG_bind = {dg_mean:.2f} ± {dg_std:.2f} kcal/mol")
log.info(f"  SEM     = {dg_sem:.2f}  |  Block-SEM = {blk_sem:.2f} kcal/mol")
log.info(f"  95% CI  = [{ci_lo:.2f}, {ci_hi:.2f}] kcal/mol")
log.info(f"  ΔSASA   = {d_sasa.mean():.0f} Å²")
log.info("=" * 60)

# Hot-spots
log.info("STEP 7: Per-residue SASA hot-spots...")
sasa_nb_cmplx = sasa_cmplx[:, nb_idx].mean(axis=0) * nm2_A2
sasa_nb_iso   = sasa_nb.mean(axis=0) * nm2_A2
sasa_l_cmplx  = sasa_cmplx[:, lasr_idx].mean(axis=0) * nm2_A2
sasa_l_iso    = sasa_lasr.mean(axis=0) * nm2_A2

hotspots = []
for res in traj_nb.topology.residues:
    aidx   = [a.index for a in res.atoms]
    burial = (sasa_nb_iso[aidx] - sasa_nb_cmplx[aidx]).sum()
    if burial > 0.5:
        hotspots.append({"resid": res.resSeq, "resname": res.name,
                         "chain": CANDIDATE, "burial_A2": float(burial)})
for res in traj_lasr.topology.residues:
    aidx   = [a.index for a in res.atoms]
    burial = (sasa_l_iso[aidx] - sasa_l_cmplx[aidx]).sum()
    if burial > 0.5:
        hotspots.append({"resid": res.resSeq, "resname": res.name,
                         "chain": "LasR", "burial_A2": float(burial)})
hotspots.sort(key=lambda x: x["burial_A2"], reverse=True)
log.info(f"  Interface residues: {len(hotspots)}")
for r in hotspots[:10]:
    log.info(f"    {r['chain']:10s}  {r['resname']}{r['resid']:4d}  {r['burial_A2']:.1f} Å²")

# Save results
results = {
    "candidate":            CANDIDATE,
    "system":               f"{CANDIDATE} : LasR LBD",
    "method":               "ST-MM-GBSA | AMBER ff14SB | OBC2 | NoCutoff",
    "n_frames":             int(n),
    "time_window_ns":       "50-100",
    "dg_bind_kcal_mol":     float(dg_mean),
    "dg_std_kcal_mol":      float(dg_std),
    "dg_sem_kcal_mol":      float(dg_sem),
    "ci_95_lo":             float(ci_lo),
    "ci_95_hi":             float(ci_hi),
    "block_sem_kcal_mol":   float(blk_sem),
    "delta_sasa_A2_mean":   float(d_sasa.mean()),
    "delta_sasa_A2_std":    float(d_sasa.std()),
    "e_complex_mean":       float(e_cmplx.mean()),
    "e_lasr_mean":          float(e_lasr.mean()),
    "e_nb_mean":            float(e_nb.mean()),
    "hotspots":             hotspots,
}
with open(f"{OUTDIR}/mmgbsa_results.json", "w") as f:
    json.dump(results, f, indent=2)
np.save(f"{OUTDIR}/dg_per_frame.npy",  dg)
np.save(f"{OUTDIR}/delta_sasa.npy",    d_sasa)
np.save(f"{OUTDIR}/cumul_mean_dg.npy", cumul)
log.info(f"  Saved → {OUTDIR}/mmgbsa_results.json")

# Figure: ΔG time series + distribution
log.info("STEP 8: Figures...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.linewidth": 1.2})

time_axis = np.linspace(50, 100, n)
fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(time_axis, dg, color="#2196F3", lw=0.6, alpha=0.55, label="Per-frame MM-GBSA")
ax1.axhline(dg_mean, color="#F44336", lw=2.2,
            label=f"Mean = {dg_mean:.1f} kcal/mol")
ax1.fill_between(time_axis, ci_lo, ci_hi, alpha=0.15, color="#F44336",
                 label=f"95% CI [{ci_lo:.1f}, {ci_hi:.1f}]")
ax1.set_xlabel("Simulation Time (ns)", fontsize=12)
ax1.set_ylabel("ΔG$_{bind}$ (kcal/mol)", fontsize=12)
ax1.set_title(f"MM-GBSA Binding Free Energy — {CANDIDATE} : LasR LBD", fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 2])
kde = gaussian_kde(dg, bw_method=0.3)
xr  = np.linspace(dg.min() - 10, dg.max() + 10, 400)
ax2.fill_between(xr, kde(xr), alpha=0.35, color="#2196F3")
ax2.plot(xr, kde(xr), color="#1565C0", lw=2)
ax2.axvline(dg_mean, color="#F44336", lw=2, ls="--", label=f"μ = {dg_mean:.1f}")
ax2.set_xlabel("ΔG$_{bind}$ (kcal/mol)", fontsize=12)
ax2.set_ylabel("Density", fontsize=12)
ax2.set_title("ΔG Distribution", fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, :2])
ax3.plot(time_axis, cumul, color="#4CAF50", lw=2.5, label="Cumulative mean")
ax3.axhline(dg_mean, color="#F44336", lw=1.5, ls="--", alpha=0.7,
            label=f"Final {dg_mean:.1f} kcal/mol")
ax3.fill_between(time_axis, dg_mean - blk_sem, dg_mean + blk_sem,
                 alpha=0.2, color="#F44336", label=f"±SEM$_{{block}}$")
ax3.set_xlabel("Simulation Time (ns)", fontsize=12)
ax3.set_ylabel("Cumulative ΔG$_{bind}$ (kcal/mol)", fontsize=12)
ax3.set_title("Convergence of MM-GBSA", fontweight="bold")
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(time_axis, d_sasa, color="#9C27B0", lw=0.7, alpha=0.55)
ax4.axhline(d_sasa.mean(), color="#4A148C", lw=2, ls="--",
            label=f"Mean {d_sasa.mean():.0f} Å²")
ax4.set_xlabel("Time (ns)", fontsize=12)
ax4.set_ylabel("ΔSASA (Å²)", fontsize=12)
ax4.set_title("Interface Burial (ΔSASA)", fontweight="bold")
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.suptitle(f"Phase 6 — MM-GBSA: {CANDIDATE} vs. LasR LBD",
             fontsize=14, fontweight="bold", y=1.01)
for fmt in ("png", "pdf"):
    fig.savefig(f"{FIGDIR}/mmgbsa_binding_affinity.{fmt}",
                dpi=200, bbox_inches="tight")
plt.close()
log.info(f"  Figure saved → {FIGDIR}/mmgbsa_binding_affinity.png")

log.info("=" * 60)
log.info(f"  PHASE 6 COMPLETE — {CANDIDATE}")
log.info(f"  ΔG_bind = {dg_mean:.2f} ± {dg_std:.2f} kcal/mol")
log.info(f"  95% CI  = [{ci_lo:.2f}, {ci_hi:.2f}] kcal/mol")
log.info(f"  ΔSASA   = {d_sasa.mean():.0f} Å²")
log.info("=" * 60)
