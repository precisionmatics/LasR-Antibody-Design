"""
Standalone analysis for NbLasR-5 — re-runs STEP 7 only.
Trajectory is already complete; previous run crashed on OOM.
Uses stride=5 (2000 frames) to stay within RAM.
"""
import os, sys, json, logging
import numpy as np

CANDIDATE = "NbLasR-5"
BASE    = "/home/stalin/Desktop/LasR_Antibody_Design"
MDDIR   = f"{BASE}/05_MD_Simulation/{CANDIDATE}"
FIGDIR  = f"{MDDIR}/figures"
os.makedirs(f"{MDDIR}/analysis", exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

import mdtraj as md

prep_pdb = f"{MDDIR}/prep/system_solvated.pdb"
traj_dcd = f"{MDDIR}/production/trajectory.dcd"

log.info("Loading trajectory with stride=5...")
traj = md.load_dcd(traj_dcd, top=prep_pdb, stride=5)
log.info(f"  {traj.n_frames} frames, {traj.n_atoms} atoms, {traj.time[-1]/1000:.1f} ns")

# Fix PBC wrapping — keeps both chains in the same periodic image before RMSD
traj.image_molecules(inplace=True)

prot_idx  = traj.topology.select("protein")
traj_prot = traj.atom_slice(prot_idx)
bb_idx    = traj_prot.topology.select("backbone")
traj_prot.superpose(traj_prot, 0, atom_indices=bb_idx)

rmsd_all  = md.rmsd(traj_prot, traj_prot, 0, atom_indices=bb_idx) * 10
lasr_idx  = traj_prot.topology.select("chainid 0 and backbone")
nb_idx    = traj_prot.topology.select("chainid 1 and backbone")
rmsd_lasr = md.rmsd(traj_prot, traj_prot, 0, atom_indices=lasr_idx) * 10
rmsd_nb   = md.rmsd(traj_prot, traj_prot, 0, atom_indices=nb_idx)   * 10
time_ns   = traj.time / 1000

ca_idx  = traj_prot.topology.select("name CA")
rmsf_ca = md.rmsf(traj_prot, traj_prot, atom_indices=ca_idx) * 10
rg      = md.compute_rg(traj_prot) * 10
log.info("Computing H-bonds...")
hbonds  = md.baker_hubbard(traj_prot, freq=0.3, periodic=False)
log.info("Computing SASA...")
sasa    = md.shrake_rupley(traj_prot).sum(axis=1) * 100
res_ids = [r.resSeq for r in traj_prot.topology.residues][:len(rmsf_ca)]

log.info(f"  RMSD complex : {rmsd_all.mean():.2f} +/- {rmsd_all.std():.2f} A")
log.info(f"  RMSD LasR    : {rmsd_lasr.mean():.2f} +/- {rmsd_lasr.std():.2f} A")
log.info(f"  RMSD Nb      : {rmsd_nb.mean():.2f} +/- {rmsd_nb.std():.2f} A")
log.info(f"  Rg           : {rg.mean():.2f} +/- {rg.std():.2f} A")
log.info(f"  H-bonds >30% : {len(hbonds)}")

results = {
    "candidate":            CANDIDATE,
    "n_frames":             int(traj.n_frames),
    "sim_time_ns":          float(traj.time[-1] / 1000),
    "rmsd_complex_mean_A":  float(rmsd_all.mean()),
    "rmsd_complex_std_A":   float(rmsd_all.std()),
    "rmsd_complex_max_A":   float(rmsd_all.max()),
    "rmsd_lasr_mean_A":     float(rmsd_lasr.mean()),
    "rmsd_lasr_std_A":      float(rmsd_lasr.std()),
    "rmsd_nb_mean_A":       float(rmsd_nb.mean()),
    "rmsd_nb_std_A":        float(rmsd_nb.std()),
    "rg_mean_A":            float(rg.mean()),
    "rg_std_A":             float(rg.std()),
    "n_hbonds_30pct":       int(len(hbonds)),
    "sasa_mean_A2":         float(sasa.mean()),
}
with open(f"{MDDIR}/analysis/md_analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

np.save(f"{MDDIR}/analysis/rmsd_complex.npy", rmsd_all)
np.save(f"{MDDIR}/analysis/rmsd_lasr.npy",    rmsd_lasr)
np.save(f"{MDDIR}/analysis/rmsd_nb.npy",      rmsd_nb)
np.save(f"{MDDIR}/analysis/rmsf_ca.npy",      rmsf_ca)
np.save(f"{MDDIR}/analysis/rg.npy",           rg)
np.save(f"{MDDIR}/analysis/sasa.npy",         sasa)
np.save(f"{MDDIR}/analysis/time_ns.npy",      time_ns)
np.save(f"{MDDIR}/analysis/res_ids.npy",      np.array(res_ids))
log.info(f"Saved -> {MDDIR}/analysis/")
log.info("STEP 7 COMPLETE.")
