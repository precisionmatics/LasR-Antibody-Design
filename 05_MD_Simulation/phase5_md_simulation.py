"""
PHASE 5 — Molecular Dynamics Simulation
NbLasR-2 : LasR LBD complex (MEGADOCK pose, chain E=LasR, chain A=nanobody)

Pipeline:
  1. System prep     — PDBFixer (missing residues, hydrogens, solvation, ions)
  2. Minimisation    — 5000 steps steepest descent
  3. NVT equilib.    — 100 ps, 300 K, heavy-atom restraints
  4. NPT equilib.    — 100 ps, 300 K, 1 bar, light restraints
  5. Production MD   — 100 ns, 300 K, 1 bar, NPT, save every 10 ps
  6. Analysis        — RMSD, RMSF, Rg, H-bonds, SASA, contact map
  7. Figures         — Publication-quality plots (12 panels)

Force field : AMBER ff14SB (protein) + TIP3P (water)
GPU         : CUDA (RTX 5060 Ti)
"""

import os, sys, time, json, logging, warnings
import numpy as np

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
BASE    = "/home/stalin/Desktop/LasR_Antibody_Design"
COMPLEX = f"{BASE}/04_Docking/NbLasR-2_MEGADOCK_complex.pdb"
MDDIR   = f"{BASE}/05_MD_Simulation/NbLasR-2"
FIGDIR  = f"{MDDIR}/figures"
LOGFILE = f"{BASE}/logs/phase5_md_{time.strftime('%Y%m%d_%H%M%S')}.log"

os.makedirs(f"{BASE}/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOGFILE), logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ── MD parameters ─────────────────────────────────────────────────────────────
TEMP_K          = 300       # Kelvin
PRESSURE_ATM    = 1.0       # bar
TIMESTEP_PS     = 0.002     # 2 fs
BOX_PADDING_NM  = 1.2       # 12 Å water shell
IONIC_STR_M     = 0.15      # 150 mM NaCl

MINIM_STEPS     = 10_000    # Stage-1 restrained (water relax)
MINIM_STEPS2    = 20_000    # Stage-2 free minimisation
NVT_STEPS       = 100_000   # 200 ps
NPT_STEPS       = 250_000   # 500 ps
PROD_STEPS      = 50_000_000  # 100 ns
SAVE_EVERY      = 5_000     # save frame every 10 ps (5000 steps × 2 fs)
REPORT_EVERY    = 5_000     # log every 10 ps

# ── OpenMM imports ────────────────────────────────────────────────────────────
log.info("Importing OpenMM...")
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from pdbfixer import PDBFixer

log.info(f"OpenMM {mm.__version__} — CUDA platform ready")

from openmm import XmlSerializer


def copy_system(system):
    """Deep-copy an OpenMM System via XML serialization."""
    return XmlSerializer.deserialize(XmlSerializer.serialize(system))


def add_restraints(system, topology, positions, k_kJ, backbone_only=False):
    """Harmonic positional restraints on protein heavy atoms (kJ/mol/nm²)."""
    SKIP = {'HOH', 'WAT', 'TIP3', 'SOL', 'NA', 'CL', 'Na+', 'Cl-'}
    BB   = {'N', 'CA', 'C', 'O'}
    force = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", k_kJ)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    n = 0
    for atom in topology.atoms():
        if atom.residue.name in SKIP:
            continue
        if backbone_only and atom.name not in BB:
            continue
        if atom.element is not None and atom.element.symbol == 'H':
            continue
        p = positions[atom.index].value_in_unit(unit.nanometers)
        force.addParticle(atom.index, [p.x, p.y, p.z])
        n += 1
    system.addForce(force)
    tag = "backbone" if backbone_only else "all-heavy"
    log.info(f"  Restraints: {n} {tag} atoms @ k={k_kJ:.0f} kJ/mol/nm²")
    return system


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — System Preparation (PDBFixer)
# ══════════════════════════════════════════════════════════════════════════════
def prepare_system():
    log.info("="*60)
    log.info("STEP 1: System Preparation (PDBFixer)")
    log.info("="*60)

    fixer = PDBFixer(filename=COMPLEX)

    # find and fix missing residues/atoms
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.4)

    log.info("  Missing residues/atoms fixed, H added at pH 7.4")

    # solvate in TIP3P water box (water model set via force field, not PDBFixer)
    fixer.addSolvent(
        padding=BOX_PADDING_NM * unit.nanometers,
        ionicStrength=IONIC_STR_M * unit.molar,
        positiveIon="Na+", negativeIon="Cl-",
    )
    log.info(f"  Solvated with TIP3P, {BOX_PADDING_NM} nm padding, {IONIC_STR_M*1000:.0f} mM NaCl")

    prep_pdb = f"{MDDIR}/prep/system_solvated.pdb"
    with open(prep_pdb, "w") as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
    log.info(f"  Saved: {prep_pdb}")

    # count atoms
    n_atoms = fixer.topology.getNumAtoms()
    n_res   = fixer.topology.getNumResidues()
    log.info(f"  System: {n_atoms} atoms, {n_res} residues")

    return fixer.topology, fixer.positions


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Build OpenMM System (AMBER ff14SB + TIP3P)
# ══════════════════════════════════════════════════════════════════════════════
def build_system(topology, positions):
    log.info("="*60)
    log.info("STEP 2: Building OpenMM system (AMBER ff14SB + TIP3P)")
    log.info("="*60)

    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    system = forcefield.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometers,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=0.0005,
    )

    # add barostat for NPT
    barostat = mm.MonteCarloBarostat(
        PRESSURE_ATM * unit.atmospheres,
        TEMP_K * unit.kelvin,
        25
    )
    system.addForce(barostat)
    log.info(f"  System built: {system.getNumParticles()} particles, PME, HBond constraints")

    return system, forcefield


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Energy Minimisation
# ══════════════════════════════════════════════════════════════════════════════
def run_minimization(system, topology, positions):
    log.info("="*60)
    log.info("STEP 3: Energy Minimisation (2-stage restrained)")
    log.info("="*60)

    platform  = mm.Platform.getPlatformByName("CUDA")
    props_dbl = {"DeviceIndex": "0", "Precision": "double"}
    props_mix = {"DeviceIndex": "0", "Precision": "mixed"}

    # Stage 1: strong restraints on all protein heavy atoms — only water moves
    sys1   = copy_system(system)
    add_restraints(sys1, topology, positions, k_kJ=5000.0, backbone_only=False)
    integ1 = mm.LangevinMiddleIntegrator(TEMP_K*unit.kelvin, 1.0/unit.picoseconds, TIMESTEP_PS*unit.picoseconds)
    sim1   = app.Simulation(topology, sys1, integ1, platform, props_dbl)
    sim1.context.setPositions(positions)
    e0 = sim1.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    log.info(f"  Stage-1 start energy: {e0:.1f} kJ/mol")
    sim1.minimizeEnergy(maxIterations=MINIM_STEPS, tolerance=1.0)
    pos1 = sim1.context.getState(getPositions=True).getPositions()
    e1 = sim1.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    log.info(f"  Stage-1 end energy:   {e1:.1f} kJ/mol")

    # Stage 2: free minimisation of everything
    sys2   = copy_system(system)
    integ2 = mm.LangevinMiddleIntegrator(TEMP_K*unit.kelvin, 1.0/unit.picoseconds, TIMESTEP_PS*unit.picoseconds)
    sim2   = app.Simulation(topology, sys2, integ2, platform, props_dbl)
    sim2.context.setPositions(pos1)
    sim2.minimizeEnergy(maxIterations=MINIM_STEPS2, tolerance=1.0)
    e2 = sim2.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    log.info(f"  Stage-2 end energy:   {e2:.1f} kJ/mol")

    positions_min = sim2.context.getState(getPositions=True).getPositions()
    minim_pdb = f"{MDDIR}/minimization/minimised.pdb"
    with open(minim_pdb, "w") as f:
        app.PDBFile.writeFile(topology, positions_min, f)
    log.info(f"  Saved: {minim_pdb}")

    return positions_min


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — NVT Equilibration (constant volume, heat to 300 K)
# ══════════════════════════════════════════════════════════════════════════════
def run_nvt(system, topology, positions):
    log.info("="*60)
    log.info("STEP 4: NVT Equilibration (200 ps, 300 K, backbone restraints)")
    log.info("="*60)

    sys_nvt = copy_system(system)
    # disable barostat in NVT copy
    for i in range(sys_nvt.getNumForces()):
        if isinstance(sys_nvt.getForce(i), mm.MonteCarloBarostat):
            sys_nvt.getForce(i).setFrequency(0)
            break
    # strong backbone restraints to keep protein in place while water/ions relax
    add_restraints(sys_nvt, topology, positions, k_kJ=500.0, backbone_only=True)

    integrator = mm.LangevinMiddleIntegrator(TEMP_K*unit.kelvin, 1.0/unit.picoseconds, TIMESTEP_PS*unit.picoseconds)
    integrator.setRandomNumberSeed(42)
    platform   = mm.Platform.getPlatformByName("CUDA")
    properties = {"DeviceIndex": "0", "Precision": "mixed"}
    simulation = app.Simulation(topology, sys_nvt, integrator, platform, properties)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(TEMP_K*unit.kelvin, 42)

    nvt_log = f"{MDDIR}/nvt/nvt.log"
    simulation.reporters.append(
        app.StateDataReporter(nvt_log, REPORT_EVERY,
            step=True, time=True, potentialEnergy=True,
            kineticEnergy=True, totalEnergy=True,
            temperature=True, volume=True, progress=True,
            remainingTime=True, speed=True,
            totalSteps=NVT_STEPS, separator="\t")
    )
    t0 = time.time()
    simulation.step(NVT_STEPS)
    elapsed = time.time() - t0
    log.info(f"  NVT done in {elapsed:.1f}s")

    nvt_pdb = f"{MDDIR}/nvt/nvt_final.pdb"
    state_nvt = simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    pos_nvt   = state_nvt.getPositions()
    vel_nvt   = state_nvt.getVelocities()
    box_nvt   = state_nvt.getPeriodicBoxVectors()
    with open(nvt_pdb, "w") as f:
        app.PDBFile.writeFile(topology, pos_nvt, f)

    return pos_nvt, vel_nvt, box_nvt


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — NPT Equilibration (constant pressure, 300 K, 1 bar)
# ══════════════════════════════════════════════════════════════════════════════
def run_npt(system, topology, positions, velocities, box_vectors):
    log.info("="*60)
    log.info("STEP 5: NPT Equilibration (500 ps, 300 K, 1 bar, light restraints)")
    log.info("="*60)

    sys_npt = copy_system(system)
    # light backbone restraints — box adjusts freely, protein still guided
    add_restraints(sys_npt, topology, positions, k_kJ=50.0, backbone_only=True)

    integrator = mm.LangevinMiddleIntegrator(TEMP_K*unit.kelvin, 1.0/unit.picoseconds, TIMESTEP_PS*unit.picoseconds)
    integrator.setRandomNumberSeed(43)
    platform   = mm.Platform.getPlatformByName("CUDA")
    properties = {"DeviceIndex": "0", "Precision": "mixed"}
    simulation = app.Simulation(topology, sys_npt, integrator, platform, properties)
    simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.setPositions(positions)
    simulation.context.setVelocities(velocities)

    npt_log = f"{MDDIR}/npt/npt.log"
    simulation.reporters.append(
        app.StateDataReporter(npt_log, REPORT_EVERY,
            step=True, time=True, potentialEnergy=True,
            kineticEnergy=True, totalEnergy=True,
            temperature=True, volume=True, density=True,
            progress=True, remainingTime=True, speed=True,
            totalSteps=NPT_STEPS, separator="\t")
    )
    t0 = time.time()
    simulation.step(NPT_STEPS)
    elapsed = time.time() - t0
    log.info(f"  NPT done in {elapsed:.1f}s")

    npt_pdb = f"{MDDIR}/npt/npt_final.pdb"
    state_npt = simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    pos_npt   = state_npt.getPositions()
    vel_npt   = state_npt.getVelocities()
    box_npt   = state_npt.getPeriodicBoxVectors()
    log.info(f"  NPT box: {[f'{v._value:.3f}' for v in box_npt[0]][:1]} nm (a-vector x)")
    with open(npt_pdb, "w") as f:
        app.PDBFile.writeFile(topology, pos_npt, f)

    return pos_npt, vel_npt, box_npt


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Production MD (100 ns)
# ══════════════════════════════════════════════════════════════════════════════
def run_production(system, topology, positions, velocities, box_vectors):
    log.info("="*60)
    log.info(f"STEP 6: Production MD — {PROD_STEPS*TIMESTEP_PS/1000:.0f} ns")
    log.info("="*60)

    integrator = mm.LangevinMiddleIntegrator(
        TEMP_K * unit.kelvin,
        1.0 / unit.picoseconds,
        TIMESTEP_PS * unit.picoseconds
    )
    integrator.setRandomNumberSeed(44)

    platform   = mm.Platform.getPlatformByName("CUDA")
    properties = {"DeviceIndex": "0", "Precision": "mixed"}
    simulation = app.Simulation(topology, system, integrator, platform, properties)
    # CRITICAL: apply equilibrated NPT box vectors before setting positions
    simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.setPositions(positions)
    simulation.context.setVelocities(velocities)

    traj_dcd  = f"{MDDIR}/production/trajectory.dcd"
    prod_log  = f"{MDDIR}/production/production.log"
    chkpt     = f"{MDDIR}/production/checkpoint.chk"

    simulation.reporters.append(app.DCDReporter(traj_dcd, SAVE_EVERY))
    simulation.reporters.append(
        app.StateDataReporter(prod_log, REPORT_EVERY,
            step=True, time=True, potentialEnergy=True,
            kineticEnergy=True, totalEnergy=True,
            temperature=True, volume=True, density=True,
            progress=True, remainingTime=True, speed=True,
            totalSteps=PROD_STEPS, separator="\t")
    )
    simulation.reporters.append(app.CheckpointReporter(chkpt, 50_000))

    log.info(f"  Trajectory: {traj_dcd}")
    log.info(f"  Saving every {SAVE_EVERY * TIMESTEP_PS:.0f} ps  "
             f"({PROD_STEPS // SAVE_EVERY} frames total)")

    t0 = time.time()
    simulation.step(PROD_STEPS)
    elapsed = time.time() - t0
    hrs = elapsed / 3600
    log.info(f"  Production MD complete in {hrs:.2f} hours")

    final_pdb = f"{MDDIR}/production/final_frame.pdb"
    pos_final = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()
    with open(final_pdb, "w") as f:
        app.PDBFile.writeFile(topology, pos_final, f)

    return traj_dcd, simulation


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — Trajectory Analysis
# ══════════════════════════════════════════════════════════════════════════════
def run_analysis(topology, traj_dcd):
    log.info("="*60)
    log.info("STEP 7: Trajectory Analysis")
    log.info("="*60)

    import mdtraj as md

    prep_pdb  = f"{MDDIR}/prep/system_solvated.pdb"
    log.info("  Loading trajectory...")
    traj = md.load_dcd(traj_dcd, top=prep_pdb)
    log.info(f"  Loaded {traj.n_frames} frames, {traj.n_atoms} atoms, "
             f"{traj.time[-1]/1000:.1f} ns")

    # select protein only (remove water/ions)
    prot_idx = traj.topology.select("protein")
    traj_prot = traj.atom_slice(prot_idx)

    # superpose on first frame (backbone)
    bb_idx = traj_prot.topology.select("backbone")
    traj_prot.superpose(traj_prot, 0, atom_indices=bb_idx)

    # ── RMSD ──────────────────────────────────────────────────────────────────
    rmsd_all  = md.rmsd(traj_prot, traj_prot, 0, atom_indices=bb_idx) * 10  # nm→Å
    # LasR chain E residues
    lasr_idx  = traj_prot.topology.select("chainid 0 and backbone")
    nb_idx    = traj_prot.topology.select("chainid 1 and backbone")
    rmsd_lasr = md.rmsd(traj_prot, traj_prot, 0, atom_indices=lasr_idx) * 10
    rmsd_nb   = md.rmsd(traj_prot, traj_prot, 0, atom_indices=nb_idx) * 10
    time_ns   = traj.time / 1000

    # ── RMSF ──────────────────────────────────────────────────────────────────
    rmsf_all  = md.rmsf(traj_prot, traj_prot, atom_indices=bb_idx) * 10
    ca_idx    = traj_prot.topology.select("name CA")
    rmsf_ca   = md.rmsf(traj_prot, traj_prot, atom_indices=ca_idx) * 10
    res_ids   = [r.resSeq for r in traj_prot.topology.residues][:len(rmsf_ca)]

    # ── Radius of Gyration ─────────────────────────────────────────────────────
    rg = md.compute_rg(traj_prot) * 10  # nm→Å

    # ── H-bonds ───────────────────────────────────────────────────────────────
    hbonds = md.baker_hubbard(traj_prot, freq=0.3, periodic=False)
    log.info(f"  H-bonds with occupancy >30%: {len(hbonds)}")

    # ── SASA ──────────────────────────────────────────────────────────────────
    sasa = md.shrake_rupley(traj_prot)
    total_sasa = sasa.sum(axis=1) * 100  # nm²→Å²

    # ── Save analysis results ──────────────────────────────────────────────────
    results = {
        "n_frames":         int(traj.n_frames),
        "sim_time_ns":      float(traj.time[-1]/1000),
        "rmsd_complex_mean_A": float(rmsd_all.mean()),
        "rmsd_complex_max_A":  float(rmsd_all.max()),
        "rmsd_lasr_mean_A":    float(rmsd_lasr.mean()),
        "rmsd_nb_mean_A":      float(rmsd_nb.mean()),
        "rg_mean_A":           float(rg.mean()),
        "rg_std_A":            float(rg.std()),
        "n_hbonds_30pct":      int(len(hbonds)),
        "sasa_mean_A2":        float(total_sasa.mean()),
    }
    json_path = f"{MDDIR}/analysis/md_analysis_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"  Saved analysis JSON: {json_path}")

    # save arrays for figures
    np.save(f"{MDDIR}/analysis/rmsd_complex.npy",  rmsd_all)
    np.save(f"{MDDIR}/analysis/rmsd_lasr.npy",     rmsd_lasr)
    np.save(f"{MDDIR}/analysis/rmsd_nb.npy",       rmsd_nb)
    np.save(f"{MDDIR}/analysis/rmsf_ca.npy",       rmsf_ca)
    np.save(f"{MDDIR}/analysis/rg.npy",            rg)
    np.save(f"{MDDIR}/analysis/sasa.npy",          total_sasa)
    np.save(f"{MDDIR}/analysis/time_ns.npy",       time_ns)
    np.save(f"{MDDIR}/analysis/res_ids.npy",       np.array(res_ids))

    log.info(f"\n  --- Analysis Summary ---")
    log.info(f"  RMSD complex : {rmsd_all.mean():.2f} ± {rmsd_all.std():.2f} Å")
    log.info(f"  RMSD LasR    : {rmsd_lasr.mean():.2f} ± {rmsd_lasr.std():.2f} Å")
    log.info(f"  RMSD Nb      : {rmsd_nb.mean():.2f} ± {rmsd_nb.std():.2f} Å")
    log.info(f"  Rg           : {rg.mean():.2f} ± {rg.std():.2f} Å")
    log.info(f"  H-bonds >30% : {len(hbonds)}")
    log.info(f"  SASA mean    : {total_sasa.mean():.1f} Å²")

    return results, time_ns, rmsd_all, rmsd_lasr, rmsd_nb, rmsf_ca, rg, total_sasa, res_ids


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8 — Figures
# ══════════════════════════════════════════════════════════════════════════════
def generate_figures(time_ns, rmsd_all, rmsd_lasr, rmsd_nb, rmsf_ca, rg, sasa, res_ids):
    log.info("="*60)
    log.info("STEP 8: Generating MD Figures")
    log.info("="*60)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    os.makedirs(FIGDIR, exist_ok=True)

    # ── Fig 1: RMSD all three ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(time_ns, rmsd_all,  color="#2196F3", linewidth=1.0, label="Complex (all backbone)", alpha=0.9)
    ax.plot(time_ns, rmsd_lasr, color="#4CAF50", linewidth=1.0, label="LasR LBD (chain E)", alpha=0.9)
    ax.plot(time_ns, rmsd_nb,   color="#FF9800", linewidth=1.0, label="NbLasR-2 (chain A)", alpha=0.9)
    ax.set_xlabel("Time (ns)", fontsize=12)
    ax.set_ylabel("RMSD (Å)", fontsize=12)
    ax.set_title("Backbone RMSD — NbLasR-2 : LasR Complex (100 ns MD)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_facecolor("#f9f9f9")
    plt.tight_layout()
    fig.savefig(f"{FIGDIR}/Fig_MD_01_RMSD.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  Saved: Fig_MD_01_RMSD.png")

    # ── Fig 2: RMSF per residue ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(range(len(rmsf_ca)), rmsf_ca, color="#9C27B0", linewidth=1.0, alpha=0.85)
    ax.fill_between(range(len(rmsf_ca)), rmsf_ca, alpha=0.15, color="#9C27B0")
    ax.set_xlabel("Residue index", fontsize=12)
    ax.set_ylabel("RMSF (Å)", fontsize=12)
    ax.set_title("Cα RMSF — Residue Flexibility over 100 ns MD", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.set_facecolor("#f9f9f9")
    plt.tight_layout()
    fig.savefig(f"{FIGDIR}/Fig_MD_02_RMSF.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  Saved: Fig_MD_02_RMSF.png")

    # ── Fig 3: Radius of Gyration ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(time_ns, rg, color="#F44336", linewidth=1.0, alpha=0.85)
    ax.axhline(rg.mean(), color="black", linewidth=1.5, linestyle="--",
               label=f"Mean = {rg.mean():.2f} Å")
    ax.fill_between(time_ns, rg.mean()-rg.std(), rg.mean()+rg.std(),
                    alpha=0.15, color="#F44336", label=f"±1σ = {rg.std():.2f} Å")
    ax.set_xlabel("Time (ns)", fontsize=12)
    ax.set_ylabel("Radius of Gyration (Å)", fontsize=12)
    ax.set_title("Radius of Gyration — Complex Compactness over 100 ns", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_facecolor("#f9f9f9")
    plt.tight_layout()
    fig.savefig(f"{FIGDIR}/Fig_MD_03_Rg.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  Saved: Fig_MD_03_Rg.png")

    # ── Fig 4: SASA ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(time_ns, sasa, color="#009688", linewidth=1.0, alpha=0.85)
    ax.axhline(sasa.mean(), color="black", linewidth=1.5, linestyle="--",
               label=f"Mean = {sasa.mean():.1f} Å²")
    ax.set_xlabel("Time (ns)", fontsize=12)
    ax.set_ylabel("Total SASA (Å²)", fontsize=12)
    ax.set_title("Solvent Accessible Surface Area — 100 ns MD", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_facecolor("#f9f9f9")
    plt.tight_layout()
    fig.savefig(f"{FIGDIR}/Fig_MD_04_SASA.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  Saved: Fig_MD_04_SASA.png")

    # ── Fig 5: RMSD distribution histogram ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, data, label, color in zip(
        axes,
        [rmsd_all, rmsd_lasr, rmsd_nb],
        ["Complex", "LasR LBD", "NbLasR-2"],
        ["#2196F3", "#4CAF50", "#FF9800"]
    ):
        ax.hist(data, bins=50, color=color, edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.axvline(data.mean(), color="red", linewidth=2, linestyle="--",
                   label=f"Mean={data.mean():.2f} Å")
        ax.set_xlabel("RMSD (Å)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(f"RMSD Distribution — {label}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_facecolor("#f9f9f9")
    fig.suptitle("RMSD Distributions — 100 ns Production MD", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{FIGDIR}/Fig_MD_05_RMSD_Distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  Saved: Fig_MD_05_RMSD_Distribution.png")

    # ── Fig 6: Free Energy Landscape (FEL) from RMSD vs Rg ────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    # compute FEL: -kT ln(P) where P is 2D histogram
    h, xedges, yedges = np.histogram2d(rmsd_all, rg, bins=50)
    h = h + 1e-10   # avoid log(0)
    FEL = -0.593 * np.log(h / h.max())   # RT at 300K = 0.593 kcal/mol
    FEL = FEL.T
    X = (xedges[:-1] + xedges[1:]) / 2
    Y = (yedges[:-1] + yedges[1:]) / 2
    im = ax.contourf(X, Y, FEL, levels=20, cmap="RdYlGn_r")
    ax.contour(X, Y, FEL, levels=10, colors="black", linewidths=0.4, alpha=0.4)
    plt.colorbar(im, ax=ax, label="Free Energy (kcal/mol)")
    ax.set_xlabel("RMSD (Å)", fontsize=12)
    ax.set_ylabel("Radius of Gyration (Å)", fontsize=12)
    ax.set_title("Free Energy Landscape — RMSD vs Rg\nNbLasR-2 : LasR Complex",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{FIGDIR}/Fig_MD_06_FEL.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  Saved: Fig_MD_06_FEL.png")

    log.info(f"  All MD figures saved to: {FIGDIR}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log.info("="*70)
    log.info("  PHASE 5 — Molecular Dynamics Simulation")
    log.info("  System: NbLasR-2 : LasR LBD Complex")
    log.info(f"  Input:  {COMPLEX}")
    log.info(f"  Run:    {PROD_STEPS * TIMESTEP_PS / 1_000_000:.0f} ns production MD")
    log.info("="*70)

    t_total = time.time()

    # Step 1-2: prepare & build
    topology, positions = prepare_system()
    system, ff          = build_system(topology, positions)

    # Step 3: minimise
    positions = run_minimization(system, topology, positions)

    # Step 4: NVT
    positions, velocities, box_vecs = run_nvt(system, topology, positions)

    # Step 5: NPT
    positions, velocities, box_vecs = run_npt(system, topology, positions, velocities, box_vecs)

    # Step 6: Production — pass equilibrated box vectors so PME uses correct box
    traj_dcd, simulation = run_production(system, topology, positions, velocities, box_vecs)

    # Step 7: Analysis
    res = run_analysis(topology, traj_dcd)
    (analysis, time_ns, rmsd_all, rmsd_lasr,
     rmsd_nb, rmsf_ca, rg, sasa, res_ids) = res

    # Step 8: Figures
    generate_figures(time_ns, rmsd_all, rmsd_lasr, rmsd_nb, rmsf_ca, rg, sasa, res_ids)

    total_hrs = (time.time() - t_total) / 3600
    log.info("\n" + "="*70)
    log.info(f"  PHASE 5 COMPLETE — Total runtime: {total_hrs:.2f} hours")
    log.info(f"  Trajectory: {MDDIR}/production/trajectory.dcd")
    log.info(f"  Analysis:   {MDDIR}/analysis/md_analysis_results.json")
    log.info(f"  Figures:    {FIGDIR}/")
    log.info(f"  Log:        {LOGFILE}")
    log.info("="*70)
