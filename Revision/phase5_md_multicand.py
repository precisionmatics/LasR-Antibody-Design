"""
Parameterized 100 ns MD for any NbLasR candidate.
Usage:  python phase5_md_multicand.py NbLasR-1

Identical protocol to phase5_md_simulation.py (NbLasR-2):
  AMBER ff14SB + TIP3P, orthorhombic PBC, PME, NPT 300 K/1 bar,
  100 ns production, 2 fs timestep, save every 10 ps.
"""

import os, sys, time, json, logging, warnings
import numpy as np
warnings.filterwarnings("ignore")

# ── CLI ───────────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python phase5_md_multicand.py <candidate>  e.g. NbLasR-1")
    sys.exit(1)

CANDIDATE = sys.argv[1]   # e.g. "NbLasR-1"

BASE    = "/home/stalin/Desktop/LasR_Antibody_Design"
COMPLEX = f"{BASE}/04_Docking/{CANDIDATE}_MEGADOCK_complex.pdb"
MDDIR   = f"{BASE}/05_MD_Simulation/{CANDIDATE}"
FIGDIR  = f"{MDDIR}/figures"
LOGFILE = f"{BASE}/logs/phase5_md_{CANDIDATE}_{time.strftime('%Y%m%d_%H%M%S')}.log"

for d in [f"{MDDIR}/prep", f"{MDDIR}/minimization",
          f"{MDDIR}/nvt", f"{MDDIR}/npt",
          f"{MDDIR}/production", f"{MDDIR}/analysis",
          FIGDIR, f"{BASE}/logs"]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOGFILE), logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ── MD parameters (identical to NbLasR-2 run) ────────────────────────────────
TEMP_K         = 300
PRESSURE_ATM   = 1.0
TIMESTEP_PS    = 0.002
BOX_PADDING_NM = 1.2
IONIC_STR_M    = 0.15

MINIM_STEPS    = 10_000
MINIM_STEPS2   = 20_000
NVT_STEPS      = 100_000
NPT_STEPS      = 250_000
PROD_STEPS     = 50_000_000   # 100 ns
SAVE_EVERY     = 5_000
REPORT_EVERY   = 5_000

# ── OpenMM imports ────────────────────────────────────────────────────────────
log.info(f"Candidate: {CANDIDATE}")
log.info("Importing OpenMM...")
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from pdbfixer import PDBFixer
from openmm import XmlSerializer

log.info(f"OpenMM {mm.__version__}")


def copy_system(system):
    return XmlSerializer.deserialize(XmlSerializer.serialize(system))


def add_restraints(system, topology, positions, k_kJ, backbone_only=False):
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


def prepare_system():
    log.info("=" * 60)
    log.info("STEP 1: System Preparation (PDBFixer)")
    log.info("=" * 60)
    fixer = PDBFixer(filename=COMPLEX)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.4)
    log.info("  Missing residues/atoms fixed, H added at pH 7.4")
    fixer.addSolvent(
        padding=BOX_PADDING_NM * unit.nanometers,
        ionicStrength=IONIC_STR_M * unit.molar,
        positiveIon="Na+", negativeIon="Cl-",
    )
    log.info(f"  Orthorhombic PBC box: TIP3P, {BOX_PADDING_NM} nm padding, "
             f"{IONIC_STR_M*1000:.0f} mM NaCl")
    prep_pdb = f"{MDDIR}/prep/system_solvated.pdb"
    with open(prep_pdb, "w") as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
    n_atoms = fixer.topology.getNumAtoms()
    n_res   = fixer.topology.getNumResidues()
    log.info(f"  System: {n_atoms} atoms, {n_res} residues  →  {prep_pdb}")
    return fixer.topology, fixer.positions


def build_system(topology, positions):
    log.info("=" * 60)
    log.info("STEP 2: Building OpenMM system (AMBER ff14SB + TIP3P, PME)")
    log.info("=" * 60)
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometers,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=0.0005,
    )
    barostat = mm.MonteCarloBarostat(
        PRESSURE_ATM * unit.atmospheres,
        TEMP_K * unit.kelvin,
        25
    )
    system.addForce(barostat)
    log.info(f"  System built: {system.getNumParticles()} particles, PME, HBond constraints")
    return system, forcefield


def run_minimization(system, topology, positions):
    log.info("=" * 60)
    log.info("STEP 3: Energy Minimisation (2-stage restrained)")
    log.info("=" * 60)
    platform  = mm.Platform.getPlatformByName("CUDA")
    props_dbl = {"DeviceIndex": "0", "Precision": "double"}

    sys1   = copy_system(system)
    add_restraints(sys1, topology, positions, k_kJ=5000.0, backbone_only=False)
    integ1 = mm.LangevinMiddleIntegrator(TEMP_K*unit.kelvin, 1.0/unit.picoseconds,
                                          TIMESTEP_PS*unit.picoseconds)
    sim1   = app.Simulation(topology, sys1, integ1, platform, props_dbl)
    sim1.context.setPositions(positions)
    e0 = sim1.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole)
    log.info(f"  Stage-1 start energy: {e0:.1f} kJ/mol")
    sim1.minimizeEnergy(maxIterations=MINIM_STEPS, tolerance=1.0)
    pos1 = sim1.context.getState(getPositions=True).getPositions()
    e1 = sim1.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole)
    log.info(f"  Stage-1 end energy:   {e1:.1f} kJ/mol")

    sys2   = copy_system(system)
    integ2 = mm.LangevinMiddleIntegrator(TEMP_K*unit.kelvin, 1.0/unit.picoseconds,
                                          TIMESTEP_PS*unit.picoseconds)
    sim2   = app.Simulation(topology, sys2, integ2, platform, props_dbl)
    sim2.context.setPositions(pos1)
    sim2.minimizeEnergy(maxIterations=MINIM_STEPS2, tolerance=1.0)
    e2 = sim2.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole)
    log.info(f"  Stage-2 end energy:   {e2:.1f} kJ/mol")

    positions_min = sim2.context.getState(getPositions=True).getPositions()
    minim_pdb = f"{MDDIR}/minimization/minimised.pdb"
    with open(minim_pdb, "w") as f:
        app.PDBFile.writeFile(topology, positions_min, f)
    log.info(f"  Saved: {minim_pdb}")
    return positions_min


def run_nvt(system, topology, positions):
    log.info("=" * 60)
    log.info("STEP 4: NVT Equilibration (200 ps, 300 K, backbone restraints)")
    log.info("=" * 60)
    sys_nvt = copy_system(system)
    for i in range(sys_nvt.getNumForces()):
        if isinstance(sys_nvt.getForce(i), mm.MonteCarloBarostat):
            sys_nvt.getForce(i).setFrequency(0)
            break
    add_restraints(sys_nvt, topology, positions, k_kJ=500.0, backbone_only=True)
    integrator = mm.LangevinMiddleIntegrator(TEMP_K*unit.kelvin, 1.0/unit.picoseconds,
                                              TIMESTEP_PS*unit.picoseconds)
    integrator.setRandomNumberSeed(42)
    platform   = mm.Platform.getPlatformByName("CUDA")
    properties = {"DeviceIndex": "0", "Precision": "mixed"}
    simulation = app.Simulation(topology, sys_nvt, integrator, platform, properties)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(TEMP_K*unit.kelvin, 42)
    nvt_log = f"{MDDIR}/nvt/nvt.log"
    simulation.reporters.append(
        app.StateDataReporter(nvt_log, REPORT_EVERY,
            step=True, time=True, potentialEnergy=True, kineticEnergy=True,
            totalEnergy=True, temperature=True, volume=True,
            progress=True, remainingTime=True, speed=True,
            totalSteps=NVT_STEPS, separator="\t"))
    t0 = time.time()
    simulation.step(NVT_STEPS)
    log.info(f"  NVT done in {time.time()-t0:.1f}s")
    state_nvt = simulation.context.getState(getPositions=True, getVelocities=True,
                                             enforcePeriodicBox=True)
    pos_nvt = state_nvt.getPositions()
    vel_nvt = state_nvt.getVelocities()
    box_nvt = state_nvt.getPeriodicBoxVectors()
    with open(f"{MDDIR}/nvt/nvt_final.pdb", "w") as f:
        app.PDBFile.writeFile(topology, pos_nvt, f)
    return pos_nvt, vel_nvt, box_nvt


def run_npt(system, topology, positions, velocities, box_vectors):
    log.info("=" * 60)
    log.info("STEP 5: NPT Equilibration (500 ps, 300 K, 1 bar, light restraints)")
    log.info("=" * 60)
    sys_npt = copy_system(system)
    add_restraints(sys_npt, topology, positions, k_kJ=50.0, backbone_only=True)
    integrator = mm.LangevinMiddleIntegrator(TEMP_K*unit.kelvin, 1.0/unit.picoseconds,
                                              TIMESTEP_PS*unit.picoseconds)
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
            step=True, time=True, potentialEnergy=True, kineticEnergy=True,
            totalEnergy=True, temperature=True, volume=True, density=True,
            progress=True, remainingTime=True, speed=True,
            totalSteps=NPT_STEPS, separator="\t"))
    t0 = time.time()
    simulation.step(NPT_STEPS)
    log.info(f"  NPT done in {time.time()-t0:.1f}s")
    state_npt = simulation.context.getState(getPositions=True, getVelocities=True,
                                             enforcePeriodicBox=True)
    pos_npt = state_npt.getPositions()
    vel_npt = state_npt.getVelocities()
    box_npt = state_npt.getPeriodicBoxVectors()
    log.info(f"  NPT box: {[f'{v._value:.3f}' for v in box_npt[0]][:1]} nm (a-vector x)")
    with open(f"{MDDIR}/npt/npt_final.pdb", "w") as f:
        app.PDBFile.writeFile(topology, pos_npt, f)
    return pos_npt, vel_npt, box_npt


def run_production(system, topology, positions, velocities, box_vectors):
    log.info("=" * 60)
    log.info(f"STEP 6: Production MD — NPT ensemble, {PROD_STEPS*TIMESTEP_PS/1000:.0f} ns")
    log.info("=" * 60)
    integrator = mm.LangevinMiddleIntegrator(
        TEMP_K * unit.kelvin,
        1.0 / unit.picoseconds,
        TIMESTEP_PS * unit.picoseconds
    )
    integrator.setRandomNumberSeed(44)
    platform   = mm.Platform.getPlatformByName("CUDA")
    properties = {"DeviceIndex": "0", "Precision": "mixed"}
    simulation = app.Simulation(topology, system, integrator, platform, properties)
    simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.setPositions(positions)
    simulation.context.setVelocities(velocities)

    traj_dcd = f"{MDDIR}/production/trajectory.dcd"
    prod_log = f"{MDDIR}/production/production.log"
    chkpt    = f"{MDDIR}/production/checkpoint.chk"

    simulation.reporters.append(app.DCDReporter(traj_dcd, SAVE_EVERY))
    simulation.reporters.append(
        app.StateDataReporter(prod_log, REPORT_EVERY,
            step=True, time=True, potentialEnergy=True, kineticEnergy=True,
            totalEnergy=True, temperature=True, volume=True, density=True,
            progress=True, remainingTime=True, speed=True,
            totalSteps=PROD_STEPS, separator="\t"))
    simulation.reporters.append(app.CheckpointReporter(chkpt, 50_000))

    log.info(f"  Trajectory → {traj_dcd}")
    log.info(f"  Saving every {SAVE_EVERY * TIMESTEP_PS:.0f} ps "
             f"({PROD_STEPS // SAVE_EVERY} frames total)")

    t0 = time.time()
    simulation.step(PROD_STEPS)
    hrs = (time.time() - t0) / 3600
    log.info(f"  Production MD complete in {hrs:.2f} hours")

    pos_final = simulation.context.getState(
        getPositions=True, enforcePeriodicBox=True).getPositions()
    with open(f"{MDDIR}/production/final_frame.pdb", "w") as f:
        app.PDBFile.writeFile(topology, pos_final, f)
    return traj_dcd


def run_analysis(traj_dcd):
    log.info("=" * 60)
    log.info("STEP 7: Trajectory Analysis")
    log.info("=" * 60)
    import mdtraj as md

    prep_pdb  = f"{MDDIR}/prep/system_solvated.pdb"
    # stride=5 → 2000 frames; sufficient for all analyses, uses ~1.3 GB vs 6.7 GB
    traj = md.load_dcd(traj_dcd, top=prep_pdb, stride=5)
    log.info(f"  Loaded {traj.n_frames} frames (stride=5), {traj.n_atoms} atoms, "
             f"{traj.time[-1]/1000:.1f} ns")

    # Fix PBC wrapping before any analysis — keeps both chains in same image
    traj.image_molecules(inplace=True)

    prot_idx  = traj.topology.select("protein")
    traj_prot = traj.atom_slice(prot_idx)
    bb_idx    = traj_prot.topology.select("backbone")
    traj_prot.superpose(traj_prot, 0, atom_indices=bb_idx)

    rmsd_all  = md.rmsd(traj_prot, traj_prot, 0, atom_indices=bb_idx) * 10
    lasr_idx  = traj_prot.topology.select("chainid 0 and backbone")
    nb_idx    = traj_prot.topology.select("chainid 1 and backbone")
    rmsd_lasr = md.rmsd(traj_prot, traj_prot, 0, atom_indices=lasr_idx) * 10
    rmsd_nb   = md.rmsd(traj_prot, traj_prot, 0, atom_indices=nb_idx) * 10
    time_ns   = traj.time / 1000

    ca_idx  = traj_prot.topology.select("name CA")
    rmsf_ca = md.rmsf(traj_prot, traj_prot, atom_indices=ca_idx) * 10
    rg      = md.compute_rg(traj_prot) * 10
    hbonds  = md.baker_hubbard(traj_prot, freq=0.3, periodic=False)
    sasa    = md.shrake_rupley(traj_prot).sum(axis=1) * 100
    res_ids = [r.resSeq for r in traj_prot.topology.residues][:len(rmsf_ca)]

    log.info(f"  RMSD complex : {rmsd_all.mean():.2f} ± {rmsd_all.std():.2f} Å")
    log.info(f"  RMSD LasR    : {rmsd_lasr.mean():.2f} ± {rmsd_lasr.std():.2f} Å")
    log.info(f"  RMSD Nb      : {rmsd_nb.mean():.2f} ± {rmsd_nb.std():.2f} Å")
    log.info(f"  Rg           : {rg.mean():.2f} ± {rg.std():.2f} Å")
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
    np.save(f"{MDDIR}/analysis/rmsd_nb.npy",       rmsd_nb)
    np.save(f"{MDDIR}/analysis/rmsf_ca.npy",       rmsf_ca)
    np.save(f"{MDDIR}/analysis/rg.npy",            rg)
    np.save(f"{MDDIR}/analysis/sasa.npy",          sasa)
    np.save(f"{MDDIR}/analysis/time_ns.npy",       time_ns)
    np.save(f"{MDDIR}/analysis/res_ids.npy",       np.array(res_ids))

    return results, time_ns, rmsd_all, rmsd_lasr, rmsd_nb, rmsf_ca, rg, sasa, res_ids


def generate_figures(time_ns, rmsd_all, rmsd_lasr, rmsd_nb, rmsf_ca, rg, sasa, res_ids):
    log.info("=" * 60)
    log.info("STEP 8: Generating Figures")
    log.info("=" * 60)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGDIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(time_ns, rmsd_all,  color="#2196F3", lw=1.0, label="Complex (backbone)", alpha=0.9)
    ax.plot(time_ns, rmsd_lasr, color="#4CAF50", lw=1.0, label="LasR LBD (chain E)", alpha=0.9)
    ax.plot(time_ns, rmsd_nb,   color="#FF9800", lw=1.0, label=f"{CANDIDATE} (chain A)", alpha=0.9)
    ax.set_xlabel("Time (ns)", fontsize=13)
    ax.set_ylabel("RMSD (Å)", fontsize=13)
    ax.set_title(f"Backbone RMSD — {CANDIDATE}:LasR Complex (100 ns MD)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{FIGDIR}/Fig_MD_01_RMSD.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, data, lbl, col in zip(
        axes,
        [rmsd_all, rmsd_lasr, rmsd_nb],
        ["Complex", "LasR LBD", CANDIDATE],
        ["#2196F3", "#4CAF50", "#FF9800"]
    ):
        ax.hist(data, bins=50, color=col, edgecolor="black", lw=0.5, alpha=0.85)
        ax.axvline(data.mean(), color="red", lw=2, ls="--",
                   label=f"Mean={data.mean():.2f} Å")
        ax.set_xlabel("RMSD (Å)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"RMSD Distribution — {lbl}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{FIGDIR}/Fig_MD_05_RMSD_Distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(time_ns, rg, color="#F44336", lw=1.0, alpha=0.85)
    ax.axhline(rg.mean(), color="black", lw=1.5, ls="--",
               label=f"Mean = {rg.mean():.2f} Å")
    ax.set_xlabel("Time (ns)", fontsize=13)
    ax.set_ylabel("Radius of Gyration (Å)", fontsize=13)
    ax.set_title(f"Rg — {CANDIDATE}:LasR Complex", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{FIGDIR}/Fig_MD_03_Rg.png", dpi=300, bbox_inches="tight")
    plt.close()

    log.info(f"  Figures saved → {FIGDIR}/")


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 70)
    log.info(f"  PHASE 5 MD — {CANDIDATE} : LasR LBD Complex")
    log.info(f"  Input: {COMPLEX}")
    log.info(f"  Run:   {PROD_STEPS * TIMESTEP_PS / 1_000_000:.0f} ns NPT production MD")
    log.info("=" * 70)

    t_total = time.time()

    topology, positions              = prepare_system()
    system, ff                       = build_system(topology, positions)
    positions                        = run_minimization(system, topology, positions)
    positions, velocities, box_vecs  = run_nvt(system, topology, positions)
    positions, velocities, box_vecs  = run_npt(system, topology, positions, velocities, box_vecs)
    traj_dcd                         = run_production(system, topology, positions, velocities, box_vecs)
    res = run_analysis(traj_dcd)
    (analysis, time_ns, rmsd_all, rmsd_lasr,
     rmsd_nb, rmsf_ca, rg, sasa, res_ids)  = res
    generate_figures(time_ns, rmsd_all, rmsd_lasr, rmsd_nb, rmsf_ca, rg, sasa, res_ids)

    total_hrs = (time.time() - t_total) / 3600
    log.info("=" * 70)
    log.info(f"  COMPLETE — {CANDIDATE}  Total: {total_hrs:.2f} hours")
    log.info(f"  Trajectory: {MDDIR}/production/trajectory.dcd")
    log.info(f"  Analysis:   {MDDIR}/analysis/md_analysis_results.json")
    log.info("=" * 70)
