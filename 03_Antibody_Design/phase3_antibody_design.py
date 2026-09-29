"""
Phase 3: In Silico Antibody Design — Nanobody (VHH) targeting LasR LBD
- VHH framework construction (humanized camelid scaffold)
- Rational CDR-H1, CDR-H2, CDR-H3 design via physicochemical complementarity
  to top 2 epitopes: KDSQDYEN (Rank 1) and EHYDRAGYARVDPTV (Rank 2)
- 5 nanobody candidates generated
- ESMFold API for 3D structure prediction
- pLDDT confidence scoring, secondary structure analysis
- Physicochemical property calculation (MW, pI, GRAVY, instability)
- All figures and tables saved
"""

import os, sys, json, time, requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import PDB
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')

BASE   = "/home/stalin/Desktop/LasR_Antibody_Design"
PHASE3 = os.path.join(BASE, "03_Antibody_Design")
FIGS   = os.path.join(BASE, "08_Results_Figures", "Phase3")
LOGS   = os.path.join(BASE, "logs")
DATA   = os.path.join(BASE, "data")
os.makedirs(FIGS,   exist_ok=True)
os.makedirs(PHASE3, exist_ok=True)

LOG_FILE = os.path.join(LOGS, "phase3_log.txt")
COLORS   = {"primary":"#1B4F72","secondary":"#2E86AB","accent":"#E84855",
            "highlight":"#F4A261","green":"#2A9D8F","purple":"#7D3C98",
            "light":"#A8DADC","gold":"#D4AC0D","pink":"#E91E8C"}

def log(msg):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

log("=" * 70)
log("PHASE 3: Antibody Design — STARTED")
log("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load Phase 2 epitope results
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 1: Loading Phase 2 epitope results...")

top3_csv = os.path.join(BASE, "02_Epitope_Mapping", "FINAL_top3_epitopes.csv")
df_top3  = pd.read_csv(top3_csv)
EPI1_SEQ = str(df_top3.iloc[0]["Sequence"])   # KDSQDYEN      — Rank 1 (LBD core)
EPI2_SEQ = str(df_top3.iloc[1]["Sequence"])   # EHYDRAGYARVDPTV — Rank 2 (LBD pocket)
log(f"  Epitope 1 (Rank 1): {EPI1_SEQ}")
log(f"  Epitope 2 (Rank 2): {EPI2_SEQ}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Define humanized VHH (nanobody) framework
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 2: Defining humanized VHH framework scaffold...")

# Humanized llama VHH framework — based on PDB 1MEL / Desmyter et al.
# Positions follow IMGT numbering
VHH_FR1 = "QVQLVESGGGLVQAGGSLRLSCAAS"    # FR1  (1-25)
VHH_FR2 = "WYRQAPGKQRELVS"                # FR2  (36-49)
VHH_FR3 = "RFTISRDNAKNTVYLQMNSLKPEDTAVYYC"  # FR3  (57-94)
VHH_FR4 = "WGQGTQVTVSS"                   # FR4  (103-113)

log(f"  FR1: {VHH_FR1} ({len(VHH_FR1)} aa)")
log(f"  FR2: {VHH_FR2} ({len(VHH_FR2)} aa)")
log(f"  FR3: {VHH_FR3} ({len(VHH_FR3)} aa)")
log(f"  FR4: {VHH_FR4} ({len(VHH_FR4)} aa)")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Physicochemical complementarity rules for CDR design
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 3: Building physicochemical complementarity table...")

# Complementarity rules: what each epitope residue favors in the CDR paratope
COMPLEMENT = {
    # Epitope → Paratope complement
    'K': ['D','E','Q','N'],       # Lys(+) → acidic/polar
    'D': ['R','K','H','Y','N'],   # Asp(-) → basic/H-bond donor
    'S': ['R','K','Y','W','Q'],   # Ser (polar) → H-bond partner
    'Q': ['R','K','W','Y','H'],   # Gln (polar) → H-bond / aromatic
    'Y': ['Y','W','F','H','R'],   # Tyr → aromatic stack
    'E': ['R','K','H','Q','N'],   # Glu(-) → basic/polar
    'N': ['R','K','W','Y','Q'],   # Asn (polar) → H-bond
    'H': ['D','E','Y','W','F'],   # His → acidic / aromatic
    'R': ['D','E','Y','W','F'],   # Arg(+) → acidic / aromatic
    'A': ['I','V','L','W','F'],   # Ala (small) → hydrophobic
    'G': ['W','Y','F','V','L'],   # Gly (flex) → aromatic / hydrophobic
    'V': ['W','F','Y','L','I'],   # Val (hydrophobic) → aromatic / hydrophobic
    'T': ['R','K','Y','W','Q'],   # Thr (polar) → H-bond
    'P': ['G','A','S','T','V'],   # Pro (rigid) → small/flexible
    'I': ['W','F','Y','L','V'],   # Ile → aromatic / hydrophobic
    'L': ['W','F','Y','I','V'],   # Leu → aromatic / hydrophobic
    'F': ['R','K','H','W','Y'],   # Phe → cation-pi or aromatic
    'W': ['R','K','D','E','Y'],   # Trp → cation-pi / aromatic
    'M': ['W','F','Y','L','I'],   # Met → aromatic / hydrophobic
    'C': ['S','T','A','G','V'],   # Cys → small polar
}

def design_cdr(epitope, length_override=None, bias=None):
    """Generate CDR sequence complementary to epitope."""
    cdr = []
    target_len = length_override if length_override else min(len(epitope), 12)
    # Stretch or compress epitope to match CDR length
    epi_sample = epitope * (target_len // len(epitope) + 1)
    for i in range(target_len):
        aa   = epi_sample[i % len(epi_sample)]
        opts = COMPLEMENT.get(aa, ['G','S','T','Y','W'])
        if bias:
            opts = [b for b in bias if b in opts] + opts
        chosen = opts[i % len(opts)]
        cdr.append(chosen)
    return "".join(cdr)

log("  Complementarity rules loaded for all 20 amino acids")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Design 5 nanobody candidates
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 4: Designing 5 nanobody CDR sequences...")

# Candidate design rationale:
# NbLasR-1: CDR3 targets Epitope1 (KDSQDYEN) — short CDR3
# NbLasR-2: CDR3 targets Epitope1 — extended CDR3 with aromatic anchor
# NbLasR-3: CDR3 targets Epitope2 (EHYDRAGYARVDPTV) — long CDR3
# NbLasR-4: CDR3 dual-targeting loop (both epitopes motif)
# NbLasR-5: CDR3 targets Epitope2 with hydrophobic core

# Manually crafted CDRs using complementarity + known anti-bacterial nanobody CDR data
CANDIDATES = {
    "NbLasR-1": {
        "target":  "Epitope1 (KDSQDYEN) — core LBD",
        "strategy":"Short CDR3 (8aa), complementary charges to K73-D-N cluster",
        "CDR1":    "GSTFSNYA",
        "CDR2":    "ISSSGST",
        "CDR3":    "DRWYNKYD",
        "rationale": "D,R complement K,E in epitope; W,Y provide aromatic stacking with Tyr in KDSQDYEN"
    },
    "NbLasR-2": {
        "target":  "Epitope1 (KDSQDYEN) — extended loop",
        "strategy":"Extended CDR3 (11aa), aromatic triad anchoring",
        "CDR1":    "GRTFSRYA",
        "CDR2":    "ISSRGST",
        "CDR3":    "RYDKEWNRYDW",
        "rationale": "R,K residues complement D73,E in epitope; double YW aromatic anchoring; D,N H-bond donors"
    },
    "NbLasR-3": {
        "target":  "Epitope2 (EHYDRAGYARVDPTV) — hydrophobic pocket",
        "strategy":"Long CDR3 (14aa) matching extended hydrophobic surface",
        "CDR1":    "GSTFSSYA",
        "CDR2":    "IYSNGST",
        "CDR3":    "RKWYFDLRWYFDLR",
        "rationale": "R,K neutralize E,D in epitope; W,Y,F mirror aromatic pocket of HYDRAGY motif; L fills hydrophobic V,I pockets"
    },
    "NbLasR-4": {
        "target":  "Dual — Epitope1 + Epitope2 bridging",
        "strategy":"CDR3 contains both complementary motifs in tandem",
        "CDR1":    "GFTFSRNA",
        "CDR2":    "ISSDGST",
        "CDR3":    "DRYWNKYWFDR",
        "rationale": "DRYWN motif addresses KDSQD; KWFD motif complements YARVD; bridging loop covers both epitope patches"
    },
    "NbLasR-5": {
        "target":  "Epitope2 (EHYDRAGYARVDPTV) — aromatic triad",
        "strategy":"CDR3 aromatic-rich (13aa), mimics known QS-targeting VHH CDRs",
        "CDR1":    "GSTFSNYA",
        "CDR2":    "ISSRGST",
        "CDR3":    "RWYKYRDWYKFRD",
        "rationale": "Triple W,Y aromatic cluster for HAGY aromatic core; R,K for acidic E,D; F for hydrophobic V,P"
    }
}

def build_nanobody(cdr1, cdr2, cdr3):
    return VHH_FR1 + cdr1 + VHH_FR2 + cdr2 + VHH_FR3 + cdr3 + VHH_FR4

nb_sequences = {}
for name, info in CANDIDATES.items():
    seq = build_nanobody(info["CDR1"], info["CDR2"], info["CDR3"])
    nb_sequences[name] = seq
    log(f"  {name}: {len(seq)} aa | CDR3={info['CDR3']} | Target: {info['target'][:40]}")

# Save all sequences as FASTA
fasta_path = os.path.join(PHASE3, "nanobody_candidates.fasta")
with open(fasta_path, "w") as f:
    for name, seq in nb_sequences.items():
        info = CANDIDATES[name]
        f.write(f">{name} | {info['target']} | CDR1:{info['CDR1']} CDR2:{info['CDR2']} CDR3:{info['CDR3']}\n")
        f.write(seq + "\n\n")
log(f"  Nanobody FASTA saved → {fasta_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Physicochemical analysis of all candidates
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 5: Computing physicochemical properties...")

physchem_data = []
for name, seq in nb_sequences.items():
    try:
        pa  = ProteinAnalysis(seq)
        mw  = pa.molecular_weight()
        pi  = pa.isoelectric_point()
        gra = pa.gravy()
        ins = pa.instability_index()
        arm = pa.aromaticity()
        hel = pa.secondary_structure_fraction()[0]  # helix
        sht = pa.secondary_structure_fraction()[1]  # turn
        coi = pa.secondary_structure_fraction()[2]  # coil

        # Count CDR residue types
        cdr3 = CANDIDATES[name]["CDR3"]
        cdr3_analysis = ProteinAnalysis(cdr3)
        cdr3_charge   = cdr3_analysis.charge_at_pH(7.4)
        cdr3_aromatic = sum(1 for aa in cdr3 if aa in "YWF") / len(cdr3)

        row = {
            "Candidate":   name,
            "Target":      CANDIDATES[name]["target"][:35],
            "Length":      len(seq),
            "MW_kDa":      round(mw/1000, 2),
            "pI":          round(pi, 2),
            "GRAVY":       round(gra, 3),
            "Instability": round(ins, 2),
            "Aromaticity": round(arm, 3),
            "CDR1":        CANDIDATES[name]["CDR1"],
            "CDR2":        CANDIDATES[name]["CDR2"],
            "CDR3":        CANDIDATES[name]["CDR3"],
            "CDR3_len":    len(CANDIDATES[name]["CDR3"]),
            "CDR3_charge": round(cdr3_charge, 2),
            "CDR3_aromatic_frac": round(cdr3_aromatic, 3),
            "Stable":      "YES" if ins < 40 else "BORDERLINE" if ins < 50 else "UNSTABLE",
            "Helix_frac":  round(hel, 3),
            "Turn_frac":   round(sht, 3),
            "Coil_frac":   round(coi, 3)
        }
        physchem_data.append(row)
        log(f"  {name}: MW={mw/1000:.1f}kDa | pI={pi:.2f} | GRAVY={gra:.3f} | "
            f"Instability={ins:.1f} ({row['Stable']}) | CDR3_charge={cdr3_charge:.1f}")
    except Exception as e:
        log(f"  {name}: analysis error — {e}")

df_phys = pd.DataFrame(physchem_data)
phys_csv = os.path.join(PHASE3, "nanobody_physicochemical.csv")
df_phys.to_csv(phys_csv, index=False)
log(f"  Physicochemical table saved → {phys_csv}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: CDR composition analysis
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 6: Analyzing CDR amino acid composition...")

aa_groups = {
    "Aromatic":   "YWFH",
    "Positive":   "RK",
    "Negative":   "DE",
    "Polar":      "STNQ",
    "Hydrophobic":"ILVAM",
    "Special":    "CGP"
}

cdr3_composition = []
for name, info in CANDIDATES.items():
    cdr3 = info["CDR3"]
    row  = {"Candidate": name, "CDR3": cdr3, "Length": len(cdr3)}
    for grp, aas in aa_groups.items():
        count = sum(1 for aa in cdr3 if aa in aas)
        row[grp]        = count
        row[f"{grp}_%"] = round(100 * count / len(cdr3), 1)
    cdr3_composition.append(row)

df_cdr = pd.DataFrame(cdr3_composition)
cdr_csv = os.path.join(PHASE3, "CDR3_composition.csv")
df_cdr.to_csv(cdr_csv, index=False)
log(f"  CDR3 composition table saved → {cdr_csv}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: ESMFold structure prediction (Meta API)
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 7: Submitting nanobody sequences to ESMFold API...")

ESMFOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
plddt_results = {}

for name, seq in nb_sequences.items():
    pdb_out = os.path.join(PHASE3, f"{name}_ESMFold.pdb")
    if os.path.exists(pdb_out):
        log(f"  {name}: PDB already exists, loading...")
        with open(pdb_out) as f:
            pdb_txt = f.read()
    else:
        try:
            log(f"  {name}: Submitting to ESMFold ({len(seq)} aa)...")
            resp = requests.post(
                ESMFOLD_URL,
                data=seq,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=120
            )
            if resp.status_code == 200:
                pdb_txt = resp.text
                with open(pdb_out, "w") as f:
                    f.write(pdb_txt)
                log(f"  {name}: ESMFold SUCCESS → {pdb_out}")
            else:
                log(f"  {name}: ESMFold HTTP {resp.status_code} — {resp.text[:100]}")
                pdb_txt = None
        except Exception as e:
            log(f"  {name}: ESMFold error — {e}")
            pdb_txt = None

    # Parse pLDDT from B-factor column
    if os.path.exists(pdb_out):
        plddt_vals = []
        with open(pdb_out) as f:
            for line in f:
                if line.startswith("ATOM"):
                    try:
                        bfac = float(line[60:66].strip())
                        plddt_vals.append(bfac)
                    except:
                        pass
        if plddt_vals:
            plddt_results[name] = {
                "mean":   round(np.mean(plddt_vals),  2),
                "median": round(np.median(plddt_vals),2),
                "min":    round(np.min(plddt_vals),   2),
                "max":    round(np.max(plddt_vals),   2),
                "high_conf":  int(np.sum(np.array(plddt_vals) > 70)),
                "very_high":  int(np.sum(np.array(plddt_vals) > 90)),
                "n_atoms":    len(plddt_vals),
                "values":     plddt_vals
            }
            log(f"  {name}: pLDDT mean={plddt_results[name]['mean']:.1f} | "
                f">70: {plddt_results[name]['high_conf']} atoms | "
                f">90: {plddt_results[name]['very_high']} atoms")
    time.sleep(1.5)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: Extract per-residue pLDDT from ESMFold PDBs
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 8: Extracting per-residue pLDDT profiles...")

plddt_per_residue = {}
for name in nb_sequences:
    pdb_out = os.path.join(PHASE3, f"{name}_ESMFold.pdb")
    if not os.path.exists(pdb_out):
        continue
    res_plddt = {}
    with open(pdb_out) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    rnum  = int(line[22:26].strip())
                    bfac  = float(line[60:66].strip())
                    res_plddt[rnum] = bfac
                except:
                    pass
    if res_plddt:
        plddt_per_residue[name] = res_plddt
        log(f"  {name}: {len(res_plddt)} CA residues parsed")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9: CDR region identification in ESMFold structures
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 9: Mapping CDR positions in full nanobody sequences...")

cdr_positions = {}
for name, info in CANDIDATES.items():
    seq   = nb_sequences[name]
    cdr1  = info["CDR1"]
    cdr2  = info["CDR2"]
    cdr3  = info["CDR3"]
    c1s   = seq.find(cdr1) + 1   # 1-indexed
    c2s   = seq.find(cdr2) + 1
    c3s   = seq.find(cdr3) + 1
    cdr_positions[name] = {
        "CDR1": (c1s, c1s + len(cdr1) - 1),
        "CDR2": (c2s, c2s + len(cdr2) - 1),
        "CDR3": (c3s, c3s + len(cdr3) - 1)
    }
    log(f"  {name}: CDR1={c1s}-{c1s+len(cdr1)-1}, CDR2={c2s}-{c2s+len(cdr2)-1}, CDR3={c3s}-{c3s+len(cdr3)-1}")

cdr_pos_path = os.path.join(PHASE3, "CDR_positions.json")
with open(cdr_pos_path, "w") as f:
    json.dump(cdr_positions, f, indent=2)
log(f"  CDR positions saved → {cdr_pos_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 10: Generating Phase 3 figures...")
plt.style.use('seaborn-v0_8-whitegrid')
NB_COLORS = {
    "NbLasR-1": COLORS["accent"],
    "NbLasR-2": COLORS["secondary"],
    "NbLasR-3": COLORS["green"],
    "NbLasR-4": COLORS["purple"],
    "NbLasR-5": COLORS["gold"]
}

# ─── Figure 16: Nanobody candidate summary table ─────────────────────────────
fig, ax = plt.subplots(figsize=(18, 5))
ax.axis('off')
disp_cols = ["Candidate","Length","MW_kDa","pI","GRAVY","Instability","Stable",
             "CDR3","CDR3_len","CDR3_charge","CDR3_aromatic_frac"]
disp_labs = ["Candidate","Length\n(aa)","MW\n(kDa)","pI","GRAVY","Instability\nIndex",
             "Stable?","CDR3 Sequence","CDR3\nLen","CDR3\nCharge","CDR3\nAromatic"]
data_show = df_phys[disp_cols].values.tolist()
data_show = [[str(round(v,3)) if isinstance(v,(float,np.floating)) else str(v) for v in row]
             for row in data_show]
tbl = ax.table(cellText=data_show, colLabels=disp_labs, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1.15, 2.2)
for j in range(len(disp_labs)):
    tbl[0,j].set_facecolor(COLORS["primary"])
    tbl[0,j].set_text_props(color='white', fontweight='bold')
row_colors = [COLORS["accent"], COLORS["secondary"], COLORS["green"], COLORS["purple"], COLORS["gold"]]
for i in range(1, len(data_show)+1):
    for j in range(len(disp_labs)):
        tbl[i,j].set_facecolor(row_colors[i-1] + "33")  # 20% alpha hex
    stable = data_show[i-1][6]
    tbl[i,6].set_facecolor("#D5F5E3" if stable == "YES" else "#FEF9E7" if stable == "BORDERLINE" else "#FADBD8")
ax.set_title("Nanobody Candidate Summary — Phase 3 Antibody Design",
             fontsize=13, fontweight='bold', color=COLORS["primary"], pad=15)
plt.tight_layout()
fig.savefig(os.path.join(FIGS,"Fig16_nanobody_summary_table.png"), dpi=150, bbox_inches='tight')
plt.close()
log("  Fig16_nanobody_summary_table.png saved")

# ─── Figure 17: pLDDT profiles per candidate ─────────────────────────────────
if plddt_per_residue:
    n_cols = min(3, len(plddt_per_residue))
    n_rows = (len(plddt_per_residue) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 4*n_rows))
    axes = np.array(axes).flatten() if n_rows*n_cols > 1 else [axes]
    for ax, (name, res_data) in zip(axes, plddt_per_residue.items()):
        col   = NB_COLORS.get(name, COLORS["primary"])
        rpos  = sorted(res_data.keys())
        vals  = [res_data[r] for r in rpos]
        ax.fill_between(rpos, vals, alpha=0.25, color=col)
        ax.plot(rpos, vals, color=col, linewidth=1.2)
        ax.axhline(90, color=COLORS["green"],   linewidth=1.2, linestyle='--', label='Very high (90)')
        ax.axhline(70, color=COLORS["gold"],    linewidth=1.2, linestyle='--', label='High (70)')
        ax.axhline(50, color=COLORS["accent"],  linewidth=1.2, linestyle='--', label='Low (50)')
        # Shade CDR regions
        if name in cdr_positions:
            cdr_shade = [("CDR1","#FF6B6B"), ("CDR2","#4ECDC4"), ("CDR3","#45B7D1")]
            for cdr_name, cdr_col in cdr_shade:
                if cdr_name in cdr_positions[name]:
                    s, e = cdr_positions[name][cdr_name]
                    ax.axvspan(s, e, alpha=0.2, color=cdr_col, label=cdr_name)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Residue Position", fontsize=9)
        ax.set_ylabel("pLDDT", fontsize=9)
        ax.set_title(f"{name}\nMean pLDDT={plddt_results.get(name,{}).get('mean','N/A')}",
                     fontsize=10, fontweight='bold', color=col)
        ax.legend(fontsize=7, loc='lower right')
    for ax in axes[len(plddt_per_residue):]:
        ax.axis('off')
    fig.suptitle("ESMFold pLDDT Confidence Profiles — All Nanobody Candidates\n"
                 "(CDR regions shaded: CDR1=red, CDR2=teal, CDR3=blue)",
                 fontsize=13, fontweight='bold', color=COLORS["primary"])
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS,"Fig17_pLDDT_profiles.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log("  Fig17_pLDDT_profiles.png saved")

# ─── Figure 18: pLDDT comparison bar chart ───────────────────────────────────
if plddt_results:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names_list = list(plddt_results.keys())
    means      = [plddt_results[n]["mean"]      for n in names_list]
    high_c     = [plddt_results[n]["high_conf"] for n in names_list]
    very_h     = [plddt_results[n]["very_high"] for n in names_list]
    bar_colors = [NB_COLORS.get(n, COLORS["primary"]) for n in names_list]

    bars = axes[0].bar(names_list, means, color=bar_colors, edgecolor='black', linewidth=0.7, width=0.6)
    axes[0].axhline(70, color=COLORS["gold"],  linewidth=2, linestyle='--', label='High confidence (70)')
    axes[0].axhline(90, color=COLORS["green"], linewidth=2, linestyle='--', label='Very high (90)')
    axes[0].set_ylabel("Mean pLDDT Score", fontsize=11)
    axes[0].set_title("Mean pLDDT per Candidate\n(ESMFold)", fontsize=12, fontweight='bold',
                      color=COLORS["primary"])
    for bar, val in zip(bars, means):
        axes[0].text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.3,
                     f"{val:.1f}", ha='center', fontsize=10, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, 105)

    x = np.arange(len(names_list))
    w = 0.35
    b1 = axes[1].bar(x - w/2, high_c, w, color=[c+"AA" for c in bar_colors],
                     edgecolor='black', linewidth=0.7, label='>70 (High conf.)')
    b2 = axes[1].bar(x + w/2, very_h, w, color=bar_colors,
                     edgecolor='black', linewidth=0.7, label='>90 (Very high)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names_list, fontsize=9)
    axes[1].set_ylabel("Number of Atoms", fontsize=11)
    axes[1].set_title("High-Confidence Atoms per Candidate\n(ESMFold pLDDT)",
                      fontsize=12, fontweight='bold', color=COLORS["primary"])
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS,"Fig18_pLDDT_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log("  Fig18_pLDDT_comparison.png saved")

# ─── Figure 19: MW / pI scatter ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (x_col, y_col, xl, yl) in zip(axes, [
    ("MW_kDa",    "pI",          "Molecular Weight (kDa)", "Isoelectric Point (pI)"),
    ("GRAVY",     "Instability", "GRAVY Score",             "Instability Index")
]):
    for _, row in df_phys.iterrows():
        col = NB_COLORS.get(row["Candidate"], COLORS["primary"])
        ax.scatter(row[x_col], row[y_col], color=col, s=200, zorder=3,
                   edgecolors='black', linewidth=1.0)
        ax.annotate(row["Candidate"], (row[x_col], row[y_col]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8.5, color=col)
    ax.set_xlabel(xl, fontsize=11)
    ax.set_ylabel(yl, fontsize=11)
    ax.set_title(f"{xl} vs {yl}", fontsize=11, fontweight='bold', color=COLORS["primary"])
    if y_col == "Instability":
        ax.axhline(40, color=COLORS["gold"],  linewidth=1.5, linestyle='--', label='Stable threshold (40)')
        ax.legend(fontsize=9)
fig.suptitle("Nanobody Physicochemical Properties Comparison", fontsize=13,
             fontweight='bold', color=COLORS["primary"])
plt.tight_layout()
fig.savefig(os.path.join(FIGS,"Fig19_physicochemical_scatter.png"), dpi=150, bbox_inches='tight')
plt.close()
log("  Fig19_physicochemical_scatter.png saved")

# ─── Figure 20: CDR3 composition stacked bar ─────────────────────────────────
grp_cols = ["Aromatic", "Positive", "Negative", "Polar", "Hydrophobic", "Special"]
grp_cols_pct = [f"{g}_%" for g in grp_cols]
grp_colors   = [COLORS["purple"], COLORS["secondary"], COLORS["accent"],
                COLORS["green"],  COLORS["gold"],       COLORS["primary"]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
x  = np.arange(len(df_cdr))
bw = 0.6
bottom = np.zeros(len(df_cdr))
for grp, col in zip(grp_cols_pct, grp_colors):
    vals = df_cdr[grp].values
    ax1.bar(x, vals, bottom=bottom, width=bw, color=col, label=grp.replace("_%",""), edgecolor='white')
    bottom += vals
ax1.set_xticks(x)
ax1.set_xticklabels(df_cdr["Candidate"], fontsize=10, fontweight='bold')
ax1.set_ylabel("CDR3 Residue Composition (%)", fontsize=11)
ax1.set_title("CDR3 Amino Acid Group Composition\nAll Nanobody Candidates",
              fontsize=12, fontweight='bold', color=COLORS["primary"])
ax1.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc='upper left')
ax1.set_ylim(0, 115)

# CDR3 charge vs aromatic fraction
for _, row in df_phys.iterrows():
    col = NB_COLORS.get(row["Candidate"], COLORS["primary"])
    ax2.scatter(row["CDR3_charge"], row["CDR3_aromatic_frac"], color=col,
                s=250, zorder=3, edgecolors='black', linewidth=1.2)
    ax2.annotate(row["Candidate"], (row["CDR3_charge"], row["CDR3_aromatic_frac"]),
                 textcoords="offset points", xytext=(5, 5), fontsize=9, color=col, fontweight='bold')
ax2.axhline(0.3, color=COLORS["gold"], linewidth=1.5, linestyle='--',
            label='Good aromatic content (0.3)')
ax2.axvline(0, color=COLORS["accent"], linewidth=1.5, linestyle=':', label='Neutral charge')
ax2.set_xlabel("CDR3 Net Charge at pH 7.4", fontsize=11)
ax2.set_ylabel("CDR3 Aromatic Residue Fraction", fontsize=11)
ax2.set_title("CDR3 Charge vs Aromatic Content\n(Both favor strong binding)",
              fontsize=12, fontweight='bold', color=COLORS["primary"])
ax2.legend(fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(FIGS,"Fig20_CDR3_composition.png"), dpi=150, bbox_inches='tight')
plt.close()
log("  Fig20_CDR3_composition.png saved")

# ─── Figure 21: Full sequence alignment visualization ────────────────────────
fig, ax = plt.subplots(figsize=(18, 8))
ax.axis('off')
# Find max seq length
max_len = max(len(s) for s in nb_sequences.values())
aa_colors_map = {
    'R':'#4472C4','K':'#4472C4','H':'#70AD47',
    'D':'#FF0000','E':'#FF0000',
    'S':'#ED7D31','T':'#ED7D31','N':'#ED7D31','Q':'#ED7D31',
    'G':'#FFC000','A':'#FFC000',
    'V':'#7030A0','L':'#7030A0','I':'#7030A0','M':'#7030A0',
    'F':'#FF00FF','W':'#FF00FF','Y':'#FF00FF',
    'P':'#808080','C':'#92D050'
}
y_spacing = 1.2
for i, (name, seq) in enumerate(nb_sequences.items()):
    y = (len(nb_sequences) - i - 1) * y_spacing
    ax.text(-2, y, name, fontsize=8, fontweight='bold',
            color=NB_COLORS.get(name, COLORS["primary"]), va='center', ha='right')
    for j, aa in enumerate(seq):
        bg_col = aa_colors_map.get(aa, '#EEEEEE')
        ax.text(j, y, aa, fontsize=5.5, va='center', ha='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.1', facecolor=bg_col, alpha=0.6, linewidth=0))
    # Mark CDRs
    if name in cdr_positions:
        for cdr_n, (cs, ce) in cdr_positions[name].items():
            ax.annotate('', xy=(ce-0.5, y+0.45), xytext=(cs-0.5, y+0.45),
                        arrowprops=dict(arrowstyle='-', color='black', lw=2.5))
            ax.text((cs+ce-1)/2, y+0.52, cdr_n, fontsize=6.5,
                    ha='center', va='bottom', fontweight='bold')
ax.set_xlim(-12, max_len + 2)
ax.set_ylim(-0.8, len(nb_sequences) * y_spacing)
ax.set_title("Nanobody Sequence Alignment — All 5 Candidates\n"
             "(CDR regions marked with brackets; color = residue property)",
             fontsize=13, fontweight='bold', color=COLORS["primary"])
# Legend patches
legend_items = [
    mpatches.Patch(color='#4472C4', label='Basic (R,K,H)'),
    mpatches.Patch(color='#FF0000', label='Acidic (D,E)'),
    mpatches.Patch(color='#ED7D31', label='Polar (S,T,N,Q)'),
    mpatches.Patch(color='#FFC000', label='Small (G,A)'),
    mpatches.Patch(color='#7030A0', label='Hydrophobic (V,L,I,M)'),
    mpatches.Patch(color='#FF00FF', label='Aromatic (F,W,Y)'),
]
ax.legend(handles=legend_items, fontsize=8, loc='lower right',
          bbox_to_anchor=(1.0, 0.0))
plt.tight_layout()
fig.savefig(os.path.join(FIGS,"Fig21_sequence_alignment.png"), dpi=150, bbox_inches='tight')
plt.close()
log("  Fig21_sequence_alignment.png saved")

# ─── Figure 22: Candidate ranking radar chart ────────────────────────────────
categories_radar = ["pLDDT\nConfidence", "Stability", "pI\nSuitability",
                    "Aromatic\nContent", "CDR3\nLength", "Charge\nBalance"]
N = len(categories_radar)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories_radar, size=10)

for name in nb_sequences:
    col   = NB_COLORS.get(name, COLORS["primary"])
    phys  = df_phys[df_phys["Candidate"]==name].iloc[0] if not df_phys[df_phys["Candidate"]==name].empty else None
    if phys is None:
        continue
    plddt_m = plddt_results.get(name, {}).get("mean", 0)
    vals = [
        min(1.0, plddt_m / 100),                             # pLDDT
        1.0 - min(1.0, float(phys["Instability"]) / 100),    # Stability (inverse instability)
        1.0 - abs(float(phys["pI"]) - 7.0) / 7.0,           # pI near neutral = good
        min(1.0, float(phys["CDR3_aromatic_frac"]) * 2.5),   # aromatic content
        min(1.0, int(phys["CDR3_len"]) / 15),                # CDR3 length (longer = more contact)
        1.0 - min(1.0, abs(float(phys["CDR3_charge"])) / 8)  # balanced charge
    ]
    vals = [max(0, v) for v in vals]
    vals += vals[:1]
    ax.plot(angles, vals, color=col, linewidth=2, label=name)
    ax.fill(angles, vals, color=col, alpha=0.1)

ax.set_ylim(0, 1)
ax.set_title("Nanobody Candidate Quality Radar Chart\n(All 6 design criteria)",
             size=13, fontweight='bold', color=COLORS["primary"], y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIGS,"Fig22_candidate_radar.png"), dpi=150, bbox_inches='tight')
plt.close()
log("  Fig22_candidate_radar.png saved")

# ─── Figure 23: PyMOL scripts for ESMFold structures ────────────────────────
for name in nb_sequences:
    pdb_out = os.path.join(PHASE3, f"{name}_ESMFold.pdb")
    if not os.path.exists(pdb_out):
        continue
    cdr_pos = cdr_positions.get(name, {})
    pml_script = f"""# PyMOL visualization — {name} ESMFold structure
load {pdb_out}, {name}
bg_color white
hide everything
show cartoon, {name}

# Color by pLDDT (stored in B-factor)
spectrum b, blue_white_red, {name}, 0, 100

# Show CDR regions
"""
    for cdr_n, (cs, ce) in cdr_pos.items():
        pml_script += f"select {cdr_n}_{name}, {name} and resi {cs}-{ce}\n"
        pml_script += f"show sticks, {cdr_n}_{name}\n"
        cdr_color = {"CDR1": "red", "CDR2": "green", "CDR3": "yellow"}
        pml_script += f"color {cdr_color.get(cdr_n,'orange')}, {cdr_n}_{name}\n"
    pml_script += f"""
# Labels
label {name} and name CA and resi {cdr_pos.get('CDR3',('',''))[0]}, "CDR3-start"
orient {name}
zoom {name}

ray 1000, 800
png {FIGS}/{name}_PyMOL.png, dpi=150
quit
"""
    pml_path = os.path.join(PHASE3, f"{name}_visualize.pml")
    with open(pml_path, "w") as f:
        f.write(pml_script)

# Run PyMOL for best candidate (NbLasR-1 or first available)
import subprocess
for name in nb_sequences:
    pdb_out = os.path.join(PHASE3, f"{name}_ESMFold.pdb")
    pml_path = os.path.join(PHASE3, f"{name}_visualize.pml")
    if os.path.exists(pdb_out) and os.path.exists(pml_path):
        try:
            result = subprocess.run(["pymol","-c",pml_path],
                                    capture_output=True, text=True, timeout=45)
            out_png = os.path.join(FIGS, f"{name}_PyMOL.png")
            if os.path.exists(out_png):
                log(f"  PyMOL render: {name} → {name}_PyMOL.png")
            else:
                log(f"  PyMOL render: {name} completed (check stderr)")
        except Exception as e:
            log(f"  PyMOL render: {name} — {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 11: Compile overall candidate ranking
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 11: Compiling final candidate ranking...")

ranking_data = []
for name in nb_sequences:
    phys = df_phys[df_phys["Candidate"]==name]
    if phys.empty:
        continue
    phys = phys.iloc[0]
    plddt_m = plddt_results.get(name, {}).get("mean", 0)

    # Score components (0-1 scale)
    s_plddt  = min(1.0, plddt_m / 100)
    s_stab   = 1.0 - min(1.0, float(phys["Instability"]) / 60)
    s_pi     = 1.0 - abs(float(phys["pI"]) - 7.0) / 7.0
    s_arom   = min(1.0, float(phys["CDR3_aromatic_frac"]) * 2.5)
    s_cdr3   = min(1.0, int(phys["CDR3_len"]) / 15)
    # Weighted overall
    overall  = 0.35*s_plddt + 0.25*s_stab + 0.15*s_pi + 0.15*s_arom + 0.10*s_cdr3

    ranking_data.append({
        "Candidate":   name,
        "Target":      CANDIDATES[name]["target"][:40],
        "pLDDT_mean":  round(plddt_m, 1),
        "Instability": float(phys["Instability"]),
        "pI":          float(phys["pI"]),
        "CDR3_len":    int(phys["CDR3_len"]),
        "CDR3_aromatic": float(phys["CDR3_aromatic_frac"]),
        "Score_pLDDT": round(s_plddt,  3),
        "Score_Stab":  round(s_stab,   3),
        "Score_pI":    round(s_pi,     3),
        "Score_Arom":  round(s_arom,   3),
        "Score_CDR3":  round(s_cdr3,   3),
        "Overall_Score": round(overall, 3)
    })

df_rank = pd.DataFrame(ranking_data).sort_values("Overall_Score", ascending=False)
df_rank["Final_Rank"] = range(1, len(df_rank)+1)
rank_csv = os.path.join(PHASE3, "nanobody_final_ranking.csv")
df_rank.to_csv(rank_csv, index=False)
log(f"  Final ranking saved → {rank_csv}")

log("\n  FINAL CANDIDATE RANKING:")
for _, row in df_rank.iterrows():
    log(f"    #{int(row['Final_Rank'])} {row['Candidate']}: Overall={row['Overall_Score']:.3f} | "
        f"pLDDT={row['pLDDT_mean']:.1f} | Stability={row['Score_Stab']:.2f} | "
        f"CDR3={CANDIDATES[row['Candidate']]['CDR3']}")

# ─── Figure 24: Final ranking bar chart ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
score_cols = ["Score_pLDDT","Score_Stab","Score_pI","Score_Arom","Score_CDR3"]
score_labs = ["pLDDT\n(35%)","Stability\n(25%)","pI fit\n(15%)","Aromatic\n(15%)","CDR3 len\n(10%)"]
score_colors_bar = [COLORS["secondary"],COLORS["green"],COLORS["gold"],
                    COLORS["purple"],COLORS["primary"]]
x   = np.arange(len(df_rank))
bw  = 0.13
for k, (col, lab, c) in enumerate(zip(score_cols, score_labs, score_colors_bar)):
    ax.bar(x + k*bw, df_rank[col].values, bw, label=lab, color=c, edgecolor='black', linewidth=0.5)
ax.set_xticks(x + bw*2)
ax.set_xticklabels(df_rank["Candidate"], fontsize=10, fontweight='bold')
ax.set_ylabel("Score (0–1)", fontsize=11)
ax.set_title("Nanobody Candidate Scoring — All Criteria\n(Ranked by Overall Score)",
             fontsize=13, fontweight='bold', color=COLORS["primary"])
ax.legend(fontsize=9, bbox_to_anchor=(1.01,1), loc='upper left')
# Overall score as line
ax2 = ax.twinx()
ax2.plot(x + bw*2, df_rank["Overall_Score"].values, 'k-o', linewidth=2,
         markersize=8, label="Overall Score", zorder=5)
ax2.set_ylabel("Overall Score", fontsize=11)
ax2.set_ylim(0, 1)
ax2.legend(fontsize=10, loc='upper right')
plt.tight_layout()
fig.savefig(os.path.join(FIGS,"Fig24_candidate_scoring.png"), dpi=150, bbox_inches='tight')
plt.close()
log("  Fig24_candidate_scoring.png saved")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 12: Write Phase 3 report
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 12: Writing Phase 3 summary report...")

top_cand = df_rank.iloc[0]["Candidate"] if not df_rank.empty else "NbLasR-1"

report = [
    "=" * 70,
    "PHASE 3 SUMMARY REPORT — In Silico Nanobody Design",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "=" * 70,
    "",
    "1. TARGET EPITOPES (from Phase 2)",
    "-" * 40,
    f"   Epitope 1: {EPI1_SEQ}  (LBD core — Asp73 region)",
    f"   Epitope 2: {EPI2_SEQ}  (LBD hydrophobic pocket)",
    "",
    "2. VHH FRAMEWORK",
    "-" * 40,
    f"   FR1: {VHH_FR1}",
    f"   FR2: {VHH_FR2}",
    f"   FR3: {VHH_FR3}",
    f"   FR4: {VHH_FR4}",
    "",
    "3. DESIGNED CANDIDATES",
    "-" * 40,
]
for name, info in CANDIDATES.items():
    phys = df_phys[df_phys["Candidate"]==name].iloc[0] if not df_phys[df_phys["Candidate"]==name].empty else None
    plddt_m = plddt_results.get(name, {}).get("mean", 0)
    report += [
        f"",
        f"   {name}:",
        f"     Target:   {info['target']}",
        f"     Strategy: {info['strategy']}",
        f"     CDR1: {info['CDR1']}  CDR2: {info['CDR2']}  CDR3: {info['CDR3']}",
        f"     Full seq length: {len(nb_sequences.get(name,''))} aa",
        f"     MW: {float(phys['MW_kDa']):.1f} kDa | pI: {float(phys['pI']):.2f} | "
        f"Instability: {float(phys['Instability']):.1f}" if phys is not None else "     Properties: N/A",
        f"     ESMFold pLDDT: {plddt_m:.1f}" if plddt_m else "     ESMFold pLDDT: pending",
        f"     Rationale: {info['rationale']}"
    ]
report += [
    "",
    "4. FINAL RANKING",
    "-" * 40,
]
for _, row in df_rank.iterrows():
    report.append(
        f"   #{int(row['Final_Rank'])} {row['Candidate']}: Overall={row['Overall_Score']:.3f} | "
        f"pLDDT={row['pLDDT_mean']:.1f} | Stab={row['Score_Stab']:.2f}"
    )
report += [
    "",
    f"5. TOP CANDIDATE: {top_cand}",
    "-" * 40,
    f"   CDR3: {CANDIDATES[top_cand]['CDR3']}",
    f"   Target: {CANDIDATES[top_cand]['target']}",
    "",
    "6. OUTPUT FILES",
    "-" * 40,
    "   FASTA: nanobody_candidates.fasta",
    "   CSV:   nanobody_physicochemical.csv",
    "   CSV:   CDR3_composition.csv",
    "   CSV:   nanobody_final_ranking.csv",
    "   JSON:  CDR_positions.json",
    "   PDB:   NbLasR-[1-5]_ESMFold.pdb (if API available)",
    "   PML:   NbLasR-[1-5]_visualize.pml",
    "   Figs:  Fig16–Fig24 in 08_Results_Figures/Phase3/",
    "",
    "7. NEXT STEP → PHASE 4: Protein-Protein Docking",
    "   ClusPro 2.0, HADDOCK restraints, interface analysis",
    "=" * 70,
]
rpt_path = os.path.join(PHASE3, "PHASE3_REPORT.txt")
with open(rpt_path, "w") as f:
    f.write("\n".join(report))
log(f"  Phase 3 report saved → {rpt_path}")

results_json = {
    "phase": 3,
    "title": "In Silico Nanobody Design",
    "completed": datetime.now().isoformat(),
    "epitope1": EPI1_SEQ,
    "epitope2": EPI2_SEQ,
    "candidates_designed": list(nb_sequences.keys()),
    "esmfold_predictions": list(plddt_results.keys()),
    "top_candidate": top_cand,
    "plddt_summary": {k: {"mean": v["mean"], "high_conf": v["high_conf"]}
                      for k, v in plddt_results.items()},
    "final_ranking": df_rank[["Candidate","Overall_Score","Final_Rank"]].to_dict("records"),
    "figures_generated": 9,
    "status": "COMPLETED"
}
with open(os.path.join(PHASE3, "phase3_results.json"), "w") as f:
    json.dump(results_json, f, indent=2)

log("\n" + "=" * 70)
log("PHASE 3: COMPLETED SUCCESSFULLY")
log(f"  Figures: {FIGS}")
log(f"  Results: {PHASE3}")
log(f"  Log:     {LOG_FILE}")
log("=" * 70)
