"""
PHASE 7 — Immunogenicity & Developability Assessment
LasR Nanobody Candidates (NbLasR-1 to NbLasR-5)

Pipeline:
  1. Physicochemical properties (BioPython ProtParam + extended)
  2. T-cell epitope / immunogenicity risk (SYFPEITHI-like 9-mer + 15-mer scoring)
  3. Aggregation-prone region (APR) detection (TANGO-inspired)
  4. Solubility & CamSol-lite score
  5. Cross-reactivity — NCBI BLAST CDR3 vs Homo sapiens proteome
  6. Developability scorecard (composite)
  7. Publication-quality figures (12 panels)
  8. PHASE7_REPORT.txt + CSV results
"""

import os, sys, json, csv, time, re, logging, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import requests
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Blast import NCBIWWW, NCBIXML

# ── paths ─────────────────────────────────────────────────────────────────────
BASE   = "/home/stalin/Desktop/LasR_Antibody_Design"
OUTDIR = f"{BASE}/07_Immunogenicity_ADMET"
FIGDIR = f"{OUTDIR}/figures"
os.makedirs(FIGDIR, exist_ok=True)

LOG_FILE = f"{BASE}/logs/phase7_immunogenicity_{time.strftime('%Y%m%d_%H%M%S')}.log"
os.makedirs(f"{BASE}/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ── nanobody data ─────────────────────────────────────────────────────────────
NANOBODIES = {
    "NbLasR-1": {
        "seq":    "QVQLVESGGGLVQAGGSLRLSCAASGSTFSNYAWYRQAPGKQRELVSISSSGSTRFTISRDNAKNTVYLQMNSLKPEDTAVYYCDRWYNKYDWGQGTQVTVSS",
        "CDR1":   "GSTFSNYA", "CDR2": "ISSSGST", "CDR3": "DRWYNKYD",
        "target": "Epitope1 (KDSQDYEN)", "MW_kDa": 11.3, "pI_phase3": 9.01,
    },
    "NbLasR-2": {
        "seq":    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSRYAWYRQAPGKQRELVSISSRGSTRFTISRDNAKNTVYLQMNSLKPEDTAVYYCRYDKEWNRYDWWGQGTQVTVSS",
        "CDR1":   "GRTFSRYA", "CDR2": "ISSRGST", "CDR3": "RYDKEWNRYDW",
        "target": "Epitope1 extended", "MW_kDa": 12.0, "pI_phase3": 9.64,
    },
    "NbLasR-3": {
        "seq":    "QVQLVESGGGLVQAGGSLRLSCAASGSTFSSYAWYRQAPGKQRELVSIYSNGSTRFTISRDNAKNTVYLQMNSLKPEDTAVYYCRKWYFDLRWYFDLRWGQGTQVTVSS",
        "CDR1":   "GSTFSSYA", "CDR2": "IYSNGST", "CDR3": "RKWYFDLRWYFDLR",
        "target": "Epitope2 hydrophobic", "MW_kDa": 12.3, "pI_phase3": 9.43,
    },
    "NbLasR-4": {
        "seq":    "QVQLVESGGGLVQAGGSLRLSCAASGFTFSRNAWYRQAPGKQRELVSISSDGSTRFTISRDNAKNTVYLQMNSLKPEDTAVYYCDRYWNKYWFDRWGQGTQVTVSS",
        "CDR1":   "GFTFSRNA", "CDR2": "ISSDGST", "CDR3": "DRYWNKYWFDR",
        "target": "Dual Ep1+2 bridge", "MW_kDa": 11.9, "pI_phase3": 9.30,
    },
    "NbLasR-5": {
        "seq":    "QVQLVESGGGLVQAGGSLRLSCAASGSTFSNYAWYRQAPGKQRELVSISSRGSTRFTISRDNAKNTVYLQMNSLKPEDTAVYYCRWYKYRDWYKFRDWGQGTQVTVSS",
        "CDR1":   "GSTFSNYA", "CDR2": "ISSRGST", "CDR3": "RWYKYRDWYKFRD",
        "target": "Epitope2 aromatic", "MW_kDa": 12.2, "pI_phase3": 9.69,
    },
}

# ── amino acid property scales ─────────────────────────────────────────────────
# Kyte-Doolittle hydrophobicity
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
      'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
      'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}

# Aggregation propensity (Zyggregator/TANGO-inspired cross-beta propensity)
AGG = {'A':0.06,'R':-0.70,'N':-0.53,'D':-0.79,'C':0.38,'Q':-0.52,'E':-1.04,
       'G':-0.01,'H':-0.26,'I':1.20,'L':1.22,'K':-0.78,'M':0.48,'F':1.56,
       'P':-0.49,'S':-0.37,'T':-0.08,'W':1.37,'Y':0.97,'V':1.01}

# MHC-II binding propensity at P1 anchor (simplified Rammensee scale)
MHC2_ANCHOR = {'A':0.3,'R':-0.5,'N':-0.3,'D':-0.8,'C':0.4,'Q':-0.1,'E':-0.7,
               'G':0.0,'H':0.1,'I':1.2,'L':1.5,'K':-0.6,'M':0.9,'F':1.4,
               'P':-0.8,'S':-0.2,'T':-0.1,'W':1.0,'Y':0.8,'V':1.1}

# Immunogenicity scale (Calis et al. 2013, simplified 9-mer position weights)
IMMUNO_SCALE = {'A':0.0,'R':-0.2,'N':-0.1,'D':-0.3,'C':0.1,'Q':-0.1,'E':-0.3,
                'G':0.0,'H':0.0,'I':0.4,'L':0.5,'K':-0.2,'M':0.3,'F':0.5,
                'P':-0.2,'S':-0.1,'T':0.0,'W':0.4,'Y':0.3,'V':0.4}


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 1 — Physicochemical Properties
# ══════════════════════════════════════════════════════════════════════════════
def compute_physicochemical(nb_name, info):
    seq = info["seq"]
    pa  = ProteinAnalysis(seq)

    mw          = pa.molecular_weight()
    pI          = pa.isoelectric_point()
    instability = pa.instability_index()
    gravy       = pa.gravy()
    aromaticity = pa.aromaticity()
    # aliphatic index: Ala + 2.9*Val + 3.9*(Ile+Leu) per 100 aa (Ikai 1980)
    aa_count = pa.count_amino_acids()
    n = len(seq)
    aliphatic = (aa_count['A'] + 2.9*aa_count['V'] + 3.9*(aa_count['I']+aa_count['L'])) / n * 100

    # net charge at pH 7.4
    charge_74 = pa.charge_at_pH(7.4)

    # % charged residues
    charged = sum(seq.count(aa) for aa in 'RKHDE') / n * 100
    # % hydrophobic
    hydrophob = sum(seq.count(aa) for aa in 'VILMFYW') / n * 100

    # extinction coefficient
    try:
        ec_reduced, ec_cystine = pa.molar_extinction_coefficient()
    except:
        ec_reduced = ec_cystine = 0

    return {
        "MW_Da":         round(mw, 2),
        "MW_kDa":        round(mw/1000, 3),
        "pI":            round(pI, 3),
        "instability":   round(instability, 2),
        "stability_class": "Stable" if instability < 40 else "Unstable",
        "GRAVY":         round(gravy, 4),
        "aromaticity":   round(aromaticity, 4),
        "aliphatic_idx": round(aliphatic, 2),
        "charge_pH74":   round(charge_74, 3),
        "pct_charged":   round(charged, 2),
        "pct_hydrophob": round(hydrophob, 2),
        "ext_coeff_red": ec_reduced,
        "length":        n,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 2 — T-Cell Epitope / Immunogenicity Prediction
# ══════════════════════════════════════════════════════════════════════════════
def sliding_window_score(seq, scale, window):
    """Mean scale score over sliding windows."""
    scores = []
    for i in range(len(seq) - window + 1):
        pep = seq[i:i+window]
        s = sum(scale.get(aa, 0) for aa in pep) / window
        scores.append((i+1, pep, round(s, 4)))
    return scores


def compute_immunogenicity(nb_name, info):
    seq = info["seq"]

    # MHC-I: 9-mer immunogenicity (Calis et al. inspired)
    mhc1_scores = sliding_window_score(seq, IMMUNO_SCALE, 9)
    mhc1_high   = [s for s in mhc1_scores if s[2] > 0.3]
    mhc1_risk   = len(mhc1_high) / max(len(mhc1_scores), 1) * 100

    # MHC-II: 15-mer binding (hydrophobicity-based P1 anchor)
    mhc2_scores = sliding_window_score(seq, MHC2_ANCHOR, 15)
    mhc2_high   = [s for s in mhc2_scores if s[2] > 0.6]
    mhc2_risk   = len(mhc2_high) / max(len(mhc2_scores), 1) * 100

    # CDR3-specific immunogenicity
    cdr3 = info["CDR3"]
    cdr3_hydro = sum(KD.get(aa, 0) for aa in cdr3) / len(cdr3)
    cdr3_immuno = sum(IMMUNO_SCALE.get(aa, 0) for aa in cdr3) / len(cdr3)

    # Overall immunogenicity risk class
    composite = 0.4*mhc1_risk + 0.4*mhc2_risk + 0.2*(cdr3_immuno*20)
    if composite < 15:
        risk_class = "Low"
    elif composite < 30:
        risk_class = "Medium"
    else:
        risk_class = "High"

    log.info(f"  {nb_name}: MHC-I risk={mhc1_risk:.1f}%  MHC-II risk={mhc2_risk:.1f}%  "
             f"CDR3 hydro={cdr3_hydro:.2f}  Class={risk_class}")

    return {
        "mhc1_high_peptides":  len(mhc1_high),
        "mhc1_risk_pct":       round(mhc1_risk, 2),
        "mhc1_mean_score":     round(np.mean([s[2] for s in mhc1_scores]), 4),
        "mhc1_max_score":      round(max(s[2] for s in mhc1_scores), 4),
        "mhc2_high_peptides":  len(mhc2_high),
        "mhc2_risk_pct":       round(mhc2_risk, 2),
        "mhc2_mean_score":     round(np.mean([s[2] for s in mhc2_scores]), 4),
        "mhc2_max_score":      round(max(s[2] for s in mhc2_scores), 4),
        "cdr3_hydrophobicity": round(cdr3_hydro, 4),
        "cdr3_immuno_score":   round(cdr3_immuno, 4),
        "immuno_composite":    round(composite, 2),
        "immunogenicity_risk": risk_class,
        "top_mhc1_peptide":    mhc1_high[0][1] if mhc1_high else "None",
        "top_mhc2_peptide":    mhc2_high[0][1] if mhc2_high else "None",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 3 — Aggregation-Prone Region (APR) Detection
# ══════════════════════════════════════════════════════════════════════════════
def compute_aggregation(nb_name, info):
    seq = info["seq"]

    # TANGO-inspired: score each 5-mer for aggregation propensity
    window = 5
    agg_scores = []
    for i in range(len(seq) - window + 1):
        pep = seq[i:i+window]
        s = sum(AGG.get(aa, 0) for aa in pep) / window
        agg_scores.append((i+1, pep, round(s, 4)))

    apr_threshold = 0.5
    aprs = [s for s in agg_scores if s[2] > apr_threshold]
    # merge overlapping APRs into regions
    apr_regions = []
    if aprs:
        cur_start, cur_end = aprs[0][0], aprs[0][0]+window
        for pos, pep, sc in aprs[1:]:
            if pos <= cur_end:
                cur_end = pos + window
            else:
                apr_regions.append((cur_start, cur_end))
                cur_start, cur_end = pos, pos + window
        apr_regions.append((cur_start, cur_end))

    # CamSol-lite: spatial aggregation = mean(hydrophobicity) per residue, penalise charged
    camsolite = sum(KD.get(aa, 0) for aa in seq) / len(seq)
    # normalise: positive = more aggregation prone
    camsolite_norm = round(camsolite, 4)

    # SAP-like: fraction of hydrophobic residues in CDR3
    cdr3 = info["CDR3"]
    cdr3_hydro_frac = sum(1 for aa in cdr3 if aa in 'VILMFYW') / len(cdr3)

    # SOLpro-lite: charge + hydrophobicity balance
    n_charged = sum(seq.count(aa) for aa in 'RKHDE')
    n_hydro   = sum(seq.count(aa) for aa in 'VILMFYW')
    sol_score = (n_charged - n_hydro) / len(seq)   # positive = more soluble
    if sol_score > 0.05:   solubility_class = "Soluble"
    elif sol_score > -0.05: solubility_class = "Borderline"
    else:                   solubility_class = "Insoluble risk"

    if len(aprs) == 0:    agg_class = "Low"
    elif len(aprs) <= 3:  agg_class = "Medium"
    else:                 agg_class = "High"

    log.info(f"  {nb_name}: APRs={len(apr_regions)}  CamSol-lite={camsolite_norm:.3f}  "
             f"Sol={sol_score:.3f} ({solubility_class})  AggRisk={agg_class}")

    return {
        "n_APR_peptides":      len(aprs),
        "n_APR_regions":       len(apr_regions),
        "apr_regions":         str(apr_regions),
        "agg_risk_class":      agg_class,
        "camsolite_score":     camsolite_norm,
        "CDR3_hydro_fraction": round(cdr3_hydro_frac, 4),
        "solubility_score":    round(sol_score, 4),
        "solubility_class":    solubility_class,
        "mean_agg_score":      round(np.mean([s[2] for s in agg_scores]), 4),
        "max_agg_score":       round(max(s[2] for s in agg_scores), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 4 — Cross-Reactivity (NCBI BLAST CDR3 vs Homo sapiens)
# ══════════════════════════════════════════════════════════════════════════════
def run_blast_cdr3(nb_name, cdr3_seq):
    log.info(f"  BLASTing CDR3 of {nb_name} ({cdr3_seq}) vs Homo sapiens proteome...")
    try:
        result_handle = NCBIWWW.qblast(
            "blastp", "nr", cdr3_seq,
            entrez_query="Homo sapiens[organism]",
            hitlist_size=5, expect=10,
            word_size=2, matrix_name="BLOSUM62"
        )
        blast_records = list(NCBIXML.parse(result_handle))
        hits = []
        if blast_records and blast_records[0].alignments:
            for aln in blast_records[0].alignments[:3]:
                hsp = aln.hsps[0]
                pct_id = hsp.identities / hsp.align_length * 100
                hits.append({
                    "hit": aln.title[:80],
                    "pct_identity": round(pct_id, 1),
                    "evalue": hsp.expect,
                    "score": hsp.score,
                })
        if not hits:
            log.info(f"    {nb_name}: No significant BLAST hits — low cross-reactivity risk")
            return {"blast_hits": 0, "max_identity_pct": 0,
                    "cross_reactivity_risk": "Low", "top_hit": "None"}
        max_id = max(h["pct_identity"] for h in hits)
        risk = "Low" if max_id < 50 else "Medium" if max_id < 80 else "High"
        log.info(f"    {nb_name}: {len(hits)} hits, max identity={max_id:.1f}%  Risk={risk}")
        return {
            "blast_hits": len(hits),
            "max_identity_pct": max_id,
            "cross_reactivity_risk": risk,
            "top_hit": hits[0]["hit"][:60] if hits else "None",
            "top_hit_evalue": hits[0]["evalue"] if hits else None,
        }
    except Exception as e:
        log.warning(f"    BLAST failed for {nb_name}: {e}")
        return {"blast_hits": -1, "max_identity_pct": None,
                "cross_reactivity_risk": "Unknown", "top_hit": "BLAST failed"}


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 5 — Developability Scorecard
# ══════════════════════════════════════════════════════════════════════════════
def compute_developability(physchem, immuno, aggr, blast):
    scores = {}

    # pI score: ideal 6–8 (neutral) gets 1.0; 8–9 gets 0.7; >9 gets 0.3
    pI = physchem["pI"]
    if 6 <= pI <= 8:       scores["pI_score"] = 1.0
    elif 8 < pI <= 9:      scores["pI_score"] = 0.7
    elif 5 <= pI < 6:      scores["pI_score"] = 0.6
    else:                  scores["pI_score"] = 0.3

    # Stability score: instability index <40=1.0, 40-50=0.6, >50=0.2
    ii = physchem["instability"]
    scores["stability_score"] = 1.0 if ii < 40 else 0.6 if ii < 50 else 0.2

    # GRAVY score: negative=soluble. -0.5–0 is ideal
    g = physchem["GRAVY"]
    scores["gravy_score"] = 1.0 if -0.5 <= g <= 0 else 0.7 if g < 0 else 0.3

    # Aggregation score: Low=1.0, Medium=0.6, High=0.2
    agg_map = {"Low": 1.0, "Medium": 0.6, "High": 0.2}
    scores["aggregation_score"] = agg_map.get(aggr["agg_risk_class"], 0.5)

    # Immunogenicity score: Low=1.0, Medium=0.6, High=0.2
    imm_map = {"Low": 1.0, "Medium": 0.6, "High": 0.2}
    scores["immunogenicity_score"] = imm_map.get(immuno["immunogenicity_risk"], 0.5)

    # Cross-reactivity: Low=1.0, Medium=0.6, High=0.2, Unknown=0.5
    cr_map = {"Low": 1.0, "Medium": 0.6, "High": 0.2, "Unknown": 0.5}
    scores["cross_react_score"] = cr_map.get(blast.get("cross_reactivity_risk", "Unknown"), 0.5)

    # Solubility score
    sol_map = {"Soluble": 1.0, "Borderline": 0.6, "Insoluble risk": 0.2}
    scores["solubility_score"] = sol_map.get(aggr["solubility_class"], 0.5)

    # Composite developability index (weighted)
    weights = {"stability_score":0.20, "aggregation_score":0.20,
               "immunogenicity_score":0.20, "solubility_score":0.15,
               "pI_score":0.10, "gravy_score":0.10, "cross_react_score":0.05}
    composite = sum(scores[k]*w for k,w in weights.items())
    scores["developability_index"] = round(composite, 4)

    if composite >= 0.75:   dev_class = "Excellent"
    elif composite >= 0.60: dev_class = "Good"
    elif composite >= 0.45: dev_class = "Moderate"
    else:                   dev_class = "Poor"
    scores["developability_class"] = dev_class

    return scores


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 6 — Figures
# ══════════════════════════════════════════════════════════════════════════════
COLORS = {
    "NbLasR-1": "#2196F3", "NbLasR-2": "#FF9800",
    "NbLasR-3": "#4CAF50", "NbLasR-4": "#9C27B0", "NbLasR-5": "#F44336"
}

def generate_figures(all_results):
    candidates = list(all_results.keys())
    bar_cols    = [COLORS[c] for c in candidates]
    fig_paths   = []

    def save(fig, name):
        p = f"{FIGDIR}/{name}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved: {name}")
        fig_paths.append(p)

    # ── Fig 1: pI ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    pI_vals = [all_results[c]["physchem"]["pI"] for c in candidates]
    bars = ax.bar(candidates, pI_vals, color=bar_cols, edgecolor="black", linewidth=0.8, zorder=3)
    ax.axhspan(6, 8, alpha=0.15, color="green", label="Ideal pI range (6–8)")
    ax.axhspan(8, 9, alpha=0.10, color="yellow")
    ax.axhline(7, color="green", linewidth=1.2, linestyle="--", alpha=0.7)
    for bar, val in zip(bars, pI_vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.03, f"{val:.2f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Isoelectric Point (pI)", fontsize=12)
    ax.set_title("Isoelectric Point (pI) — LasR Nanobody Candidates", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.set_facecolor("#f9f9f9")
    plt.tight_layout()
    save(fig, "Fig_Phase7_01_pI.png")

    # ── Fig 2: Instability Index ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ii_vals = [all_results[c]["physchem"]["instability"] for c in candidates]
    bars = ax.bar(candidates, ii_vals, color=bar_cols, edgecolor="black", linewidth=0.8, zorder=3)
    ax.axhline(40, color="red", linewidth=2, linestyle="--", label="Stability threshold (40)")
    for bar, val in zip(bars, ii_vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.3, f"{val:.1f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Instability Index", fontsize=12)
    ax.set_title("Instability Index — Stable (<40) vs Unstable (≥40)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.set_facecolor("#f9f9f9")
    plt.tight_layout()
    save(fig, "Fig_Phase7_02_Instability.png")

    # ── Fig 3: GRAVY + Aliphatic Index ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    gravy_vals = [all_results[c]["physchem"]["GRAVY"] for c in candidates]
    ali_vals   = [all_results[c]["physchem"]["aliphatic_idx"] for c in candidates]
    axes[0].bar(candidates, gravy_vals, color=bar_cols, edgecolor="black", linewidth=0.8, zorder=3)
    axes[0].axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    axes[0].axhspan(-0.5, 0, alpha=0.1, color="green", label="Ideal (hydrophilic)")
    axes[0].set_title("GRAVY Score", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("GRAVY (negative = hydrophilic)", fontsize=11)
    for bar, val in zip(axes[0].patches, gravy_vals):
        axes[0].text(bar.get_x()+bar.get_width()/2,
                     val+(0.005 if val>=0 else -0.015),
                     f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    axes[1].bar(candidates, ali_vals, color=bar_cols, edgecolor="black", linewidth=0.8, zorder=3)
    axes[1].set_title("Aliphatic Index", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Aliphatic Index (higher = more thermostable)", fontsize=11)
    for bar, val in zip(axes[1].patches, ali_vals):
        axes[1].text(bar.get_x()+bar.get_width()/2, val+0.3,
                     f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")
    for ax in axes:
        ax.grid(axis="y", alpha=0.4, zorder=0)
        ax.set_facecolor("#f9f9f9")
    fig.suptitle("Physicochemical Properties — GRAVY & Aliphatic Index", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "Fig_Phase7_03_GRAVY_Aliphatic.png")

    # ── Fig 4: MHC-I & MHC-II Immunogenicity Risk ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    mhc1 = [all_results[c]["immuno"]["mhc1_risk_pct"] for c in candidates]
    mhc2 = [all_results[c]["immuno"]["mhc2_risk_pct"] for c in candidates]
    axes[0].bar(candidates, mhc1, color=bar_cols, edgecolor="black", linewidth=0.8, zorder=3)
    axes[0].axhline(20, color="orange", linestyle="--", linewidth=1.5, label="Medium threshold (20%)")
    axes[0].axhline(35, color="red", linestyle="--", linewidth=1.5, label="High threshold (35%)")
    axes[0].set_title("MHC-I T-cell Epitope Risk (%)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("% high-risk 9-mers", fontsize=11)
    axes[0].legend(fontsize=9)
    for bar, val in zip(axes[0].patches, mhc1):
        axes[0].text(bar.get_x()+bar.get_width()/2, val+0.3,
                     f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
    axes[1].bar(candidates, mhc2, color=bar_cols, edgecolor="black", linewidth=0.8, zorder=3)
    axes[1].set_title("MHC-II T-cell Epitope Risk (%)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("% high-risk 15-mers", fontsize=11)
    for bar, val in zip(axes[1].patches, mhc2):
        axes[1].text(bar.get_x()+bar.get_width()/2, val+0.3,
                     f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
    for ax in axes:
        ax.grid(axis="y", alpha=0.4, zorder=0)
        ax.set_facecolor("#f9f9f9")
    fig.suptitle("Immunogenicity Risk — MHC-I (9-mer) and MHC-II (15-mer)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "Fig_Phase7_04_Immunogenicity.png")

    # ── Fig 5: Aggregation-Prone Regions ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    apr_n    = [all_results[c]["aggr"]["n_APR_regions"] for c in candidates]
    camsol   = [all_results[c]["aggr"]["camsolite_score"] for c in candidates]
    axes[0].bar(candidates, apr_n, color=bar_cols, edgecolor="black", linewidth=0.8, zorder=3)
    axes[0].set_title("Aggregation-Prone Regions (APR count)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Number of APR regions", fontsize=11)
    for bar, val in zip(axes[0].patches, apr_n):
        axes[0].text(bar.get_x()+bar.get_width()/2, val+0.05,
                     str(val), ha="center", fontsize=10, fontweight="bold")
    axes[1].bar(candidates, camsol, color=bar_cols, edgecolor="black", linewidth=0.8, zorder=3)
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    axes[1].set_title("CamSol-lite Score", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("CamSol-lite (negative = more soluble)", fontsize=11)
    for bar, val in zip(axes[1].patches, camsol):
        axes[1].text(bar.get_x()+bar.get_width()/2,
                     val+(0.005 if val>=0 else -0.015),
                     f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    for ax in axes:
        ax.grid(axis="y", alpha=0.4, zorder=0)
        ax.set_facecolor("#f9f9f9")
    fig.suptitle("Aggregation Propensity — APR Regions & CamSol-lite",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "Fig_Phase7_05_Aggregation.png")

    # ── Fig 6: Solubility + Charge at pH 7.4 ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sol_s  = [all_results[c]["aggr"]["solubility_score"] for c in candidates]
    chg74  = [all_results[c]["physchem"]["charge_pH74"] for c in candidates]
    axes[0].bar(candidates, sol_s, color=bar_cols, edgecolor="black", linewidth=0.8, zorder=3)
    axes[0].axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    axes[0].axhline(0.05, color="green", linestyle=":", linewidth=1.5, label="Soluble threshold")
    axes[0].set_title("SOLpro-lite Solubility Score", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Score (positive = soluble)", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[1].bar(candidates, chg74, color=bar_cols, edgecolor="black", linewidth=0.8, zorder=3)
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    axes[1].set_title("Net Charge at pH 7.4", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Net charge (e)", fontsize=11)
    for ax, vals in zip(axes, [sol_s, chg74]):
        ax.grid(axis="y", alpha=0.4, zorder=0)
        ax.set_facecolor("#f9f9f9")
        for bar, val in zip(ax.patches, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    val+(0.003 if val>=0 else -0.02),
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    fig.suptitle("Solubility & Net Charge — Developability Metrics",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "Fig_Phase7_06_Solubility_Charge.png")

    # ── Fig 7: CDR3 Properties Comparison ─────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    cdr3_lens  = [len(all_results[c]["info"]["CDR3"]) for c in candidates]
    cdr3_hydro = [all_results[c]["immuno"]["cdr3_hydrophobicity"] for c in candidates]
    cdr3_agg   = [all_results[c]["aggr"]["CDR3_hydro_fraction"] for c in candidates]
    for ax, vals, title, ylabel in zip(
        axes,
        [cdr3_lens, cdr3_hydro, cdr3_agg],
        ["CDR3 Length (aa)", "CDR3 Hydrophobicity (KD)", "CDR3 Hydrophobic Fraction"],
        ["Amino acids", "KD score", "Fraction"]
    ):
        ax.bar(candidates, vals, color=bar_cols, edgecolor="black", linewidth=0.8, zorder=3)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", alpha=0.4, zorder=0)
        ax.set_facecolor("#f9f9f9")
        for bar, val in zip(ax.patches, vals):
            ax.text(bar.get_x()+bar.get_width()/2, val * 1.02,
                    f"{val:.2f}" if isinstance(val, float) else str(val),
                    ha="center", fontsize=9, fontweight="bold")
    fig.suptitle("CDR3 Properties — Length, Hydrophobicity, Hydrophobic Fraction",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "Fig_Phase7_07_CDR3_Properties.png")

    # ── Fig 8: Developability Scores Breakdown ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 7))
    score_keys = ["stability_score","aggregation_score","immunogenicity_score",
                  "solubility_score","pI_score","gravy_score","cross_react_score"]
    score_labels = ["Stability","Aggregation\nResistance","Immunogenicity\nSafety",
                    "Solubility","pI","GRAVY","Cross-\nReactivity"]
    x  = np.arange(len(candidates))
    w  = 0.11
    colors_bar = plt.cm.Set2(np.linspace(0, 1, len(score_keys)))
    for i, (key, label, col) in enumerate(zip(score_keys, score_labels, colors_bar)):
        vals = [all_results[c]["devlp"][key] for c in candidates]
        ax.bar(x + i*w - len(score_keys)*w/2 + w/2, vals, w*0.9,
               label=label, color=col, edgecolor="black", linewidth=0.5, zorder=3)
    ax.axhline(0.75, color="green", linestyle="--", linewidth=1.5, alpha=0.7, label="Excellent threshold")
    ax.axhline(0.60, color="orange", linestyle=":", linewidth=1.5, alpha=0.7, label="Good threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(candidates, fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score (0–1, higher = better)", fontsize=12)
    ax.set_title("Developability Score Breakdown — All 7 Metrics",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.set_facecolor("#f9f9f9")
    plt.tight_layout()
    save(fig, "Fig_Phase7_08_Developability_Breakdown.png")

    # ── Fig 9: Composite Developability Index ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    dev_idx = [all_results[c]["devlp"]["developability_index"] for c in candidates]
    bars = ax.bar(candidates, dev_idx, color=bar_cols, edgecolor="black", linewidth=0.8, zorder=3)
    ax.axhline(0.75, color="green", linestyle="--", linewidth=2, label="Excellent (≥0.75)")
    ax.axhline(0.60, color="orange", linestyle="--", linewidth=2, label="Good (≥0.60)")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, dev_idx):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.01,
                f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Composite Developability Index (0–1)", fontsize=12)
    ax.set_title("Overall Developability Index — LasR Nanobody Candidates",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.set_facecolor("#f9f9f9")
    plt.tight_layout()
    save(fig, "Fig_Phase7_09_Developability_Index.png")

    # ── Fig 10: Radar / Spider Chart ───────────────────────────────────────────
    metrics_radar = ["Stability","Agg.\nResist.","Immuno.\nSafety",
                     "Solubility","pI","GRAVY","Cross-\nReact."]
    score_keys_r  = ["stability_score","aggregation_score","immunogenicity_score",
                     "solubility_score","pI_score","gravy_score","cross_react_score"]
    N = len(metrics_radar)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for nb, color in COLORS.items():
        vals = [all_results[nb]["devlp"][k] for k in score_keys_r]
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=2, label=nb)
        ax.fill(angles, vals, color=color, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_radar, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0.25","0.50","0.75","1.0"], fontsize=8)
    ax.set_title("Developability Radar Chart\nLasR Nanobody Candidates",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
    plt.tight_layout()
    save(fig, "Fig_Phase7_10_Radar_Chart.png")

    # ── Fig 11: Heatmap — all metrics ─────────────────────────────────────────
    all_keys = ["pI","instability","GRAVY","aliphatic_idx","charge_pH74",
                "mhc1_risk_pct","mhc2_risk_pct","n_APR_regions","solubility_score",
                "developability_index"]
    all_labels = ["pI","Instability\nIndex","GRAVY","Aliphatic\nIndex","Charge\n@pH7.4",
                  "MHC-I\nRisk (%)","MHC-II\nRisk (%)","# APR\nRegions","Solubility\nScore",
                  "Develop.\nIndex"]
    raw = np.array([
        [all_results[c]["physchem"].get(k) or all_results[c]["immuno"].get(k) or
         all_results[c]["aggr"].get(k) or all_results[c]["devlp"].get(k, 0)
         for k in all_keys]
        for c in candidates
    ], dtype=float).T   # shape (metrics, candidates)

    # normalise each row 0–1
    norm = np.zeros_like(raw)
    for i in range(raw.shape[0]):
        mn, mx = raw[i].min(), raw[i].max()
        if mx > mn:
            norm[i] = (raw[i] - mn) / (mx - mn)
        else:
            norm[i] = 0.5

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(candidates)))
    ax.set_xticklabels(candidates, fontsize=11)
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels(all_labels, fontsize=10)
    for i in range(len(all_labels)):
        for j in range(len(candidates)):
            ax.text(j, i, f"{raw[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color="black")
    plt.colorbar(im, ax=ax, label="Normalised (1=best)")
    ax.set_title("Phase 7 — Complete Immunogenicity & Developability Heatmap",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "Fig_Phase7_11_Full_Heatmap.png")

    # ── Fig 12: Final Risk Summary Table ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    col_hdrs = ["Candidate","Immunogenicity","Aggregation","Solubility","Stability","Dev. Index","Class"]
    table_data = []
    for c in candidates:
        d = all_results[c]
        table_data.append([
            c,
            d["immuno"]["immunogenicity_risk"],
            d["aggr"]["agg_risk_class"],
            d["aggr"]["solubility_class"],
            d["physchem"]["stability_class"],
            f"{d['devlp']['developability_index']:.3f}",
            d["devlp"]["developability_class"],
        ])
    tbl = ax.table(cellText=table_data, colLabels=col_hdrs,
                   cellLoc="center", loc="center",
                   bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    for (r, c_), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2196F3")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4ff")
        # colour risk cells
        if r > 0:
            txt = cell.get_text().get_text()
            if txt in ("Low","Stable","Soluble","Excellent","Good"):
                cell.set_facecolor("#c8e6c9")
            elif txt in ("Medium","Borderline","Moderate"):
                cell.set_facecolor("#fff9c4")
            elif txt in ("High","Unstable","Insoluble risk","Poor"):
                cell.set_facecolor("#ffcdd2")
    ax.set_title("Phase 7 — Risk Summary Table", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    save(fig, "Fig_Phase7_12_Risk_Summary_Table.png")

    return fig_paths


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 7 — Save results + Report
# ══════════════════════════════════════════════════════════════════════════════
def save_results(all_results):
    # JSON
    json_path = f"{OUTDIR}/phase7_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"Saved JSON: {json_path}")

    # CSV
    csv_path = f"{OUTDIR}/phase7_ranking.csv"
    rows = []
    for nb, d in all_results.items():
        row = {"candidate": nb, "target": d["info"]["target"],
               "CDR3": d["info"]["CDR3"], "CDR3_length": len(d["info"]["CDR3"])}
        row.update(d["physchem"])
        row.update(d["immuno"])
        row.update({k: v for k,v in d["aggr"].items() if k != "apr_regions"})
        row.update(d["devlp"])
        row["blast_cross_reactivity_risk"] = d["blast"].get("cross_reactivity_risk")
        row["blast_max_identity_pct"]       = d["blast"].get("max_identity_pct")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    log.info(f"Saved CSV: {csv_path}")

    # Text report
    rep_path = f"{OUTDIR}/PHASE7_REPORT.txt"
    with open(rep_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("PHASE 7 REPORT — Immunogenicity & Developability Assessment\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.write("METHOD SUMMARY\n" + "-"*40 + "\n")
        f.write("  Physicochemical: BioPython ProtParam (MW, pI, instability, GRAVY, aliphatic)\n")
        f.write("  Immunogenicity:  SYFPEITHI-inspired 9-mer (MHC-I) + 15-mer (MHC-II) scoring\n")
        f.write("  Aggregation:     TANGO-inspired APR detection + CamSol-lite + SOLpro-lite\n")
        f.write("  Cross-reactivity:NCBI BLAST CDR3 vs Homo sapiens proteome\n")
        f.write("  Developability:  Composite of 7 weighted sub-scores\n\n")
        f.write("RESULTS\n" + "-"*40 + "\n")
        sorted_res = sorted(all_results.items(),
                            key=lambda x: x[1]["devlp"]["developability_index"], reverse=True)
        for rank, (nb, d) in enumerate(sorted_res, 1):
            pc = d["physchem"]; im = d["immuno"]; ag = d["aggr"]; dv = d["devlp"]
            f.write(f"\n  RANK {rank}: {nb} — {dv['developability_class']} "
                    f"(Index: {dv['developability_index']:.3f})\n")
            f.write(f"    CDR3: {d['info']['CDR3']}  ({len(d['info']['CDR3'])} aa)\n")
            f.write(f"    pI={pc['pI']:.2f}  MW={pc['MW_kDa']:.2f} kDa  "
                    f"Instability={pc['instability']:.1f} ({pc['stability_class']})\n")
            f.write(f"    GRAVY={pc['GRAVY']:.4f}  Aliphatic={pc['aliphatic_idx']:.1f}  "
                    f"Charge@pH7.4={pc['charge_pH74']:.2f}\n")
            f.write(f"    MHC-I risk={im['mhc1_risk_pct']:.1f}%  "
                    f"MHC-II risk={im['mhc2_risk_pct']:.1f}%  "
                    f"Immunogenicity: {im['immunogenicity_risk']}\n")
            f.write(f"    APR regions={ag['n_APR_regions']}  "
                    f"CamSol={ag['camsolite_score']:.4f}  "
                    f"Aggregation: {ag['agg_risk_class']}\n")
            f.write(f"    Solubility: {ag['solubility_class']} (score={ag['solubility_score']:.3f})\n")
            f.write(f"    Cross-reactivity: {d['blast'].get('cross_reactivity_risk','Unknown')}  "
                    f"(BLAST max id={d['blast'].get('max_identity_pct','N/A')}%)\n")
        f.write("\n" + "="*70 + "\n")
        f.write("FIGURES (12 panels)\n" + "-"*40 + "\n")
        for i in range(1, 13):
            f.write(f"  Fig Phase7 {i:02d}: 08_Results_Figures/Phase7/ (see figures/ subdir)\n")
        f.write("\nNEXT STEP → PHASE 5: Molecular Dynamics Simulation\n")
        f.write("  Target: NbLasR-1 (primary) + NbLasR-2 (secondary)\n")
        f.write("="*70 + "\n")
    log.info(f"Saved Report: {rep_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log.info("="*70)
    log.info("  PHASE 7 — Immunogenicity & Developability Assessment")
    log.info("="*70)

    all_results = {}

    for nb_name, info in NANOBODIES.items():
        log.info(f"\n{'='*50}\n  {nb_name}\n{'='*50}")
        pc    = compute_physicochemical(nb_name, info)
        im    = compute_immunogenicity(nb_name, info)
        ag    = compute_aggregation(nb_name, info)
        blast = run_blast_cdr3(nb_name, info["CDR3"])
        dv    = compute_developability(pc, im, ag, blast)

        log.info(f"  → Developability: {dv['developability_class']} "
                 f"(index={dv['developability_index']:.3f})")
        all_results[nb_name] = {
            "info": info, "physchem": pc, "immuno": im,
            "aggr": ag, "blast": blast, "devlp": dv
        }

    save_results(all_results)

    log.info("\nGenerating 12 figures...")
    generate_figures(all_results)

    log.info("\n" + "="*70)
    log.info("  PHASE 7 COMPLETE")
    log.info(f"  Results: {OUTDIR}/phase7_ranking.csv")
    log.info(f"  Report:  {OUTDIR}/PHASE7_REPORT.txt")
    log.info(f"  Figures: {FIGDIR}/")
    log.info(f"  Log:     {LOG_FILE}")
    log.info("="*70)
