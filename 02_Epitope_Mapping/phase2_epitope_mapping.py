"""
Phase 2: Epitope Mapping — LasR Quorum Sensing Receptor
- Extract LasR protein sequence from 2UV0
- B-cell epitope prediction: Kolaskar-Tongaonkar, Parker, Emini, Karplus-Schulz
- Antigenicity scoring (VaxiJen-like method)
- Conservation analysis: fetch multiple LasR homologs from NCBI + MSA
- Surface accessibility estimation
- Identify top 3 epitope regions focused on LBD
- Generate all figures and tables
"""

import os, sys, json, time, requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from datetime import datetime
from Bio import PDB, SeqIO, Entrez, AlignIO
from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import pairwise2
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

Entrez.email = "pharmafriend23@gmail.com"

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = "/home/stalin/Desktop/LasR_Antibody_Design"
PHASE2 = os.path.join(BASE, "02_Epitope_Mapping")
FIGS   = os.path.join(BASE, "08_Results_Figures", "Phase2")
LOGS   = os.path.join(BASE, "logs")
DATA   = os.path.join(BASE, "data")
os.makedirs(FIGS, exist_ok=True)

LOG_FILE = os.path.join(LOGS, "phase2_log.txt")
COLORS = {"primary": "#1B4F72", "secondary": "#2E86AB", "accent": "#E84855",
          "highlight": "#F4A261", "green": "#2A9D8F", "purple": "#7D3C98",
          "light": "#A8DADC", "gold": "#D4AC0D"}

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

log("=" * 70)
log("PHASE 2: Epitope Mapping — STARTED")
log("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Extract LasR sequence from 2UV0
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 1: Extracting LasR sequence from 2UV0...")

AA_3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M',
    'SEL':'C','HYP':'P','CSE':'C'
}

parser = PDBParser(QUIET=True)
struct  = parser.get_structure("2UV0", os.path.join(DATA, "2UV0.pdb"))
model   = struct[0]

chain_seqs = {}
chain_resnums = {}
for chain in model.get_chains():
    residues = [r for r in chain.get_residues() if r.get_id()[0] in (' ', 'H_MSE')]
    aa_res = []
    res_nums = []
    for r in residues:
        rname = r.get_resname().strip()
        if rname in AA_3TO1:
            aa_res.append(AA_3TO1[rname])
            res_nums.append(r.get_id()[1])
    if len(aa_res) > 50:
        chain_seqs[chain.id]   = "".join(aa_res)
        chain_resnums[chain.id] = res_nums

log(f"  Chains extracted: {list(chain_seqs.keys())}")
for cid, seq in chain_seqs.items():
    log(f"  Chain {cid}: {len(seq)} residues")

# Use chain E as primary (chain A of biological unit)
primary_chain  = "E"
LasR_seq       = chain_seqs[primary_chain]
LasR_resnums   = chain_resnums[primary_chain]
log(f"  Primary chain: {primary_chain} | Length: {len(LasR_seq)}")
log(f"  Sequence: {LasR_seq[:60]}...")

seq_fasta = os.path.join(PHASE2, "LasR_2UV0_chainE.fasta")
with open(seq_fasta, "w") as f:
    f.write(f">LasR_2UV0_chainE\n{LasR_seq}\n")
log(f"  FASTA saved → {seq_fasta}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Kolaskar-Tongaonkar Antigenicity Scale
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 2: Kolaskar-Tongaonkar antigenicity scale...")

KT_SCALE = {
    'A':1.064,'R':0.873,'N':0.776,'D':0.924,'C':1.020,'Q':0.853,'E':0.837,
    'G':0.874,'H':1.105,'I':1.152,'L':1.250,'K':0.897,'M':0.826,'F':1.091,
    'P':0.922,'S':0.883,'T':0.909,'W':1.137,'Y':1.159,'V':1.383
}

def kolaskar_tongaonkar(seq, window=7):
    scores = []
    half_w = window // 2
    seq_scores = [KT_SCALE.get(aa, 1.0) for aa in seq]
    for i in range(len(seq)):
        start = max(0, i - half_w)
        end   = min(len(seq), i + half_w + 1)
        scores.append(np.mean(seq_scores[start:end]))
    return np.array(scores)

kt_scores = kolaskar_tongaonkar(LasR_seq)
kt_mean   = np.mean(kt_scores)
kt_threshold = kt_mean * 1.02
log(f"  KT mean: {kt_mean:.4f} | threshold: {kt_threshold:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Parker Hydrophilicity Scale
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 3: Parker hydrophilicity scale...")

PARKER_SCALE = {
    'A':-0.5,'R': 3.0,'N': 0.2,'D': 3.0,'C':-1.0,'Q': 0.2,'E': 3.0,
    'G': 0.0,'H':-0.5,'I':-1.8,'L':-1.8,'K': 3.0,'M':-1.3,'F':-2.5,
    'P': 0.0,'S': 0.3,'T':-0.4,'W':-3.4,'Y':-2.3,'V':-1.5
}

def parker_hydrophilicity(seq, window=7):
    scores = []
    half_w = window // 2
    seq_scores = [PARKER_SCALE.get(aa, 0.0) for aa in seq]
    for i in range(len(seq)):
        start = max(0, i - half_w)
        end   = min(len(seq), i + half_w + 1)
        scores.append(np.mean(seq_scores[start:end]))
    return np.array(scores)

parker_scores = parker_hydrophilicity(LasR_seq)
log(f"  Parker mean: {np.mean(parker_scores):.4f} | range: [{np.min(parker_scores):.2f}, {np.max(parker_scores):.2f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Emini Surface Accessibility Scale
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 4: Emini surface accessibility scale...")

EMINI_SCALE = {
    'A':0.49,'R':1.41,'N':1.28,'D':1.35,'C':0.26,'Q':0.89,'E':1.45,
    'G':0.48,'H':0.40,'I':0.29,'L':0.34,'K':1.06,'M':0.36,'F':0.42,
    'P':0.75,'S':1.15,'T':1.03,'W':0.67,'Y':0.76,'V':0.33
}

def emini_accessibility(seq, window=6):
    scores = []
    seq_scores = [EMINI_SCALE.get(aa, 0.5) for aa in seq]
    for i in range(len(seq) - window + 1):
        product = 1.0
        for j in range(window):
            product *= seq_scores[i+j]
        scores.append((product ** (1.0/window)) / 6.0)
    # Pad to full length
    pad_front = window // 2
    pad_back  = len(seq) - len(scores) - pad_front
    return np.array([scores[0]] * pad_front + scores + [scores[-1]] * pad_back)

emini_scores = emini_accessibility(LasR_seq)
log(f"  Emini mean: {np.mean(emini_scores):.4f} | range: [{np.min(emini_scores):.4f}, {np.max(emini_scores):.4f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Karplus-Schulz Flexibility Scale
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 5: Karplus-Schulz backbone flexibility scale...")

KS_SCALE = {
    'A':0.357,'R':0.529,'N':0.463,'D':0.511,'C':0.346,'Q':0.493,'E':0.497,
    'G':0.544,'H':0.323,'I':0.462,'L':0.365,'K':0.466,'M':0.295,'F':0.314,
    'P':0.509,'S':0.507,'T':0.444,'W':0.305,'Y':0.420,'V':0.386
}

def karplus_schulz_flex(seq, window=7):
    scores = []
    half_w = window // 2
    seq_scores = [KS_SCALE.get(aa, 0.4) for aa in seq]
    for i in range(len(seq)):
        start = max(0, i - half_w)
        end   = min(len(seq), i + half_w + 1)
        scores.append(np.mean(seq_scores[start:end]))
    return np.array(scores)

ks_scores = karplus_schulz_flex(LasR_seq)
log(f"  KS flexibility mean: {np.mean(ks_scores):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: Chou-Fasman Beta-Turn Prediction
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 6: Chou-Fasman beta-turn prediction...")

CF_TURN = {
    'A':0.66,'R':0.95,'N':1.56,'D':1.46,'C':1.19,'Q':0.98,'E':0.74,
    'G':1.56,'H':0.95,'I':0.47,'L':0.59,'K':1.01,'M':0.60,'F':0.60,
    'P':1.52,'S':1.43,'T':0.96,'W':0.96,'Y':1.14,'V':0.50
}

def chou_fasman_turns(seq, window=4):
    scores = []
    seq_scores = [CF_TURN.get(aa, 1.0) for aa in seq]
    for i in range(len(seq) - window + 1):
        product = np.prod(seq_scores[i:i+window])
        scores.append(product ** (1.0/window))
    pad_front = window // 2
    pad_back  = len(seq) - len(scores) - pad_front
    return np.array([scores[0]] * pad_front + scores + [scores[-1]] * max(0, pad_back))

cf_scores = chou_fasman_turns(LasR_seq)
log(f"  CF turn score mean: {np.mean(cf_scores):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: Combined Epitope Score & Region Identification
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 7: Computing combined epitope score...")

def zscore(arr):
    if np.std(arr) == 0:
        return np.zeros_like(arr)
    return (arr - np.mean(arr)) / np.std(arr)

# Normalize each score (z-score), combine
# KT and emini are pro-antigenicity, Parker high = hydrophilic (good for surface)
# KS high = flexible loops (good for antibody binding)
# CF high = beta-turn (good for epitopes)
z_kt     = zscore(kt_scores)
z_parker = zscore(parker_scores)
z_emini  = zscore(emini_scores)
z_ks     = zscore(ks_scores)
z_cf     = zscore(cf_scores)

# Weighted sum: KT and Emini weighted highest
combined = (0.30 * z_kt + 0.25 * z_emini + 0.20 * z_parker + 0.15 * z_ks + 0.10 * z_cf)
combined_threshold = np.mean(combined) + 0.5 * np.std(combined)
log(f"  Combined score mean: {np.mean(combined):.4f} | threshold: {combined_threshold:.4f}")

# Build per-residue dataframe
df_epi = pd.DataFrame({
    "Position":   range(1, len(LasR_seq)+1),
    "ResNum":     LasR_resnums[:len(LasR_seq)],
    "AA":         list(LasR_seq),
    "KT_score":   kt_scores,
    "Parker":     parker_scores,
    "Emini":      emini_scores,
    "KS_flex":    ks_scores,
    "CF_turn":    cf_scores[:len(LasR_seq)],
    "Combined":   combined,
    "Above_thresh": combined > combined_threshold
})

epi_csv = os.path.join(PHASE2, "epitope_scores_all_residues.csv")
df_epi.to_csv(epi_csv, index=False)
log(f"  Epitope scores table saved ({len(df_epi)} residues) → {epi_csv}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: Identify Continuous Epitope Regions
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 8: Identifying continuous epitope regions...")

def find_epitope_regions(scores, positions, resnums, aas, threshold, min_len=6):
    regions = []
    in_region = False
    start = 0
    for i, (s, pos, rnum, aa) in enumerate(zip(scores, positions, resnums, aas)):
        if s > threshold and not in_region:
            in_region = True
            start = i
        elif s <= threshold and in_region:
            length = i - start
            if length >= min_len:
                region_scores = scores[start:i]
                regions.append({
                    "Region_ID":  len(regions) + 1,
                    "Start_Pos":  positions[start],
                    "End_Pos":    positions[i-1],
                    "Start_ResNum": resnums[start],
                    "End_ResNum":   resnums[i-1],
                    "Length":     length,
                    "Sequence":   "".join(aas[start:i]),
                    "Mean_Score": round(float(np.mean(region_scores)), 4),
                    "Max_Score":  round(float(np.max(region_scores)),  4),
                    "Peak_Pos":   positions[start + int(np.argmax(region_scores))]
                })
            in_region = False
    if in_region:
        length = len(scores) - start
        if length >= min_len:
            region_scores = scores[start:]
            regions.append({
                "Region_ID":  len(regions) + 1,
                "Start_Pos":  positions[start],
                "End_Pos":    positions[-1],
                "Start_ResNum": resnums[start],
                "End_ResNum":   resnums[-1],
                "Length":     length,
                "Sequence":   "".join(aas[start:]),
                "Mean_Score": round(float(np.mean(region_scores)), 4),
                "Max_Score":  round(float(np.max(region_scores)),  4),
                "Peak_Pos":   positions[start + int(np.argmax(region_scores))]
            })
    return regions

positions = list(range(1, len(LasR_seq)+1))
resnums   = LasR_resnums[:len(LasR_seq)]
aas       = list(LasR_seq)

regions = find_epitope_regions(combined, positions, resnums, aas, combined_threshold)
df_regions = pd.DataFrame(regions)

if len(df_regions) > 0:
    df_regions = df_regions.sort_values("Mean_Score", ascending=False).reset_index(drop=True)
    df_regions["Rank"] = range(1, len(df_regions)+1)
    log(f"  Found {len(df_regions)} epitope regions")
    for _, row in df_regions.head(5).iterrows():
        log(f"  Region {int(row['Region_ID'])}: pos {int(row['Start_Pos'])}-{int(row['End_Pos'])} "
            f"({int(row['Length'])} aa) | {row['Sequence']} | score={row['Mean_Score']:.4f}")
    regions_csv = os.path.join(PHASE2, "epitope_regions.csv")
    df_regions.to_csv(regions_csv, index=False)
    log(f"  Regions table saved → {regions_csv}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9: LBD Region Annotation (focus on residues 40-165)
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 9: Annotating LBD and key residue overlap with epitopes...")

LBD_START = 40
LBD_END   = 165
KEY_RES   = {60, 64, 73, 75, 93, 107, 111, 129, 136, 149}

# Flag each residue as LBD, key residue, and if epitope
df_epi["In_LBD"]    = df_epi["ResNum"].apply(lambda x: LBD_START <= x <= LBD_END)
df_epi["Key_Res"]   = df_epi["ResNum"].apply(lambda x: x in KEY_RES)

lbd_epitopes = df_epi[df_epi["In_LBD"] & df_epi["Above_thresh"]]
key_epitopes = df_epi[df_epi["Key_Res"] & df_epi["Above_thresh"]]
log(f"  Residues in LBD zone: {len(df_epi[df_epi['In_LBD']])}")
log(f"  LBD residues above epitope threshold: {len(lbd_epitopes)}")
log(f"  Key LBD residues that are epitope peaks: {len(key_epitopes)}")

# Top 3 epitope regions (ranked by score, prefer LBD overlap)
if len(df_regions) > 0:
    df_regions["LBD_overlap"] = df_regions.apply(
        lambda r: len([x for x in range(int(r["Start_ResNum"]), int(r["End_ResNum"])+1)
                       if LBD_START <= x <= LBD_END]), axis=1
    )
    df_regions["Priority_score"] = df_regions["Mean_Score"] + 0.2 * df_regions["LBD_overlap"] / 10
    df_top3 = df_regions.sort_values("Priority_score", ascending=False).head(3).copy()
    df_top3["Final_Rank"] = [1, 2, 3][:len(df_top3)]
    top3_csv = os.path.join(PHASE2, "top3_epitope_regions.csv")
    df_top3.to_csv(top3_csv, index=False)
    log(f"\n  TOP 3 EPITOPE REGIONS:")
    for _, row in df_top3.iterrows():
        log(f"    Rank {int(row['Final_Rank'])}: Pos {int(row['Start_Pos'])}-{int(row['End_Pos'])} "
            f"| Seq: {row['Sequence']} | Score: {row['Mean_Score']:.4f} | LBD overlap: {int(row['LBD_overlap'])} res")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 10: Fetch LasR homologs from NCBI for conservation analysis
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 10: Fetching LasR homolog sequences from NCBI...")

homolog_seqs = [SeqRecord(Seq(LasR_seq), id="LasR_2UV0", description="LasR 2UV0 chain E")]
homolog_file = os.path.join(PHASE2, "LasR_homologs.fasta")

try:
    search_handle = Entrez.esearch(db="protein", term="LasR quorum sensing Pseudomonas aeruginosa[Organism]",
                                   retmax=20)
    search_results = Entrez.read(search_handle)
    search_handle.close()
    ids = search_results["IdList"]
    log(f"  NCBI search returned {len(ids)} protein IDs")

    fetched = 0
    for pid in ids[:15]:
        try:
            handle = Entrez.efetch(db="protein", id=pid, rettype="fasta", retmode="text")
            record = SeqIO.read(handle, "fasta")
            handle.close()
            if 100 < len(record.seq) < 400:
                homolog_seqs.append(record)
                fetched += 1
            time.sleep(0.35)
        except:
            pass

    log(f"  Fetched {fetched} valid homolog sequences")
    SeqIO.write(homolog_seqs, homolog_file, "fasta")
    log(f"  Homologs FASTA saved → {homolog_file}")
except Exception as e:
    log(f"  NCBI fetch error: {e} — using query sequence only")
    SeqIO.write(homolog_seqs, homolog_file, "fasta")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 11: Conservation scoring via pairwise alignment
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 11: Computing conservation scores across homologs...")

conservation = np.zeros(len(LasR_seq))
alignment_count = 0

if len(homolog_seqs) > 1:
    for rec in homolog_seqs[1:]:
        try:
            target = str(rec.seq).replace('-', '')
            if len(target) < 50:
                continue
            alns = pairwise2.align.globalms(LasR_seq, target, 2, -1, -5, -0.5)
            if not alns:
                continue
            aln_ref, aln_tgt = alns[0].seqA, alns[0].seqB
            pos = 0
            for r, t in zip(aln_ref, aln_tgt):
                if r != '-' and pos < len(LasR_seq):
                    if r == t:
                        conservation[pos] += 1
                    pos += 1
            alignment_count += 1
        except Exception as e:
            pass

    if alignment_count > 0:
        conservation = conservation / alignment_count
        log(f"  Aligned against {alignment_count} homologs")
        log(f"  Conservation mean: {np.mean(conservation):.3f} | "
            f"highly conserved (>0.8): {np.sum(conservation > 0.8)} residues")
    else:
        conservation = np.ones(len(LasR_seq)) * 0.5
        log("  No valid alignments — using uniform conservation = 0.5")
else:
    conservation = np.ones(len(LasR_seq)) * 0.5
    log("  Only 1 sequence — conservation set to 0.5")

df_epi["Conservation"] = conservation[:len(df_epi)]
df_epi.to_csv(epi_csv, index=False)

conservation_csv = os.path.join(PHASE2, "conservation_scores.csv")
df_epi[["Position","ResNum","AA","Conservation"]].to_csv(conservation_csv, index=False)
log(f"  Conservation data saved → {conservation_csv}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 12: Final integrated scoring (epitope + conservation)
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 12: Final integrated epitope + conservation score...")

z_cons = zscore(conservation[:len(df_epi)])
final_score = 0.7 * zscore(combined) + 0.3 * z_cons
df_epi["Final_Score"] = final_score

final_thresh = np.mean(final_score) + 0.5 * np.std(final_score)
df_epi["Final_Epitope"] = final_score > final_thresh
df_epi.to_csv(epi_csv, index=False)

# Re-identify regions with final score
final_regions = find_epitope_regions(
    final_score, positions, resnums[:len(final_score)], aas, final_thresh
)
if final_regions:
    df_final_regions = pd.DataFrame(final_regions)
    df_final_regions["LBD_overlap"] = df_final_regions.apply(
        lambda r: len([x for x in range(int(r["Start_ResNum"]), int(r["End_ResNum"])+1)
                       if LBD_START <= x <= LBD_END]), axis=1
    )
    df_final_regions["Priority_score"] = (
        df_final_regions["Mean_Score"] + 0.3 * df_final_regions["LBD_overlap"] / 10
    )
    df_final_regions = df_final_regions.sort_values("Priority_score", ascending=False).reset_index(drop=True)
    df_final_top3 = df_final_regions.head(3).copy()
    df_final_top3["Final_Rank"] = [1, 2, 3][:len(df_final_top3)]
    final_top3_csv = os.path.join(PHASE2, "FINAL_top3_epitopes.csv")
    df_final_top3.to_csv(final_top3_csv, index=False)
    log(f"\n  FINAL TOP 3 EPITOPES (Combined score + Conservation):")
    for _, row in df_final_top3.iterrows():
        log(f"    Rank {int(row['Final_Rank'])}: Pos {int(row['Start_Pos'])}-{int(row['End_Pos'])} "
            f"({int(row['Length'])} aa) | Seq: {row['Sequence']} "
            f"| Score: {row['Mean_Score']:.4f} | LBD overlap: {int(row['LBD_overlap'])}")
else:
    log("  No final regions found — reducing threshold")
    df_final_top3 = df_top3 if len(df_regions) > 0 else pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 13: Generating Phase 2 figures...")
plt.style.use('seaborn-v0_8-whitegrid')

pos_arr  = df_epi["Position"].values
resnum_arr = df_epi["ResNum"].values

# ─── Figure 8: All 5 epitope scales side-by-side ─────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(18, 16), sharex=True)
scale_data = [
    (kt_scores,     "Kolaskar-Tongaonkar Antigenicity",   COLORS["primary"],    kt_threshold),
    (parker_scores, "Parker Hydrophilicity",               COLORS["secondary"],  np.mean(parker_scores)),
    (emini_scores,  "Emini Surface Accessibility",         COLORS["green"],      np.mean(emini_scores)),
    (ks_scores,     "Karplus-Schulz Backbone Flexibility", COLORS["purple"],     np.mean(ks_scores)),
    (cf_scores[:len(pos_arr)], "Chou-Fasman Beta-Turn",   COLORS["gold"],       np.mean(cf_scores)),
]
for ax, (scores, title, color, thresh) in zip(axes, scale_data):
    ax.fill_between(pos_arr, scores, alpha=0.25, color=color)
    ax.plot(pos_arr, scores, color=color, linewidth=1.0)
    ax.axhline(thresh, color=COLORS["accent"], linewidth=1.2, linestyle='--', alpha=0.8)
    # Mark LBD zone
    lbd_mask = (resnum_arr >= LBD_START) & (resnum_arr <= LBD_END)
    ax.axvspan(pos_arr[lbd_mask][0] if lbd_mask.any() else 0,
               pos_arr[lbd_mask][-1] if lbd_mask.any() else 0,
               alpha=0.08, color=COLORS["highlight"], label="LBD zone")
    ax.set_ylabel(title, fontsize=8.5, color=color, fontweight='bold')
    ax.tick_params(axis='y', labelsize=8)
axes[-1].set_xlabel("Residue Position", fontsize=11)
fig.suptitle("LasR Epitope Prediction — All 5 Physicochemical Scales\n(LBD zone highlighted in orange, threshold as red dashes)",
             fontsize=13, fontweight='bold', color=COLORS["primary"], y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "Fig8_epitope_all_scales.png"), dpi=150, bbox_inches='tight')
plt.close()
log("  Fig8_epitope_all_scales.png saved")

# ─── Figure 9: Combined epitope score ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 5))
above = df_epi["Above_thresh"].values
ax.fill_between(pos_arr, combined,
                where=above,  alpha=0.5, color=COLORS["accent"],  label="Above threshold (epitope)")
ax.fill_between(pos_arr, combined,
                where=~above, alpha=0.3, color=COLORS["secondary"], label="Below threshold")
ax.plot(pos_arr, combined, color=COLORS["primary"], linewidth=1.0)
ax.axhline(combined_threshold, color=COLORS["accent"], linewidth=1.5, linestyle='--',
           label=f"Threshold = {combined_threshold:.3f}")
# LBD shading
lbd_mask = (resnum_arr >= LBD_START) & (resnum_arr <= LBD_END)
if lbd_mask.any():
    ax.axvspan(pos_arr[lbd_mask][0], pos_arr[lbd_mask][-1],
               alpha=0.1, color=COLORS["highlight"], label=f"LBD (res {LBD_START}-{LBD_END})")
# Mark top regions
if len(df_regions) > 0:
    for _, row in df_regions.head(3).iterrows():
        ax.annotate(f"E{int(row['Region_ID'])}",
                    xy=(int(row['Peak_Pos']), float(row['Max_Score'])),
                    xytext=(int(row['Peak_Pos']), float(row['Max_Score']) + 0.3),
                    fontsize=9, fontweight='bold', color=COLORS["accent"],
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color=COLORS["accent"], lw=1.0))
ax.set_xlabel("Residue Position", fontsize=12)
ax.set_ylabel("Combined Epitope Score (z-normalized)", fontsize=12)
ax.set_title("LasR Combined B-cell Epitope Score\n(KT×0.30 + Emini×0.25 + Parker×0.20 + KS×0.15 + CF×0.10)",
             fontsize=13, fontweight='bold', color=COLORS["primary"])
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "Fig9_combined_epitope_score.png"), dpi=150, bbox_inches='tight')
plt.close()
log("  Fig9_combined_epitope_score.png saved")

# ─── Figure 10: Conservation profile ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
# Top: conservation
cons_arr = df_epi["Conservation"].values
ax1.fill_between(pos_arr, cons_arr, alpha=0.4, color=COLORS["green"])
ax1.plot(pos_arr, cons_arr, color=COLORS["primary"], linewidth=1.0)
ax1.axhline(0.8, color=COLORS["accent"],   linewidth=1.5, linestyle='--', label="High conservation (0.8)")
ax1.axhline(0.5, color=COLORS["highlight"],linewidth=1.0, linestyle=':',  label="Moderate conservation (0.5)")
lbd_mask = (resnum_arr >= LBD_START) & (resnum_arr <= LBD_END)
if lbd_mask.any():
    ax1.axvspan(pos_arr[lbd_mask][0], pos_arr[lbd_mask][-1],
                alpha=0.1, color=COLORS["highlight"], label="LBD zone")
ax1.set_ylabel("Conservation Score", fontsize=11)
ax1.set_title(f"Conservation Profile — LasR (aligned against {alignment_count} homologs)",
              fontsize=12, fontweight='bold', color=COLORS["primary"])
ax1.legend(fontsize=9)
# Bottom: combined epitope
ax2.fill_between(pos_arr, df_epi["Final_Score"].values,
                 where=df_epi["Final_Epitope"].values, alpha=0.5,
                 color=COLORS["accent"], label="Final epitope regions")
ax2.fill_between(pos_arr, df_epi["Final_Score"].values,
                 where=~df_epi["Final_Epitope"].values, alpha=0.25,
                 color=COLORS["secondary"])
ax2.plot(pos_arr, df_epi["Final_Score"].values, color=COLORS["primary"], linewidth=1.0)
ax2.axhline(final_thresh, color=COLORS["accent"], linewidth=1.5, linestyle='--',
            label=f"Threshold = {final_thresh:.3f}")
if lbd_mask.any():
    ax2.axvspan(pos_arr[lbd_mask][0], pos_arr[lbd_mask][-1], alpha=0.1, color=COLORS["highlight"])
ax2.set_xlabel("Residue Position", fontsize=11)
ax2.set_ylabel("Final Score (Epitope + Conservation)", fontsize=11)
ax2.set_title("Final Integrated Score (70% Epitope + 30% Conservation)",
              fontsize=12, fontweight='bold', color=COLORS["primary"])
ax2.legend(fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "Fig10_conservation_and_final_score.png"), dpi=150, bbox_inches='tight')
plt.close()
log("  Fig10_conservation_and_final_score.png saved")

# ─── Figure 11: Top 3 epitope regions zoom ───────────────────────────────────
if len(df_final_top3) > 0:
    n_top = len(df_final_top3)
    fig, axes = plt.subplots(1, n_top, figsize=(6*n_top, 5))
    if n_top == 1:
        axes = [axes]
    rank_colors = [COLORS["accent"], COLORS["secondary"], COLORS["green"]]
    for ax, (_, row), col in zip(axes, df_final_top3.iterrows(), rank_colors):
        sp  = int(row["Start_Pos"]) - 3
        ep  = int(row["End_Pos"])   + 3
        mask = (pos_arr >= sp) & (pos_arr <= ep)
        zoom_pos   = pos_arr[mask]
        zoom_final = df_epi["Final_Score"].values[mask]
        zoom_cons  = df_epi["Conservation"].values[mask]
        ax2 = ax.twinx()
        ax.fill_between(zoom_pos, zoom_final, alpha=0.6, color=col,
                        label="Epitope score")
        ax.plot(zoom_pos, zoom_final, color=col, linewidth=2)
        ax2.plot(zoom_pos, zoom_cons, color=COLORS["gold"],
                 linewidth=1.5, linestyle='--', label="Conservation")
        ax2.set_ylabel("Conservation", fontsize=9, color=COLORS["gold"])
        ax2.tick_params(axis='y', labelcolor=COLORS["gold"])
        ax.axhline(final_thresh, color='gray', linewidth=1, linestyle=':')
        seq_str = str(row["Sequence"])
        ax.set_title(
            f"Rank {int(row['Final_Rank'])} Epitope\n"
            f"Pos {int(row['Start_Pos'])}–{int(row['End_Pos'])} "
            f"({int(row['Length'])} aa)\n{seq_str[:20]}{'...' if len(seq_str)>20 else ''}",
            fontsize=10, fontweight='bold', color=col
        )
        ax.set_xlabel("Residue Position", fontsize=9)
        ax.set_ylabel("Final Epitope Score", fontsize=9)
    fig.suptitle("Top 3 Epitope Regions — Zoomed View\n(Epitope score + Conservation overlay)",
                 fontsize=13, fontweight='bold', color=COLORS["primary"])
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "Fig11_top3_epitopes_zoom.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log("  Fig11_top3_epitopes_zoom.png saved")

# ─── Figure 12: Heatmap — all scores across LBD zone ─────────────────────────
lbd_df = df_epi[df_epi["In_LBD"]].copy().reset_index(drop=True)
if len(lbd_df) > 0:
    heat_cols = ["KT_score", "Parker", "Emini", "KS_flex", "CF_turn", "Conservation"]
    heat_data = lbd_df[heat_cols].copy()
    for col in heat_cols:
        heat_data[col] = (heat_data[col] - heat_data[col].min()) / \
                         (heat_data[col].max() - heat_data[col].min() + 1e-9)
    heat_data.index = lbd_df["ResNum"].astype(int).astype(str) + "\n" + lbd_df["AA"]

    fig, ax = plt.subplots(figsize=(14, max(6, len(lbd_df) * 0.18)))
    sns.heatmap(heat_data.T, ax=ax, cmap="YlOrRd", linewidths=0.3,
                linecolor='white', cbar_kws={"label": "Normalized Score"},
                xticklabels=3, yticklabels=True)
    ax.set_xlabel("Residue (LBD zone)", fontsize=10)
    ax.set_title("Epitope Score Heatmap — LasR LBD Zone (Res 40–165)\n"
                 "All scales normalized 0→1", fontsize=12,
                 fontweight='bold', color=COLORS["primary"])
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=7, rotation=90)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "Fig12_LBD_epitope_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log("  Fig12_LBD_epitope_heatmap.png saved")

# ─── Figure 13: Top 3 epitopes summary table ─────────────────────────────────
if len(df_final_top3) > 0:
    fig, ax = plt.subplots(figsize=(16, 3.5))
    ax.axis('off')
    cols_show = ["Final_Rank","Sequence","Start_ResNum","End_ResNum","Length",
                 "Mean_Score","Max_Score","LBD_overlap"]
    col_labels = ["Rank","Sequence","Start Res","End Res","Length",
                  "Mean Score","Max Score","LBD Overlap"]
    data_show  = df_final_top3[cols_show].values.tolist()
    data_show  = [[str(round(v, 4)) if isinstance(v, float) else str(int(v))
                   if isinstance(v, (int, np.integer, np.floating)) else str(v)
                   for v in row] for row in data_show]
    tbl = ax.table(cellText=data_show, colLabels=col_labels, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.3, 2.5)
    rank_colors_table = ["#FADBD8", "#D6EAF8", "#D5F5E3"]
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor(COLORS["primary"])
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(data_show)+1):
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(rank_colors_table[i-1] if i-1 < len(rank_colors_table) else "white")
    ax.set_title("Final Top 3 Epitope Regions — LasR Antibody Target Sites",
                 fontsize=13, fontweight='bold', color=COLORS["primary"], pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "Fig13_top3_epitopes_table.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log("  Fig13_top3_epitopes_table.png saved")

# ─── Figure 14: Sequence map with epitope annotation ─────────────────────────
fig, ax = plt.subplots(figsize=(18, 4))
seq_len = len(LasR_seq)
# Background
ax.barh(0, seq_len, height=0.4, color="#E8F4F8", edgecolor="gray", linewidth=0.8)
# LBD zone
lbd_start_idx = min([i for i, r in enumerate(resnums) if r >= LBD_START], default=0)
lbd_end_idx   = max([i for i, r in enumerate(resnums) if r <= LBD_END], default=0)
ax.barh(0, lbd_end_idx - lbd_start_idx, left=lbd_start_idx,
        height=0.4, color=COLORS["highlight"], alpha=0.5, label=f"LBD (res {LBD_START}–{LBD_END})")
# Epitope regions
epi_colors_map = {1: COLORS["accent"], 2: COLORS["secondary"], 3: COLORS["green"]}
if len(df_final_top3) > 0:
    for _, row in df_final_top3.iterrows():
        rk     = int(row["Final_Rank"])
        s_pos  = int(row["Start_Pos"]) - 1
        length = int(row["Length"])
        ax.barh(0, length, left=s_pos, height=0.4,
                color=epi_colors_map.get(rk, COLORS["purple"]), alpha=0.85,
                label=f"Epitope #{rk} (pos {int(row['Start_Pos'])}–{int(row['End_Pos'])})")
        ax.text(s_pos + length/2, 0,
                f"#{rk}", ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
# Key residues
for rnum in KEY_RES:
    pos_idx = next((i for i, r in enumerate(resnums) if r == rnum), None)
    if pos_idx is not None:
        ax.plot(pos_idx, 0.25, 'v', color=COLORS["primary"], markersize=7)
        ax.text(pos_idx, 0.35, str(rnum), ha='center', fontsize=6.5,
                color=COLORS["primary"], fontweight='bold')
ax.set_xlim(-5, seq_len + 5)
ax.set_ylim(-0.5, 0.8)
ax.set_xlabel("Residue Position", fontsize=11)
ax.set_yticks([])
ax.set_title("LasR Sequence Map — Epitope Regions & LBD Annotation\n"
             "(triangles = key HSL-binding residues)",
             fontsize=13, fontweight='bold', color=COLORS["primary"])
ax.legend(fontsize=9, loc='upper right')
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "Fig14_sequence_epitope_map.png"), dpi=150, bbox_inches='tight')
plt.close()
log("  Fig14_sequence_epitope_map.png saved")

# ─── Figure 15: Radar/spider chart for top 3 epitopes ────────────────────────
if len(df_final_top3) > 0:
    categories = ["KT Antigenicity", "Parker Hydrophilicity", "Emini Accessibility",
                  "KS Flexibility", "Conservation", "LBD Overlap"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, size=10)

    rank_colors_radar = [COLORS["accent"], COLORS["secondary"], COLORS["green"]]
    for (_, row), col in zip(df_final_top3.iterrows(), rank_colors_radar):
        sp, ep = int(row["Start_Pos"])-1, int(row["End_Pos"])
        sub = df_epi.iloc[sp:ep]
        if len(sub) == 0:
            continue
        lbd_ov = float(row["LBD_overlap"]) / max(10, float(row["Length"]))
        vals = [
            float(sub["KT_score"].mean()),
            float(sub["Parker"].mean() + 3) / 6,
            float(sub["Emini"].mean() * 10),
            float(sub["KS_flex"].mean()),
            float(sub["Conservation"].mean()),
            min(1.0, lbd_ov)
        ]
        # Normalize 0-1
        vals = [max(0, min(1, v)) for v in vals]
        vals += vals[:1]
        ax.plot(angles, vals, color=col, linewidth=2,
                label=f"Epitope #{int(row['Final_Rank'])} (pos {int(row['Start_Pos'])}–{int(row['End_Pos'])})")
        ax.fill(angles, vals, color=col, alpha=0.15)

    ax.set_ylim(0, 1)
    ax.set_title("Epitope Quality Radar Chart — Top 3 Candidates",
                 size=13, fontweight='bold', color=COLORS["primary"], y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "Fig15_epitope_radar_chart.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log("  Fig15_epitope_radar_chart.png saved")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 14: Write Phase 2 report
# ═══════════════════════════════════════════════════════════════════════════════
log("\nSTEP 14: Writing Phase 2 summary report...")

report = [
    "=" * 70,
    "PHASE 2 SUMMARY REPORT — LasR Epitope Mapping",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "=" * 70,
    "",
    "1. SEQUENCE EXTRACTED FROM 2UV0",
    "-" * 40,
    f"   Chain: E | Length: {len(LasR_seq)} residues",
    f"   Sequence: {LasR_seq[:40]}...",
    "",
    "2. EPITOPE PREDICTION METHODS APPLIED",
    "-" * 40,
    "   1. Kolaskar-Tongaonkar Antigenicity Scale (weight: 30%)",
    "   2. Parker Hydrophilicity Scale            (weight: 25%)",
    "   3. Emini Surface Accessibility Scale      (weight: 20%)",
    "   4. Karplus-Schulz Backbone Flexibility    (weight: 15%)",
    "   5. Chou-Fasman Beta-Turn Propensity       (weight: 10%)",
    "",
    "3. CONSERVATION ANALYSIS",
    "-" * 40,
    f"   Homologs fetched from NCBI: {len(homolog_seqs)-1}",
    f"   Alignments completed:       {alignment_count}",
    f"   Mean conservation score:    {np.mean(conservation):.3f}",
    f"   Highly conserved (>0.8):    {np.sum(conservation > 0.8)} residues",
    "",
    "4. FINAL TOP 3 EPITOPE REGIONS",
    "-" * 40,
]
if len(df_final_top3) > 0:
    for _, row in df_final_top3.iterrows():
        report += [
            f"",
            f"   RANK {int(row['Final_Rank'])}:",
            f"     Position:    {int(row['Start_Pos'])} – {int(row['End_Pos'])} (chain E numbering)",
            f"     Residue #:   {int(row['Start_ResNum'])} – {int(row['End_ResNum'])} (PDB numbering)",
            f"     Length:      {int(row['Length'])} amino acids",
            f"     Sequence:    {row['Sequence']}",
            f"     Mean Score:  {float(row['Mean_Score']):.4f}",
            f"     Max Score:   {float(row['Max_Score']):.4f}",
            f"     LBD Overlap: {int(row['LBD_overlap'])} residues",
        ]
report += [
    "",
    "5. OUTPUT FILES",
    "-" * 40,
    "   FASTA:  LasR_2UV0_chainE.fasta",
    "   FASTA:  LasR_homologs.fasta",
    "   CSV:    epitope_scores_all_residues.csv",
    "   CSV:    epitope_regions.csv",
    "   CSV:    top3_epitope_regions.csv",
    "   CSV:    conservation_scores.csv",
    "   CSV:    FINAL_top3_epitopes.csv",
    "   Figs:   Fig8–Fig15 in 08_Results_Figures/Phase2/",
    "",
    "6. NEXT STEP → PHASE 3: Antibody Design",
    "   RFdiffusion, ProteinMPNN, ABodyBuilder2",
    "=" * 70,
]
report_path = os.path.join(PHASE2, "PHASE2_REPORT.txt")
with open(report_path, "w") as f:
    f.write("\n".join(report))
log(f"  Phase 2 report saved → {report_path}")

results_json = {
    "phase": 2,
    "title": "Epitope Mapping",
    "completed": datetime.now().isoformat(),
    "sequence_length": len(LasR_seq),
    "methods": ["Kolaskar-Tongaonkar", "Parker", "Emini", "Karplus-Schulz", "Chou-Fasman"],
    "homologs_aligned": alignment_count,
    "total_regions_found": len(df_regions) if len(df_regions) > 0 else 0,
    "top3_epitopes": df_final_top3[["Final_Rank","Sequence","Start_ResNum","End_ResNum",
                                    "Length","Mean_Score","LBD_overlap"]].to_dict("records") if len(df_final_top3) > 0 else [],
    "figures_generated": 8,
    "status": "COMPLETED"
}
with open(os.path.join(PHASE2, "phase2_results.json"), "w") as f:
    json.dump(results_json, f, indent=2)

log("\n" + "=" * 70)
log("PHASE 2: COMPLETED SUCCESSFULLY")
log(f"  Figures: {FIGS}")
log(f"  Results: {PHASE2}")
log(f"  Log:     {LOG_FILE}")
log("=" * 70)
