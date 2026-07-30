"""Reference data for the mechanism module: a curated human transcription-factor list and
TRRUST/literature-informed TF regulons (TF -> known target genes), focused on the cancer /
CAF / ECM / EMT / angiogenesis / immune / proliferation programs SIDISH surfaces.

These are curated, offline references so the mechanism narrative is grounded, not invented.
On a lab server they can be swapped for a full regulon DB (DoRothEA / TRRUST / ChEA) or a
live Enrichr TF-enrichment query for arbitrary datasets.
"""

# --- human transcription factors (curated; used to flag TFs that are themselves perturbed) ---
HUMAN_TFS = {
    "TP53", "MYC", "MYCN", "JUN", "JUNB", "JUND", "FOS", "FOSB", "FOSL1", "FOSL2", "ATF3",
    "EGR1", "EGR2", "EGR3", "KLF2", "KLF4", "KLF5", "KLF6", "SP1", "SP3", "YY1",
    "STAT1", "STAT2", "STAT3", "STAT4", "STAT5A", "STAT5B", "STAT6", "IRF1", "IRF2", "IRF3",
    "IRF4", "IRF5", "IRF7", "IRF8", "NFKB1", "NFKB2", "RELA", "RELB", "REL", "NFATC1", "NFATC2",
    "E2F1", "E2F2", "E2F3", "E2F4", "FOXM1", "FOXO1", "FOXO3", "FOXP3", "FOXA1", "FOXA2",
    "FOXC1", "FOXC2", "FOXQ1", "MYB", "MYBL2", "RUNX1", "RUNX2", "RUNX3", "CBFB",
    "SNAI1", "SNAI2", "TWIST1", "TWIST2", "ZEB1", "ZEB2", "PRRX1", "TCF3", "TCF4", "TCF7",
    "TCF7L1", "TCF7L2", "LEF1", "TEAD1", "TEAD2", "TEAD3", "TEAD4", "SRF", "MYOCD", "MRTFA",
    "SMAD1", "SMAD2", "SMAD3", "SMAD4", "SMAD7", "ETS1", "ETS2", "ELK1", "ELK3", "ETV1",
    "ETV4", "ETV5", "GABPA", "SPI1", "SPIB", "GATA1", "GATA2", "GATA3", "GATA4", "GATA6",
    "TAL1", "SOX2", "SOX4", "SOX9", "SOX10", "SOX17", "SOX18", "POU5F1", "NANOG", "KLF1",
    "HIF1A", "EPAS1", "ARNT", "HAND1", "HAND2", "MEF2A", "MEF2C", "MEF2D", "NR3C1", "ESR1",
    "ESR2", "AR", "PGR", "PPARG", "PPARA", "RXRA", "NR2F2", "THRB", "VDR", "RARA", "RARB",
    "CEBPA", "CEBPB", "CEBPD", "CEBPE", "MAF", "MAFB", "BACH1", "NFE2L2", "NR4A1",
    "TFAP2A", "TFAP2C", "AP1", "CREB1", "CREBBP", "ATF4", "XBP1", "DDIT3", "USF1", "MAX",
    "MXD1", "MNT", "MITF", "TFEB", "TFE3", "SREBF1", "SREBF2", "NR1H2", "NR1H3",
    "PAX3", "PAX5", "PAX6", "PAX8", "MEIS1", "PBX1", "HOXA9", "HOXB7", "HOXA10", "CDX2",
    "GLI1", "GLI2", "GLI3", "RBPJ", "HEY1", "HEY2", "HES1", "HES5", "NOTCH1", "NOTCH3",
    "TBX2", "TBX3", "TBX21", "EOMES", "BCL6", "BCL11A", "IKZF1", "PRDM1", "BATF", "MAFF",
    "ID1", "ID2", "ID3", "NR2F1", "ONECUT1", "HNF1A", "HNF4A", "FOXF1", "FOXF2", "WT1",
    "MSX1", "MSX2", "DLX2", "SIX1", "EYA1", "TCF21", "OSR1", "NKX2-1", "GRHL2", "OVOL2",
    "ELF3", "EHF", "SPDEF", "KLF15", "ATOH1", "NEUROD1", "ASCL1", "POU2F1", "OTX2",
}

# --- TF -> target genes (TRRUST/literature-informed regulons). Used for TF-enrichment of an
#     affected gene set (which upstream TFs' targets are over-represented). ---
TF_REGULONS = {
    # EMT / invasion
    "TWIST1": ["CDH1", "CDH2", "MMP2", "MMP9", "SNAI2", "FN1", "VIM", "AKT2", "BMI1", "MIR10B"],
    "SNAI1":  ["CDH1", "CLDN1", "OCLN", "MMP2", "FN1", "VIM", "MUC1", "CDH2", "TJP1"],
    "SNAI2":  ["CDH1", "MMP2", "MMP9", "BMP4", "KRT18", "CDH2", "FN1"],
    "ZEB1":   ["CDH1", "CRB3", "MIR200A", "EPCAM", "ESRP1", "COL1A1", "VIM"],
    "PRRX1":  ["FN1", "COL1A1", "TNC", "POSTN"],
    # myofibroblast / contractile (CAF core)
    "SRF":    ["ACTA2", "MYL9", "TAGLN", "CNN1", "TPM1", "TPM2", "VCL", "MYH11", "FLNA", "ACTB"],
    "MYOCD":  ["ACTA2", "MYH11", "CNN1", "TAGLN", "MYL9", "SMTN"],
    "MRTFA":  ["ACTA2", "MYL9", "TAGLN", "CNN1", "CTGF", "CYR61"],
    # Hippo / mechanotransduction
    "TEAD1":  ["CTGF", "CYR61", "ANKRD1", "AMOTL2", "MYC", "BIRC5", "AXL", "THBS1"],
    "TEAD4":  ["CTGF", "CYR61", "ANKRD1", "MYC", "CCN1", "CCN2"],
    # TGF-beta / fibrosis
    "SMAD3":  ["SERPINE1", "COL1A1", "COL1A2", "COL3A1", "CTGF", "TIMP1", "MMP2", "JUNB", "SKIL"],
    "SMAD4":  ["SERPINE1", "CDKN1A", "COL1A1", "SMAD7", "JUNB", "ID1"],
    # ECM remodeling / AP-1
    "FOSL1":  ["MMP1", "MMP9", "VIM", "PLAU", "IL6", "CD44", "UPA"],
    "JUN":    ["MMP1", "MMP9", "CCND1", "IL6", "VEGFA", "PLAU", "TIMP1", "CD44"],
    "ETS1":   ["MMP1", "MMP3", "MMP9", "VEGFA", "PLAU", "ICAM1", "ANGPT2", "FLT1"],
    "RUNX2":  ["SPP1", "MMP13", "COL1A1", "IBSP", "BGLAP", "VEGFA", "MMP9"],
    # angiogenesis / hypoxia
    "HIF1A":  ["VEGFA", "LOX", "CA9", "SLC2A1", "PGK1", "LDHA", "PDK1", "ANGPT2", "SERPINE1", "MET"],
    "EPAS1":  ["VEGFA", "FLT1", "KDR", "ANGPT2", "EPO", "OCT4"],
    # inflammation / immune
    "RELA":   ["IL6", "CXCL8", "CCL2", "TNF", "ICAM1", "VCAM1", "MMP9", "BCL2", "BIRC3", "NFKBIA"],
    "NFKB1":  ["IL6", "CXCL8", "CCL2", "TNF", "PTGS2", "SOD2", "BCL2L1", "CCL5"],
    "STAT3":  ["IL6", "VEGFA", "MMP2", "MMP9", "BCL2", "MCL1", "MYC", "CCND1", "SOCS3", "HIF1A"],
    "STAT1":  ["IRF1", "GBP1", "CXCL9", "CXCL10", "HLA-A", "HLA-B", "TAP1", "PSMB9", "B2M"],
    "IRF1":   ["CXCL9", "CXCL10", "TAP1", "PSMB9", "GBP1", "CIITA", "IL12B"],
    "SPI1":   ["CSF1R", "ITGAM", "CD68", "FCGR1A", "MPO", "SPP1", "LYZ"],
    # proliferation / cell cycle
    "E2F1":   ["CCNE1", "CDK1", "MCM2", "MCM3", "PCNA", "CDC6", "TYMS", "RRM2", "MYBL2"],
    "FOXM1":  ["CDK1", "PLK1", "CCNB1", "CCNB2", "AURKB", "CENPF", "BIRC5", "TOP2A", "CDC20"],
    "MYC":    ["CDK4", "CCND2", "CAD", "ODC1", "NCL", "LDHA", "TERT", "CDKN1A", "BAX"],
    # tumor suppressor
    "TP53":   ["CDKN1A", "MDM2", "BAX", "BBC3", "PMAIP1", "GADD45A", "SFN", "SERPINB5", "TP53I3"],
    # steroid / breast
    "ESR1":   ["PGR", "GREB1", "TFF1", "CCND1", "MYC", "BCL2", "CXCL12"],
}
