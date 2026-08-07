#!/usr/bin/Rscript
# ==============================================================================
# RBPs_energy_pvalue.R
# ------------------------------------------------------------------------------
# R script using ggpubr (ggboxplot and ggpaired) to compute t-tests and generate
# publication-quality boxplots and paired line plots for Total Torsional Energy.
#
# Direct JSON loading to preserve exact string formats (e.g. 4E78).
# Saves both TIFF (600 DPI LZW) and PNG (600 DPI) formats.
#
# Dynamic Filename Formatting:
#   total_energy_<subset>_<region>_boxplot.tiff / .png
#   total_energy_<subset>_<region>_paired.tiff / .png
#   where subset in {pdb, aa, aromatic} and region in {int, nint}
#
# Usage:
#   Rscript script/r_scripts/RBPs_energy_pvalue.R [json_file] [region_mode: INT|NINT|AA_INT|AA_NINT|AROMATIC_INT|AROMATIC_NINT]
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggpubr)
  library(jsonlite)
  library(tools)
})

args <- commandArgs(trailingOnly = TRUE)
json_file <- if (length(args) >= 1) args[1] else "/home/labuser/Projects/PhD_projects/RBPs_rotamer_lib/output_files/torsion_potential_energy_summary_187.json"
region_mode <- if (length(args) >= 2) toupper(args[2]) else "INT"

if (!file.exists(json_file)) {
  cat("Error: JSON input file not found:", json_file, "\n")
  q(status = 1)
}

cat("Processing JSON data file:", json_file, "(Region Mode:", region_mode, ")\n")

# Load JSON directly to preserve PDB IDs and string formats
data_list <- jsonlite::fromJSON(json_file)
df_raw <- as.data.frame(data_list)

fig_dir1 <- "/home/labuser/Projects/PhD_projects/RBPs_rotamer_lib/figure/energy_pvalues"
fig_dir2 <- "/home/labuser/Projects/PhD_projects/RBPs_rotamer_lib/figures/energy_pvalues"
dir.create(fig_dir1, showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir2, showWarnings = FALSE, recursive = TRUE)

is_aa_level <- grepl("amino_acid", json_file, ignore.case = TRUE) || grepl("AA", region_mode, ignore.case = TRUE) || grepl("AROMATIC", region_mode, ignore.case = TRUE)

if (is_aa_level) {
  if (grepl("AROMATIC", region_mode, ignore.case = TRUE)) {
    df_raw <- df_raw[df_raw$AA %in% c("HIS", "PHE", "TRP", "TYR"), ]
    subset_tag <- "aromatic"
    mlab <- "Aromatic Interface"
  } else {
    subset_tag <- "aa"
    mlab <- "Amino Acid Interface"
  }
  
  if (grepl("NINT", region_mode, ignore.case = TRUE)) {
    u_vals <- df_raw$NU_DELG
    b_vals <- df_raw$NB_DELG
    res_ids <- df_raw$AA
    region_tag <- "nint"
    mlab <- gsub("Interface", "Non-interface", mlab)
  } else {
    u_vals <- df_raw$IU_DELG
    b_vals <- df_raw$IB_DELG
    res_ids <- df_raw$AA
    region_tag <- "int"
  }
} else {
  subset_tag <- "pdb"
  res_ids <- df_raw$PDB_ID
  if (grepl("NINT", region_mode, ignore.case = TRUE)) {
    u_vals <- df_raw$NU_DELG
    b_vals <- df_raw$NB_DELG
    region_tag <- "nint"
    mlab <- "Non-interface"
  } else {
    u_vals <- df_raw$IU_DELG
    b_vals <- df_raw$IB_DELG
    region_tag <- "int"
    mlab <- "Interface"
  }
}

# Construct long dataframe for ggpubr
mydata <- data.frame(
  Res = rep(res_ids, 2),
  Avg.E = c(u_vals, b_vals),
  Form = factor(rep(c("U", "B"), each = length(res_ids)), levels = c("U", "B"))
)

mydata <- mydata[!is.na(mydata$Avg.E), ]

# Dynamic publication filename stem
filename_base <- paste0("total_energy_", subset_tag, "_", region_tag)

# 1. ggboxplot using ggpubr + lancet palette + stat_compare_means (t.test)
p_box <- ggboxplot(
  mydata, x = "Form", y = "Avg.E", xlab = mlab, ylab = "E (Kcal/mol)",
  color = "Form", palette = "lancet", add = "jitter"
) +
  stat_compare_means(method = "t.test", label.x.npc = "center", label.y.npc = "top") +
  font("xlab", size = 16, face = "bold") +
  font("ylab", size = 16, face = "bold") +
  font("legend.title", face = "bold", size = 16) +
  theme(plot.title = element_blank())

# Save PNG and TIFF
ggsave(file.path(fig_dir1, paste0(filename_base, "_boxplot.png")), plot = p_box, width = 4.5, height = 4.5, dpi = 600)
ggsave(file.path(fig_dir2, paste0(filename_base, "_boxplot.png")), plot = p_box, width = 4.5, height = 4.5, dpi = 600)
ggsave(file.path(fig_dir1, paste0(filename_base, "_boxplot.tiff")), plot = p_box, width = 4.5, height = 4.5, dpi = 600, device = "tiff", compression = "lzw")
ggsave(file.path(fig_dir2, paste0(filename_base, "_boxplot.tiff")), plot = p_box, width = 4.5, height = 4.5, dpi = 600, device = "tiff", compression = "lzw")

# 2. ggpaired using ggpubr + lancet palette + stat_compare_means (paired t.test)
p_paired <- ggpaired(
  mydata, x = "Form", y = "Avg.E", xlab = mlab, ylab = "E (Kcal/mol)",
  color = "Form", palette = "lancet", line.color = "gray", line.size = 0.4, add = "jitter"
) +
  stat_compare_means(method = "t.test", paired = TRUE, label.x.npc = "center", label.y.npc = "top") +
  font("xlab", size = 16, face = "bold") +
  font("ylab", size = 16, face = "bold") +
  font("legend.title", face = "bold", size = 16) +
  theme(plot.title = element_blank())

# Save PNG and TIFF
ggsave(file.path(fig_dir1, paste0(filename_base, "_paired.png")), plot = p_paired, width = 4.5, height = 4.5, dpi = 600)
ggsave(file.path(fig_dir2, paste0(filename_base, "_paired.png")), plot = p_paired, width = 4.5, height = 4.5, dpi = 600)
ggsave(file.path(fig_dir1, paste0(filename_base, "_paired.tiff")), plot = p_paired, width = 4.5, height = 4.5, dpi = 600, device = "tiff", compression = "lzw")
ggsave(file.path(fig_dir2, paste0(filename_base, "_paired.tiff")), plot = p_paired, width = 4.5, height = 4.5, dpi = 600, device = "tiff", compression = "lzw")

cat("Successfully generated publication-quality plots (ggpubr Lancet, PNG + TIFF):", filename_base, "\n")
