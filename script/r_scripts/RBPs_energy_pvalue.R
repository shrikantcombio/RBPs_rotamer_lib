#!/usr/bin/Rscript
# ==============================================================================
# RBPs_energy_pvalue.R
# ------------------------------------------------------------------------------
# Refactored R script matching sample_R_plot aesthetics for Total Torsional Energy.
# Loads directly from JSON summary files (preserving exact PDB IDs and string formats).
#
# Aesthetics & Parameters (matching sample_R_plot):
#   - Lancet palette: U = #00468B (Dark Blue), B = #ED0000 (Lancet Red)
#   - Independent & paired t-test p-value annotation top-center
#   - font("xlab", size = 16), font("ylab", size = 16), font("axis.text", size = 14)
#   - Line color = "gray", line size = 0.4
#   - Title REMOVED (plot.title = element_blank())
#
# Dynamic Filename Formatting:
#   total_energy_<subset>_<region>_boxplot.png / paired.png / .tiff
#   where subset in {pdb, aa, aromatic} and region in {int, nint}
#
# Usage:
#   Rscript script/r_scripts/RBPs_energy_pvalue.R [json_file] [region_mode: INT|NINT|AA_INT|AA_NINT|AROMATIC_INT|AROMATIC_NINT]
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
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

# Construct long dataframe for ggplot
mydata <- data.frame(
  Res = rep(res_ids, 2),
  Avg.E = c(u_vals, b_vals),
  Form = factor(rep(c("U", "B"), each = length(res_ids)), levels = c("U", "B"))
)

mydata <- mydata[!is.na(mydata$Avg.E), ]

b_vec <- mydata$Avg.E[mydata$Form == "B"]
u_vec <- mydata$Avg.E[mydata$Form == "U"]

# Compute two-tailed independent and paired t-tests
tt_ind <- t.test(b_vec, u_vec, paired = FALSE)
tt_pair <- t.test(b_vec, u_vec, paired = TRUE)

pval_ind_str <- paste0("p = ", format.pval(tt_ind$p.value, digits = 3))
pval_pair_str <- paste0("p = ", format.pval(tt_pair$p.value, digits = 3))

max_y <- max(mydata$Avg.E, na.rm = TRUE)
min_y <- min(mydata$Avg.E, na.rm = TRUE)
y_top <- max_y + (max_y - min_y) * 0.12

# Dynamic publication filename stem
filename_base <- paste0("total_energy_", subset_tag, "_", region_tag)

# Lancet palette colors (Dark Blue for U, Lancet Red for B)
lancet_colors <- c("U" = "#00468B", "B" = "#ED0000")

# 1. Boxplot (matching sample_R_plot ggboxplot + Lancet palette + jitter)
p_box <- ggplot(mydata, aes(x = Form, y = Avg.E, color = Form, fill = Form)) +
  geom_boxplot(alpha = 0.5, outlier.shape = NA, width = 0.45, linewidth = 0.6) +
  geom_jitter(width = 0.2, alpha = 0.7, size = 1.8) +
  annotate("text", x = 1.5, y = y_top, label = pval_ind_str, size = 5, fontface = "bold") +
  scale_color_manual(values = lancet_colors) +
  scale_fill_manual(values = lancet_colors) +
  scale_x_discrete(labels = c("U" = "U", "B" = "B")) +
  labs(x = mlab, y = "E (Kcal/mol)") +
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_blank(),
    axis.title.x = element_text(size = 16, face = "bold"),
    axis.title.y = element_text(size = 16, face = "bold"),
    axis.text = element_text(size = 14, face = "bold", color = "black"),
    legend.position = "none"
  )

ggsave(file.path(fig_dir1, paste0(filename_base, "_boxplot.png")), plot = p_box, width = 4.5, height = 4.5, dpi = 600)
ggsave(file.path(fig_dir2, paste0(filename_base, "_boxplot.png")), plot = p_box, width = 4.5, height = 4.5, dpi = 600)

# 2. Paired Plot (matching sample_R_plot ggpaired + line.color="gray", line.size=0.4 + jitter)
p_paired <- ggplot(mydata, aes(x = Form, y = Avg.E, group = Res, color = Form)) +
  geom_line(color = "gray", linewidth = 0.4, alpha = 0.7) +
  geom_point(size = 2.2, alpha = 0.85) +
  annotate("text", x = 1.5, y = y_top, label = pval_pair_str, size = 5, fontface = "bold") +
  scale_color_manual(values = lancet_colors) +
  scale_x_discrete(labels = c("U" = "U", "B" = "B")) +
  labs(x = mlab, y = "E (Kcal/mol)") +
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_blank(),
    axis.title.x = element_text(size = 16, face = "bold"),
    axis.title.y = element_text(size = 16, face = "bold"),
    axis.text = element_text(size = 14, face = "bold", color = "black"),
    legend.position = "none"
  )

ggsave(file.path(fig_dir1, paste0(filename_base, "_paired.png")), plot = p_paired, width = 4.5, height = 4.5, dpi = 600)
ggsave(file.path(fig_dir2, paste0(filename_base, "_paired.png")), plot = p_paired, width = 4.5, height = 4.5, dpi = 600)

cat("Successfully generated publication-quality plots (Lancet palette, No title, dynamic filename):", filename_base, "\n")
