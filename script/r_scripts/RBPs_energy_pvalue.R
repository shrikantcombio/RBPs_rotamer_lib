#!/usr/bin/Rscript
# ==============================================================================
# RBPs_energy_pvalue.R
# ------------------------------------------------------------------------------
# Refactored R script for publication-quality total energy distributions (Bound vs Unbound).
# Reads directly from JSON summary files (preserving exact PDB IDs and string formats).
#
# Generates:
#   - Boxplot with independent t-test p-value annotation
#   - Paired line plot with paired two-tailed t-test p-value annotation
#
# Output filenames dynamically constructed based on region/subset arguments:
#   e.g., total_energy_pdb_int_boxplot.png, total_energy_aa_int_paired.png, etc.
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
    mlab <- "Aromatic Residues"
  } else {
    subset_tag <- "aa"
    mlab <- "Amino Acid"
  }
  
  if (grepl("NINT", region_mode, ignore.case = TRUE)) {
    u_vals <- df_raw$NU_DELG
    b_vals <- df_raw$NB_DELG
    res_ids <- df_raw$AA
    region_tag <- "nint"
    mlab <- paste(mlab, "Non-interface")
  } else {
    u_vals <- df_raw$IU_DELG
    b_vals <- df_raw$IB_DELG
    res_ids <- df_raw$AA
    region_tag <- "int"
    mlab <- paste(mlab, "Interface")
  }
} else {
  subset_tag <- "pdb"
  res_ids <- df_raw$PDB_ID
  if (grepl("NINT", region_mode, ignore.case = TRUE)) {
    u_vals <- df_raw$NU_DELG
    b_vals <- df_raw$NB_DELG
    region_tag <- "nint"
    mlab <- "PDB Non-interface"
  } else {
    u_vals <- df_raw$IU_DELG
    b_vals <- df_raw$IB_DELG
    region_tag <- "int"
    mlab <- "PDB Interface"
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

# Compute two-tailed t-tests
tt_ind <- t.test(b_vec, u_vec, paired = FALSE)
tt_pair <- t.test(b_vec, u_vec, paired = TRUE)

pval_ind_str <- paste0("t-test, p = ", format.pval(tt_ind$p.value, digits = 3))
pval_pair_str <- paste0("paired t-test, p = ", format.pval(tt_pair$p.value, digits = 3))

max_y <- max(mydata$Avg.E, na.rm = TRUE)
min_y <- min(mydata$Avg.E, na.rm = TRUE)
y_top <- max_y + (max_y - min_y) * 0.12

# Construct output filename stem: total_energy_<subset>_<region>
filename_base <- paste0("total_energy_", subset_tag, "_", region_tag)

# Color scheme matching sample publication plots
state_colors <- c("B" = "#004225", "U" = "#B22222")

# 1. Boxplot with independent t-test p-value (No title)
p_box <- ggplot(mydata, aes(x = Form, y = Avg.E, fill = Form, color = Form)) +
  geom_boxplot(alpha = 0.6, outlier.shape = NA, width = 0.4, linewidth = 0.6) +
  geom_jitter(width = 0.18, alpha = 0.7, size = 2) +
  annotate("text", x = 1.5, y = y_top, label = pval_ind_str, size = 4.8, fontface = "bold") +
  scale_fill_manual(values = state_colors) +
  scale_color_manual(values = state_colors) +
  scale_x_discrete(labels = c("U" = "Unbound (U)", "B" = "Bound (B)")) +
  labs(x = mlab, y = "Total Torsional Energy (kcal/mol)") +
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_blank(),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12, face = "bold", color = "black"),
    legend.position = "none"
  )

ggsave(file.path(fig_dir1, paste0(filename_base, "_boxplot.png")), plot = p_box, width = 5.5, height = 5.5, dpi = 600)
ggsave(file.path(fig_dir2, paste0(filename_base, "_boxplot.png")), plot = p_box, width = 5.5, height = 5.5, dpi = 600)

# 2. Paired Line Plot with paired t-test p-value (No title)
p_paired <- ggplot(mydata, aes(x = Form, y = Avg.E, group = Res, color = Form)) +
  geom_line(color = "gray50", alpha = 0.6, linewidth = 0.5) +
  geom_point(size = 2.2, alpha = 0.85) +
  annotate("text", x = 1.5, y = y_top, label = pval_pair_str, size = 4.8, fontface = "bold") +
  scale_color_manual(values = state_colors) +
  scale_x_discrete(labels = c("U" = "Unbound (U)", "B" = "Bound (B)")) +
  labs(x = mlab, y = "Total Torsional Energy (kcal/mol)") +
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_blank(),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12, face = "bold", color = "black"),
    legend.position = "none"
  )

ggsave(file.path(fig_dir1, paste0(filename_base, "_paired.png")), plot = p_paired, width = 5.5, height = 5.5, dpi = 600)
ggsave(file.path(fig_dir2, paste0(filename_base, "_paired.png")), plot = p_paired, width = 5.5, height = 5.5, dpi = 600)

cat("Successfully generated publication-quality plots (No title, dynamic filename):", filename_base, "\n")
