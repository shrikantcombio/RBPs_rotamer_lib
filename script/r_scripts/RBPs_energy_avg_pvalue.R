#!/usr/bin/Rscript
# ==============================================================================
# RBPs_energy_avg_pvalue.R
# ------------------------------------------------------------------------------
# R script to compute statistical significance (independent & paired t-test)
# and generate publication-quality boxplots and paired line plots for average energy
# distributions (Bound vs Unbound states) from JSON summary files.
#
# Usage:
#   Rscript script/r_scripts/RBPs_energy_avg_pvalue.R [json_input_file] [region: INT|NINT|AA_INT|AA_NINT|AROMATIC_INT|AROMATIC_NINT]
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(jsonlite)
  library(tools)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  json_file <- "output_files/torsion_potential_energy_summary_187.json"
} else {
  json_file <- args[1]
}

if (!file.exists(json_file)) {
  cat("Error: JSON input file not found:", json_file, "\n")
  q(status = 1)
}

region_mode <- if (length(args) >= 2) toupper(args[2]) else "INT"

cat("Processing JSON data file:", json_file, "(Region Mode:", region_mode, ")\n")

# Load JSON directly to preserve PDB IDs and string formats
data_list <- jsonlite::fromJSON(json_file)
df_raw <- as.data.frame(data_list)

fig_dir1 <- "figure/energy_pvalues"
fig_dir2 <- "figures/energy_pvalues"
dir.create(fig_dir1, showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir2, showWarnings = FALSE, recursive = TRUE)

if (grepl("amino_acid", json_file, ignore.case = TRUE) || grepl("AA", region_mode, ignore.case = TRUE)) {
  # Amino Acid Level
  if (grepl("AROMATIC", region_mode, ignore.case = TRUE)) {
    df_raw <- df_raw[df_raw$AA %in% c("HIS", "PHE", "TRP", "TYR"), ]
    mlab <- "Aromatic Amino Acid"
  } else {
    mlab <- "Amino Acid"
  }
  
  if (grepl("NINT", region_mode, ignore.case = TRUE)) {
    u_vals <- df_raw$NU_AVG_DELG
    b_vals <- df_raw$NB_AVG_DELG
    res_ids <- df_raw$AA
    mlab <- paste(mlab, "Non-interface (Avg E)")
  } else {
    u_vals <- df_raw$IU_AVG_DELG
    b_vals <- df_raw$IB_AVG_DELG
    res_ids <- df_raw$AA
    mlab <- paste(mlab, "Interface (Avg E)")
  }
} else {
  # PDB Level
  res_ids <- df_raw$PDB_ID
  if (grepl("NINT", region_mode, ignore.case = TRUE)) {
    u_vals <- df_raw$NU_AVG_DELG
    b_vals <- df_raw$NB_AVG_DELG
    mlab <- "PDB Non-interface (Avg E)"
  } else {
    u_vals <- df_raw$IU_AVG_DELG
    b_vals <- df_raw$IB_AVG_DELG
    mlab <- "PDB Interface (Avg E)"
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

filename_base <- paste0("avg_energy_", tolower(gsub("[^A-Za-z0-9]", "_", region_mode)))

# 1. Boxplot with independent t-test p-value
p_box <- ggplot(mydata, aes(x = Form, y = Avg.E, fill = Form, color = Form)) +
  geom_boxplot(alpha = 0.6, outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.7, size = 1.8) +
  annotate("text", x = 1.5, y = y_top, label = pval_ind_str, size = 4.5, fontface = "bold") +
  scale_fill_manual(values = c("B" = "#004225", "U" = "#B22222")) +
  scale_color_manual(values = c("B" = "#004225", "U" = "#B22222")) +
  labs(title = paste("Average Energy Distribution (", mlab, ")", sep = ""), x = "State", y = "E (kcal/mol)") +
  theme_bw(base_size = 14) +
  theme(plot.title = element_text(size = 13, color = "red", face = "bold.italic", hjust = 0.5), axis.title = element_text(size = 14, face = "bold"), legend.position = "none")

ggsave(file.path(fig_dir1, paste0(filename_base, "_avg_boxplot.png")), plot = p_box, width = 5.5, height = 5.5, dpi = 600)
ggsave(file.path(fig_dir2, paste0(filename_base, "_avg_boxplot.png")), plot = p_box, width = 5.5, height = 5.5, dpi = 600)

# 2. Paired Plot with paired t-test p-value
p_paired <- ggplot(mydata, aes(x = Form, y = Avg.E, group = Res, color = Form)) +
  geom_line(color = "gray60", alpha = 0.6, linewidth = 0.5) +
  geom_point(size = 2, alpha = 0.8) +
  annotate("text", x = 1.5, y = y_top, label = pval_pair_str, size = 4.5, fontface = "bold") +
  scale_color_manual(values = c("B" = "#004225", "U" = "#B22222")) +
  labs(title = paste("Paired Average Energy Comparison (", mlab, ")", sep = ""), x = "State", y = "E (kcal/mol)") +
  theme_bw(base_size = 14) +
  theme(plot.title = element_text(size = 13, color = "red", face = "bold.italic", hjust = 0.5), axis.title = element_text(size = 14, face = "bold"), legend.position = "none")

ggsave(file.path(fig_dir1, paste0(filename_base, "_avg_paired.png")), plot = p_paired, width = 5.5, height = 5.5, dpi = 600)
ggsave(file.path(fig_dir2, paste0(filename_base, "_avg_paired.png")), plot = p_paired, width = 5.5, height = 5.5, dpi = 600)

cat("Successfully computed t-tests and saved average energy plots for region:", mlab, "\n")
