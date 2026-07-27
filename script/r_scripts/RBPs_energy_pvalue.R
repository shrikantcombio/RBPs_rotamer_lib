#!/usr/bin/Rscript
# ==============================================================================
# RBPs_energy_pvalue.R
# ------------------------------------------------------------------------------
# R script to compute statistical significance (independent & paired t-test)
# and generate publication-quality boxplots and paired line plots for protein
# side-chain energy distributions (Bound vs Unbound states).
#
# Plots are saved under figure/energy_pvalues/ and figures/energy_pvalues/
#
# Usage:
#   Rscript script/r_scripts/RBPs_energy_pvalue.R <input_dat_file> [label]
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(tools)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  input_file <- "input_files/RBPs_BU_INT_187.dat"
} else {
  input_file <- args[1]
}

if (!file.exists(input_file)) {
  cat("Error: Input file not found:", input_file, "\n")
  q(status = 1)
}

# Determine region label
if (length(args) >= 2) {
  mlab <- args[2]
} else {
  if (grepl("NINT", input_file, ignore.case = TRUE)) {
    mlab <- "Non-interface"
  } else {
    mlab <- "Interface"
  }
}

cat("Processing input data file:", input_file, "(Region:", mlab, ")\n")

mydata <- read.table(input_file, sep = "\t", header = TRUE)

filename_base <- file_path_sans_ext(basename(input_file))

# Setup figure output directories
fig_dir1 <- "figure/energy_pvalues"
fig_dir2 <- "figures/energy_pvalues"

dir.create(fig_dir1, showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir2, showWarnings = FALSE, recursive = TRUE)

# Calculate t-test p-values
b_vals <- mydata$Avg.E[mydata$Form == "B"]
u_vals <- mydata$Avg.E[mydata$Form == "U"]

tt_ind <- t.test(b_vals, u_vals, paired = FALSE)
tt_pair <- t.test(b_vals, u_vals, paired = TRUE)

pval_ind_str <- paste0("t-test, p = ", format.pval(tt_ind$p.value, digits = 3))
pval_pair_str <- paste0("paired t-test, p = ", format.pval(tt_pair$p.value, digits = 3))

max_y <- max(mydata$Avg.E, na.rm = TRUE)
min_y <- min(mydata$Avg.E, na.rm = TRUE)
y_top <- max_y + (max_y - min_y) * 0.12

# 1. Boxplot + Jitter with independent t-test p-value
p_box <- ggplot(mydata, aes(x = Form, y = Avg.E, fill = Form, color = Form)) +
  geom_boxplot(alpha = 0.6, outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.7, size = 1.8) +
  annotate("text", x = 1.5, y = y_top, label = pval_ind_str, size = 5, fontface = "bold") +
  scale_fill_manual(values = c("B" = "#004225", "U" = "#B22222")) +
  scale_color_manual(values = c("B" = "#004225", "U" = "#B22222")) +
  labs(
    title = paste("Energy Distribution (", mlab, ")", sep = ""),
    x = mlab,
    y = "E (Kcal/mol)"
  ) +
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_text(size = 14, color = "red", face = "bold.italic", hjust = 0.5),
    axis.title = element_text(size = 16, face = "bold"),
    legend.position = "none"
  )

out_box1 <- file.path(fig_dir1, paste0(filename_base, "_boxplot.png"))
out_box2 <- file.path(fig_dir2, paste0(filename_base, "_boxplot.png"))
ggsave(out_box1, plot = p_box, width = 5, height = 5, dpi = 600)
ggsave(out_box2, plot = p_box, width = 5, height = 5, dpi = 600)

# 2. Paired Plot with paired t-test p-value
p_paired <- ggplot(mydata, aes(x = Form, y = Avg.E, group = Res, color = Form)) +
  geom_line(color = "gray60", alpha = 0.6, linewidth = 0.5) +
  geom_point(size = 2, alpha = 0.8) +
  annotate("text", x = 1.5, y = y_top, label = pval_pair_str, size = 5, fontface = "bold") +
  scale_color_manual(values = c("B" = "#004225", "U" = "#B22222")) +
  labs(
    title = paste("Paired Energy Comparison (", mlab, ")", sep = ""),
    x = mlab,
    y = "E (Kcal/mol)"
  ) +
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_text(size = 14, color = "red", face = "bold.italic", hjust = 0.5),
    axis.title = element_text(size = 16, face = "bold"),
    legend.position = "none"
  )

out_paired1 <- file.path(fig_dir1, paste0(filename_base, "_paired.png"))
out_paired2 <- file.path(fig_dir2, paste0(filename_base, "_paired.png"))
ggsave(out_paired1, plot = p_paired, width = 5, height = 5, dpi = 600)
ggsave(out_paired2, plot = p_paired, width = 5, height = 5, dpi = 600)

cat("Successfully saved energy plots to:\n  -", out_box1, "\n  -", out_paired1, "\n")
