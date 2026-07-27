#!/usr/bin/Rscript
# ==============================================================================
# RBPs_rotamers.R
# ------------------------------------------------------------------------------
# Refactored R script to process side-chain rotamer distribution summaries
# (T, p, P, t, M, m rotamer states) across Bound ('B') and Unbound ('U') states,
# and generate publication-quality faceted bar charts for all amino acids.
#
# Plots are saved automatically under figure/rotamer_distributions/ and figures/rotamer_distributions/
#
# Usage:
#   Rscript script/r_scripts/RBPs_rotamers.R [input_csv] [sasa_filter]
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(tools)
})

args <- commandArgs(trailingOnly = TRUE)

input_file <- ifelse(length(args) >= 1, args[1], "output_files/rotamer_distribution/rotamer_distribution_summary.csv")
sasa_filter <- ifelse(length(args) >= 2, args[2], "Interface")

if (!file.exists(input_file)) {
  cat("Error: Input summary CSV file not found:", input_file, "\n")
  q(status = 1)
}

cat("Loading rotamer distribution data from:", input_file, "\n")

# Load CSV data
df <- read.csv(input_file, stringsAsFactors = FALSE)

# Filter by SASA if SASA column exists
if ("SASA" %in% colnames(df)) {
  if (any(tolower(df$SASA) == tolower(sasa_filter))) {
    df <- df %>% filter(tolower(SASA) == tolower(sasa_filter))
  }
}

# Filter Chi1 level if Chi_Level column exists
if ("Chi_Level" %in% colnames(df)) {
  df <- df %>% filter(Chi_Level == "Chi1")
}

# Rename columns if needed
if ("forms" %in% colnames(df) && !"BoundState" %in% colnames(df)) {
  df <- df %>% rename(BoundState = forms)
}

# Map state codes 'B' -> 'Bound', 'U' -> 'Unbound'
df$BoundState <- ifelse(df$BoundState == "B", "Bound", ifelse(df$BoundState == "U", "Unbound", df$BoundState))
df$BoundState <- factor(df$BoundState, levels = c("Bound", "Unbound"))

rotamer_cols <- intersect(c("T", "p", "P", "t", "M", "m"), colnames(df))

if (length(rotamer_cols) == 0) {
  cat("Error: No recognized rotamer columns (T, p, P, t, M, m) found in input data.\n")
  q(status = 1)
}

# Pivot rotamer columns to long format
df_long <- df %>%
  pivot_longer(cols = all_of(rotamer_cols), names_to = "RotamerState", values_to = "Percentage") %>%
  mutate(
    Amino_acid = as.factor(Amino_acid),
    RotamerState = factor(RotamerState, levels = rotamer_cols)
  )

# Define custom color palette for rotamer states
rotamer_colors <- c(
  "p" = "#1f77b4",  # g+ (p)
  "t" = "#2ca02c",  # t (t)
  "m" = "#d62728",  # g- (m)
  "T" = "#ff7f0e",  # gg (T)
  "P" = "#9467bd",  # gt (P)
  "M" = "#8c564b"   # tg (M)
)

# Output figure directories
fig_dir1 <- "figure/rotamer_distributions"
fig_dir2 <- "figures/rotamer_distributions"

dir.create(fig_dir1, showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir2, showWarnings = FALSE, recursive = TRUE)

# Create publication-quality faceted bar plot
p <- ggplot(df_long, aes(x = BoundState, y = Percentage, fill = RotamerState)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.9, color = "black", linewidth = 0.2) +
  facet_wrap(~ Amino_acid, scales = "free_y", ncol = 4) +
  scale_fill_manual(values = rotamer_colors) +
  labs(
    title = paste("Chi1 Rotamer Distribution across Amino Acids (", sasa_filter, ")", sep = ""),
    x = "Conformational State",
    y = "Rotamer Frequency (%)",
    fill = "Rotamer State"
  ) +
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    strip.text = element_text(size = 12, face = "bold"),
    axis.text.x = element_text(size = 11, face = "bold"),
    axis.title = element_text(size = 13, face = "bold"),
    legend.position = "right",
    legend.title = element_text(face = "bold")
  )

filename_base <- paste0("rotamer_distribution_", tolower(sasa_filter))

out_png1 <- file.path(fig_dir1, paste0(filename_base, ".png"))
out_png2 <- file.path(fig_dir2, paste0(filename_base, ".png"))
out_pdf1 <- file.path(fig_dir1, paste0(filename_base, ".pdf"))

ggsave(out_png1, plot = p, width = 12, height = 9, dpi = 300)
ggsave(out_png2, plot = p, width = 12, height = 9, dpi = 300)
ggsave(out_pdf1, plot = p, width = 12, height = 9)

cat("Successfully generated and saved rotamer distribution plots to:\n  -", out_png1, "\n  -", out_png2, "\n")
