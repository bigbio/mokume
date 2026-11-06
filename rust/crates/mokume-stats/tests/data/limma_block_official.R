# Regenerate the limma/DEqMS random-block oracle:
#   Rscript limma_block_official.R limma_block_official.json
# Each case estimates the intra-block correlation with duplicateCorrelation,
# then fits lmFit(block =, correlation =) + contrasts.fit + eBayes on a
# fixed-effects design (cell means per condition plus covariates).
suppressMessages({
  library(limma)
  library(statmod)
  library(DEqMS)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript limma_block_official.R output.json")
if (getRversion() != "4.5.3") stop("The pinned reference is R 4.5.3")
if (packageVersion("limma") != "3.66.0") stop("The pinned reference is limma 3.66.0")
if (packageVersion("statmod") != "1.5.2") stop("The pinned reference is statmod 1.5.2")
if (packageVersion("DEqMS") != "1.28.0") stop("The pinned reference is DEqMS 1.28.0")

# Log2 intensities with a random effect per block, rounded to 4 decimals so
# the JSON input is exact.
simulate <- function(seed, design, block, n_proteins, effects, block_sd) {
  set.seed(seed)
  base <- runif(n_proteins, 18, 26)
  sd <- runif(n_proteins, 0.1, 0.6)
  coef <- matrix(0, n_proteins, ncol(design))
  for (name in names(effects)) {
    hit <- sample.int(n_proteins, effects[[name]]$n)
    coef[hit, match(name, colnames(design))] <- effects[[name]]$size * sign(rnorm(effects[[name]]$n))
  }
  levels <- unique(block)
  shift <- matrix(rnorm(n_proteins * length(levels), sd = block_sd), n_proteins)
  noise <- matrix(rnorm(n_proteins * nrow(design)), n_proteins) * sd
  round(base + coef %*% t(design) + shift[, match(block, levels)] + noise, 4)
}

drop_cells <- function(seed, y, choices) {
  set.seed(seed)
  for (i in seq_len(nrow(y))) {
    drop <- sample(choices, 1L)
    if (drop > 0L) y[i, sample.int(ncol(y), drop)] <- NA
  }
  y
}

run_case <- function(name, y, design, block, contrasts, groups, counts = NULL) {
  dc <- duplicateCorrelation(y, design, block = block)
  fit <- lmFit(y, design, block = block, correlation = dc$consensus.correlation)
  fit2 <- contrasts.fit(fit, contrasts)
  fit3 <- eBayes(fit2)
  if (!is.null(counts)) fit3$count <- counts
  results <- lapply(seq_len(ncol(contrasts)), function(j) {
    out <- list(
      weights = unname(contrasts[, j]),
      group_a = groups[[j]][[1]] - 1L,
      group_b = groups[[j]][[2]] - 1L,
      coefficient = unname(fit2$coefficients[, j]),
      stdev_unscaled = unname(fit2$stdev.unscaled[, j]),
      t = unname(fit3$t[, j]),
      p_value = unname(fit3$p.value[, j])
    )
    if (!is.null(counts)) {
      fit4 <- spectraCounteBayes(fit3, coef_col = j)
      out$sca_t <- unname(fit4$sca.t[, 1])
      out$sca_p <- unname(fit4$sca.p[, 1])
      out$sca_dfprior <- fit4$sca.dfprior
    }
    out
  })
  out <- list(
    name = name,
    matrix = unname(y),
    design = unname(design),
    block = match(block, unique(block)) - 1L,
    consensus = dc$consensus.correlation,
    atanh_correlations = unname(dc$atanh.correlations),
    sigma = unname(fit$sigma),
    df_residual = unname(fit$df.residual),
    s2_prior = fit3$s2.prior,
    df_prior = fit3$df.prior,
    df_total = unname(fit3$df.total),
    contrasts = results
  )
  if (!is.null(counts)) out$counts <- counts
  out
}

cases <- list()

# Paired design: six subjects measured in both conditions, subject as a random
# block, with missing cells (per-protein GLS fits; some proteins fall below
# duplicateCorrelation's observation gate).
condition <- factor(rep(c("ctrl", "trt"), times = 6))
subject <- rep(sprintf("s%d", 1:6), each = 2)
design <- model.matrix(~ 0 + condition)
y <- simulate(41, design, subject, 60, list(conditiontrt = list(n = 10, size = 1)), 0.8)
y <- drop_cells(42, y, c(0L, 0L, 1L, 2L, 3L))
y[7, 1:8] <- NA
counts <- sample(1:30, 60, replace = TRUE)
cm <- makeContrasts(conditiontrt - conditionctrl, levels = design)
cases[[length(cases) + 1L]] <- run_case("paired_block_missing", y, design, subject, cm,
  list(list(seq(2L, 12L, 2L), seq(1L, 11L, 2L))), counts)

# Three conditions, five donors of unequal size crossing the conditions, and a
# numeric covariate in the fixed effects.
condition <- factor(c("A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "A", "C", "B", "C"))
donor <- c("d1", "d1", "d2", "d2", "d2", "d3", "d3", "d3", "d4", "d4", "d4", "d5", "d5", "d5", "d5")
age <- c(34, 34, 51, 51, 51, 47, 47, 47, 29, 29, 29, 62, 62, 62, 62)
design <- model.matrix(~ 0 + condition + age)
y <- simulate(43, design, donor, 50,
  list(conditionB = list(n = 8, size = 1.2), age = list(n = 15, size = 0.02)), 0.6)
y <- drop_cells(44, y, c(0L, 0L, 1L, 2L))
cm <- makeContrasts(conditionA - conditionB, conditionA - conditionC, levels = design)
cases[[length(cases) + 1L]] <- run_case("unequal_blocks_covariate", y, design, donor, cm,
  list(list(which(condition == "A"), which(condition == "B")),
       list(which(condition == "A"), which(condition == "C"))))

# No missing cells: lmFit's single-transform path.
condition <- factor(rep(rep(c("A", "B"), each = 2), times = 4))
block <- rep(c("b1", "b2", "b3", "b4"), each = 4)
design <- model.matrix(~ 0 + condition)
y <- simulate(45, design, block, 40, list(conditionB = list(n = 8, size = 1)), 0.5)
counts <- sample(1:25, 40, replace = TRUE)
cm <- makeContrasts(conditionA - conditionB, levels = design)
cases[[length(cases) + 1L]] <- run_case("complete_blocks", y, design, block, cm,
  list(list(which(condition == "A"), which(condition == "B"))), counts)

# Technical replicates nested in biological replicates, which are nested in
# the conditions: the block is the biological replicate.
condition <- factor(rep(c("ctrl", "trt"), c(8, 7)))
biorep <- c("c1", "c1", "c1", "c2", "c2", "c3", "c3", "c3",
            "t1", "t1", "t2", "t2", "t2", "t3", "t3")
design <- model.matrix(~ 0 + condition)
y <- simulate(47, design, biorep, 45, list(conditiontrt = list(n = 9, size = 1.5)), 0.7)
y <- drop_cells(48, y, c(0L, 0L, 0L, 1L, 2L))
cm <- makeContrasts(conditiontrt - conditionctrl, levels = design)
cases[[length(cases) + 1L]] <- run_case("nested_technical_replicates", y, design, biorep, cm,
  list(list(9:15, 1:8)))

write_json(
  list(upstream = "limma 3.66.0, statmod 1.5.2, DEqMS 1.28.0, R 4.5.3", cases = cases),
  args[[1]], digits = NA, auto_unbox = TRUE, na = "null"
)
