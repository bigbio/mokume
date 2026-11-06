# Regenerate the limma/DEqMS design-matrix oracle:
#   Rscript limma_design_official.R limma_design_official.json
# Each case fits lmFit + contrasts.fit + eBayes on a fixed-effects design
# (cell means per condition plus covariates, as model.matrix(~0 + condition + ...)).
suppressMessages({
  library(limma)
  library(DEqMS)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript limma_design_official.R output.json")
if (getRversion() != "4.5.3") stop("The pinned reference is R 4.5.3")
if (packageVersion("limma") != "3.66.0") stop("The pinned reference is limma 3.66.0")
if (packageVersion("DEqMS") != "1.28.0") stop("The pinned reference is DEqMS 1.28.0")

# Log2 intensities rounded to 4 decimals so the JSON input is exact.
simulate <- function(seed, design, n_proteins, effects) {
  set.seed(seed)
  base <- runif(n_proteins, 18, 26)
  sd <- runif(n_proteins, 0.1, 0.6)
  coef <- matrix(0, n_proteins, ncol(design))
  for (name in names(effects)) {
    hit <- sample.int(n_proteins, effects[[name]]$n)
    coef[hit, match(name, colnames(design))] <- effects[[name]]$size * sign(rnorm(effects[[name]]$n))
  }
  noise <- matrix(rnorm(n_proteins * nrow(design)), n_proteins) * sd
  round(base + coef %*% t(design) + noise, 4)
}

run_case <- function(name, y, design, contrasts, groups, counts = NULL) {
  fit <- lmFit(y, design)
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

# Three conditions with an unbalanced two-level batch: a non-orthogonal design
# without missing values (lmFit's single-QR path, equal residual df).
condition <- factor(rep(c("A", "B", "C"), each = 4))
batch <- factor(c("b1", "b1", "b1", "b2", "b1", "b2", "b2", "b2", "b1", "b1", "b2", "b2"))
design <- model.matrix(~ 0 + condition + batch)
y <- simulate(11, design, 40, list(conditionB = list(n = 8, size = 1.5), batchb2 = list(n = 30, size = 0.8)))
counts <- sample(1:30, 40, replace = TRUE)
cm <- makeContrasts(conditionA - conditionB, conditionA - conditionC, conditionB - conditionC, levels = design)
cases[[length(cases) + 1L]] <- run_case("batch_three_conditions", y, design, cm,
  list(list(1:4, 5:8), list(1:4, 9:12), list(5:8, 9:12)), counts)

# Two conditions with a numeric covariate and missing cells: per-protein fits
# with unequal residual df (fitFDistUnequalDF1).
condition <- factor(rep(c("A", "B"), each = 6))
age <- c(34, 51, 47, 29, 62, 40, 38, 55, 44, 31, 58, 49)
design <- model.matrix(~ 0 + condition + age)
y <- simulate(23, design, 50, list(conditionB = list(n = 10, size = 1.2), age = list(n = 20, size = 0.02)))
set.seed(24)
for (i in seq_len(nrow(y))) {
  drop <- sample(c(0L, 0L, 1L, 2L), 1L)
  if (drop > 0L) y[i, sample(c(1:6, 7:12), drop)] <- NA
}
counts <- sample(1:25, 50, replace = TRUE)
cm <- makeContrasts(conditionA - conditionB, levels = design)
cases[[length(cases) + 1L]] <- run_case("numeric_covariate_missing", y, design, cm,
  list(list(1:6, 7:12)), counts)

# Paired design: four subjects measured in both conditions. One protein misses
# both samples of a subject (that subject's coefficient is not estimable) and
# one keeps as many cells as coefficients (zero residual df).
condition <- factor(rep(c("ctrl", "trt"), times = 4))
subject <- factor(rep(c("s1", "s2", "s3", "s4"), each = 2))
design <- model.matrix(~ 0 + condition + subject)
y <- simulate(37, design, 30, list(conditiontrt = list(n = 6, size = 1), subjects2 = list(n = 15, size = 0.7)))
y[3, c(7, 8)] <- NA
y[5, c(2, 4, 6)] <- NA
y[9, 1] <- NA
cm <- makeContrasts(conditiontrt - conditionctrl, levels = design)
cases[[length(cases) + 1L]] <- run_case("paired_subjects", y, design, cm,
  list(list(c(2L, 4L, 6L, 8L), c(1L, 3L, 5L, 7L))))

write_json(
  list(upstream = "limma 3.66.0, DEqMS 1.28.0, R 4.5.3", cases = cases),
  args[[1]], digits = NA, auto_unbox = TRUE, na = "null"
)
