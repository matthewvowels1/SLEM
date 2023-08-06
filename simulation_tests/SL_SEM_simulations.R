

library(dplyr)
library(lavaan)

path_to_folder = 'data/'
results_dir <- "results"

N_values <- c(50, 100, 250, 500, 1000, 5000)
i_values <- 0:70  #


# simple linear model
mod <- 'Y~ X + Z'

results_df <- data.frame(N = integer(), i = integer(), coeff = numeric())

for (N in N_values) {
  for (i in i_values) {

    filename <- paste(path_to_folder, "simple_linear_confounder_N_", N, "_i_", i, ".csv", sep = "")

    if (file.exists(filename)) {
      print(filename)
      df <- read.csv(filename)

      mod = sem(mod, data=df)
      estimates = parameterEstimates(mod)
      coef <- estimates$est[estimates$lhs == "Y" & estimates$rhs == "X"]
      new_row <- data.frame(N = N, i = i, coeff = coef)

      results_df <- rbind(results_df, new_row)

    }}}
write.csv(results_df, file = paste(results_dir, "/simple_linear_confounder_sem_results.csv", sep = ""), row.names = FALSE)


# many confounders linear
mod <- 'Z4 ~ Z3
                  Z5 ~ Z2
                  Z6 ~ Z5
                  Z8 ~ Z7
                  Z10 ~ Z9
                  Z11 ~ Z10
                  X ~ Z1 + Z2 + Z3 + Z4 + Z5 + Z6 + Z7 + Z8 + Z9 + Z10 + Z11 + Z12
                  Y ~ Z1 + Z2 + Z3 + Z4 + Z5 + Z6 + Z7 + Z8 + Z9 + Z10 + Z11 + Z12 + X '

results_df <- data.frame(N = integer(), i = integer(), coeff = numeric())

for (N in N_values) {
  for (i in i_values) {

    filename <- paste(path_to_folder, "linear_many_confounders_N_", N, "_i_", i, ".csv", sep = "")

    if (file.exists(filename)) {
      print(filename)
      df <- read.csv(filename)

      mod = sem(mod, data=df)
      estimates = parameterEstimates(mod)
      coef <- estimates$est[estimates$lhs == "Y" & estimates$rhs == "X"]
      new_row <- data.frame(N = N, i = i, coeff = coef)

      results_df <- rbind(results_df, new_row)

    }}}
write.csv(results_df, file = paste(results_dir, "/linear_many_confounders_sem_results.csv", sep = ""), row.names = FALSE)



# simple non-linear
mod <- 'X ~ Z1 + Z2
        Y ~ X + Z1 + Z2'

results_df <- data.frame(N = integer(), i = integer(), coeff = numeric())

for (N in N_values) {
  for (i in i_values) {

    filename <- paste(path_to_folder, "simple_nonlinear_confounder_N_", N, "_i_", i, ".csv", sep = "")

    if (file.exists(filename)) {
      print(filename)
      df <- read.csv(filename)

      mod = sem(mod, data=df)
      estimates = parameterEstimates(mod)
      coef <- estimates$est[estimates$lhs == "Y" & estimates$rhs == "X"]
      new_row <- data.frame(N = N, i = i, coeff = coef)

      results_df <- rbind(results_df, new_row)

    }}}
write.csv(results_df, file = paste(results_dir, "/simple_nonlinear_confounder_sem_results.csv", sep = ""), row.names = FALSE)





# simple mediation
mod <- 'M ~ xm*X
        Y ~ xy*X + my*M
        direct:= xy
        indirect:=xm*my
        total := direct+indirect'

results_df <- data.frame(N = integer(), i = integer(), coeff_direct = numeric(), coeff_indirect = numeric(), coeff_total = numeric())

for (N in N_values) {
  for (i in i_values) {

    filename <- paste(path_to_folder, "simple_mediation_N_", N, "_i_", i, ".csv", sep = "")

    if (file.exists(filename)) {
      print(filename)
      df <- read.csv(filename)

      mod = sem(mod, data=df)
      estimates = parameterEstimates(mod)
      coeff_direct = estimates$est[estimates$lhs == "direct"]
      coeff_indirect = estimates$est[estimates$lhs == "indirect"]
      coeff_total = estimates$est[estimates$lhs == "total"]

      new_row <- data.frame(N = N, i = i, coeff_direct = coeff_direct, coeff_indirect = coeff_indirect, coeff_total = coeff_total)

      results_df <- rbind(results_df, new_row)

    }}}
write.csv(results_df, file = paste(results_dir, "/simple_mediation_sem_results.csv", sep = ""), row.names = FALSE)






