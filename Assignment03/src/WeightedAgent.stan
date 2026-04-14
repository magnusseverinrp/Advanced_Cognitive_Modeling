// Weighted Bayesian Agent (raw parameterisation).
// w_d, w_s > 0: independent weights on direct and social evidence.
// Jeffreys prior pseudo-counts (0.5) consistent with SBA.
data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> choice;
  array[N] int<lower=0, upper=8> rating_1;
  array[N] int<lower=0, upper=8> rating_g;
  array[N] int<lower=0> total_1;
  array[N] int<lower=0> total_g;
  array[N] int<lower=-3, upper=3> feedback;
  array[N] int<lower=-7, upper=7> change;
}

parameters {
  real<lower=0> weight_direct;  // w_d
  real<lower=0> weight_social;  // w_s
}

model {
  // Priors: lognormal centered on 1 (evidence taken at face value)
  target += lognormal_lpdf(weight_direct | 0, 0.5);
  target += lognormal_lpdf(weight_social | 0, 0.5);

  // Vectorized likelihood
  vector[N] alpha_post = 0.5 + weight_direct * to_vector(rating_1)
                             + weight_social * to_vector(rating_g);
  vector[N] beta_post  = 0.5 + weight_direct * (to_vector(total_1) - to_vector(rating_1))
                             + weight_social * (to_vector(total_g) - to_vector(rating_g));
                             
  target += beta_binomial_lpmf(choice | 1, alpha_post, beta_post);
}

generated quantities {
  vector[N] log_lik;
  array[N] int posterior_pred;

  for (i in 1:N) {
    real alpha_post = 0.5 + weight_direct * rating_1[i] + weight_social * rating_g[i];
    real beta_post  = 0.5 + weight_direct * (total_1[i] - rating_1[i]) 
                         + weight_social * (total_g[i] - rating_g[i]);

    log_lik[i]        = beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
    posterior_pred[i] = beta_binomial_rng(1, alpha_post, beta_post);
  }
}
