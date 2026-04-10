// Simple Bayesian Agent (SBA).
// No free parameters — evidence is counted at face value.
// Jeffreys prior pseudo-counts: alpha0 = beta0 = 0.5.
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

model {
  // Vectorized likelihood with fixed weights = 1
  vector[N] alpha_post = 0.5 + to_vector(rating_1) + to_vector(rating_g);
  vector[N] beta_post  = 0.5 + (to_vector(total_1) - rating_1)
                             + (to_vector(total_g)  - to_vector(rating_g));
                             
  target += beta_binomial_lpmf(choice | 1, alpha_post, beta_post);
}

generated quantities {
  vector[N] log_lik;
  array[N] int posterior_pred;

  for (i in 1:N) {
    real alpha_post = 0.5 + rating_1[i] + rating_g[i];
    real beta_post  = 0.5 + (total_1[i] - rating_1[i]) + (total_g[i] - rating_g[i]);

    log_lik[i]        = beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
    posterior_pred[i] = beta_binomial_rng(1, alpha_post, beta_post);
  }
}
