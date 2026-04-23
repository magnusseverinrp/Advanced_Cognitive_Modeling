// Weighted Bayesian Agent (reparameterised).
// rho in (0,1): relative weight of direct vs social evidence.
// kappa > 0: total evidence scaling (w_d + w_s).
data {
  int<lower=1> N;
  array[N] int<lower=1, upper=8> SecondRating;
  array[N] int<lower=1, upper=8> FirstRating;
  array[N] int<lower=1, upper=8> GroupRating;
  array[N] int<lower=0> total;
}

parameters {
  real<lower=0, upper=1> rho;    // relative weight: w_d / (w_d + w_s)
  real<lower=0>          kappa;  // total weight: w_d + w_s
}

transformed parameters {
  real<lower=0> weight_direct = rho * kappa;
  real<lower=0> weight_social = (1.0 - rho) * kappa;
}

model {
  // rho: weakly centred on equal weighting
  target += beta_lpdf(rho | 2, 2);
  // kappa: lognormal centered on 2 (SBA equivalent)
  target += lognormal_lpdf(kappa | log(2), 0.5);

  // Vectorized likelihood
  vector[N] alpha_post = 0.5 + weight_direct * to_vector(FirstRating)
                             + weight_social * to_vector(GroupRating);
  vector[N] beta_post  = 0.5 + weight_direct * (to_vector(total) - to_vector(FirstRating))
                             + weight_social * (to_vector(total) - to_vector(GroupRating));
                             
  target += beta_binomial_lpmf(SecondRating | 8, alpha_post, beta_post);
}

generated quantities {
  vector[N] log_lik;
  array[N] int prior_pred;
  array[N] int posterior_pred;
  real lprior = beta_lpdf(rho | 2, 2) + lognormal_lpdf(kappa | log(2), 0.5);

  real rho_prior   = beta_rng(2, 2);
  real kappa_prior = lognormal_rng(log(2), 0.5);
  real wd_prior    = rho_prior * kappa_prior;
  real ws_prior    = (1.0 - rho_prior) * kappa_prior;

  for (i in 1:N) {
    real alpha_post = 0.5 + weight_direct * FirstRating[i] + weight_social * GroupRating[i];
    real beta_post  = 0.5 + weight_direct * (total[i] - FirstRating[i]) 
                         + weight_social * (total[i] - GroupRating[i]);

    log_lik[i]        = beta_binomial_lpmf(SecondRating[i] | 8, alpha_post, beta_post);
    posterior_pred[i] = beta_binomial_rng(8, alpha_post, beta_post);

    real ap = 0.5 + wd_prior * FirstRating[i] + ws_prior * GroupRating[i];
    real bp = 0.5 + wd_prior * (total[i] - FirstRating[i]) 
                  + ws_prior * (total[i] - GroupRating[i]);
    prior_pred[i] = beta_binomial_rng(8, ap, bp);
  }
}
