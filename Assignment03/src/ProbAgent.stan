// Proportional Bayesian Agent (PBA).
// p in [0,1] allocates the unit evidence budget between direct and social.
// p = 0.5 approximates balanced weighting; p -> 1 ignores social; p -> 0 ignores direct.
data {
  int<lower=1> N;
  array[N] int<lower=1, upper=8> SecondRating; // noget kunne være er off her ift. 0-7 eller 1-8
  array[N] int<lower=1, upper=8> FirstRating;
  array[N] int<lower=1, upper=8> GroupRating;
  array[N] int<lower=0> total_1;
  array[N] int<lower=0> total_g;
  // array[N] int<lower=0, upper=1> theta;
  // array[N] int<lower=0> N_PSEUDO;
}

parameters {
  real<lower=0, upper=1> w;  // Allocation to direct evidence
}

model {
  vector[N] alpha_post;
  vector[N] beta_post;
  
  w ~ beta(2, 2);

  // Vectorized likelihood
  alpha_post = 0.5 + w * to_vector(FirstRating) + (1.0 - w) * to_vector(GroupRating);
  beta_post  = 0.5 + w * (to_vector(total_1) - to_vector(FirstRating)) 
          + (1.0 - w) * (to_vector(total_g) - to_vector(GroupRating));
                             
  target += beta_binomial_lpmf(SecondRating | 8, alpha_post, beta_post);
  // theta ~ beta_binomial(N_PSEUDO, alpha_post, beta_post);
}


generated quantities {
  vector[N] log_lik;
  array[N] int prior_pred;
  array[N] int posterior_pred;
  real lprior = beta_lpdf(w | 2, 2);
  real w_prior = beta_rng(2, 2);

  for (i in 1:N) {
    real alpha_post = 0.5 + w * FirstRating[i] + (1.0 - w) * GroupRating[i];
    real beta_post  = 0.5 + w * (total_1[i] - FirstRating[i])
                          + (1.0 - w) * (total_g[i] - GroupRating[i]);

    log_lik[i]        = beta_binomial_lpmf(SecondRating[i] | 8, alpha_post, beta_post);
    posterior_pred[i] = beta_binomial_rng(8, alpha_post, beta_post);

    real ap = 0.5 + w_prior * FirstRating[i] + (1.0 - w_prior) * GroupRating[i];
    real bp = 0.5 + w_prior * (total_1[i] - FirstRating[i])
                  + (1.0 - w_prior) * (total_g[i] - GroupRating[i]);
    prior_pred[i] = beta_binomial_rng(1, ap, bp);
  }
}
