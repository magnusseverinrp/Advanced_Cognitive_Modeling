// Stan model for our Reinforcement Learning model. 


// ========= Define the data that Stan will expect to receive from R. =========
data {
  int < lower = 1 > players; // No. of players, must be at least 1. Defined in the R script.
  int < lower = 1 > n; // No. of trials, must be at least 1. Defined in the R script.
  array[n] int choice; // RL agent's actual choices per trial (0 = left, 1 = right). This is Choice_RL from the R script.
  array[n] int opp_choice; // Opponent's choices per trial (0 = left, 1 = right). This is Opponent_RA from the R script.
}



// ========= Define parameters =========
// The parameters accepted by our model are 'alpha' (the learning rate i.e. how fast the agent updates its beliefs) and 'tau' (inverse temperature i.e. how deterministically the agent exploits its best guess).
parameters {
  real alpha_logit; // Define alpha as logit because this will be unbounded (-∞ to +∞), which Stan can sample from better.
  real tau_logit; // Define tau as logit because this will be unbounded (-∞ to +∞), which Stan can sample from better.
}

// Define how to transform our unbounded parameters (log odds) into bounded, interpretable parameters.
transformed parameters {
  real <lower = 0, upper =1 > alpha = inv_logit(alpha_logit); // Define the use of inv_logit to transform unbounded alpha parameter (-∞ to +∞) to bounded. The boundary defined as 0 to 1.
  real < lower = 0, upper = 20 > tau = (inv_logit(tau_logit)*20); // Define the use of inv_logit to transform unbounded tau parameter (-∞ to +∞) to bounded. The boundary is defined as 0 to 20, and since inv_logit will give 0 to 1, we multiply inv_logit by 20 to stretch the boundary to 0 to 20.
}



// ========= Model setup and initiation =========
model {
  
  // Define priors: Belief about alpha and tau before seeing the data
  // Target represents the log posterior, which is updated by the log prior and log likelihood. So 'target +=' adds the log-probability density to the overall model log-probability.
    target += normal_lpdf(alpha_logit | 0, 1.5 ); // Normal distribution of alpha_logit around 0 with SD 1.5.
    target += normal_lpdf(tau_logit | 0, 1.5); // Normal distribution of tau_logit around 0 with SD 1.5
      
  // Likelihood: How the data depends on the parameters i.e. how likely the observed choice probability p is given a value of alpha and tau.
  real PE; // Define prediction error (PE = opp_choice - estimated_value). It stores only a single number because it will be overwritten every trial.
  array[n] real EV;  // Define an array of n decimal numbers storing the expected value for every trial. We need the history of EVs because because each trial's EV is calculated from the previous trial's EV.???
  real p; // Define choice probability p. It stores only a single number because it will be overwritten every trial.
  
  // Initialise EV on trial 1 to 0.5 so the agent has no prior belief about left or right before the task starts.
  // We model only one hand (left) since left and right are implicitly linked (EV_right = 1 - EV_left)  
  EV[1] = 0.5; 
  
  // Loop through trials 2 to n
  for (t in 2:n){
    
    // RL model with Rescorla-Wagner update: we compute prediction error and update expected value
    PE = opp_choice[t-1] - EV[t-1]; // Prediction error: What did the opponent do vs. what did the RL agent expect.
    EV[t] = EV[t-1] + (alpha * PE); // Expected value: Update by incrementing the prior belief by the prediction error, weighted by the learning rate.
    
    // Apply softmax function to convert expected value to choice probability
    p = exp(tau * EV[t]) / (exp(tau * EV[t]) + exp(tau * (1-EV[t]))); // The RL agent's expected value is multiplied by tau, exponentiated, then divided by the sum of both options (right and left, where left is just 1 - EV), which normalises it into a probability between 0 and 1.

    // We have the data/choices, so we want to see how likely the choices are. Opposite way around to Assignment 01!
    // So here we are asking how likely our observed binary choice is given the predicted probability p.
    // This tells Stan which alpha and tau values best explain the observed data.
    target += bernoulli_lpmf(choice[t] | p); 
  }
}



// ========= Generated quantities =========
// This is a diagnostic tool for model assessment. It executes after the model has been fitted and does not affect sampling.
// Within generated quantities, we will do 1) prior predictive check, 2) posterior predictive check, 3) log likelihood, 4) joint log prior

generated quantities {
  
  // Create the prior for alpha and tau
  real alpha_logit_prior = normal_rng(0, 1.5);
  real tau_logit_prior   = normal_rng(0, 1.5);
  
  // Parameter Transformation
  real alpha_prior = inv_logit(alpha_logit_prior);
  real tau_prior   = inv_logit(tau_logit_prior) * 20;



  // --------- Prior Predictive Check Simulation --------- 
  // Simulate data based *only* on the PRIOR distribution. 
  // Purpose: check whether our model priors generate plausible choices before seeing any data.
  // The structure largely follows the model structure above, except 1) we use alpha_prior and tau_prior instead of the posterior alpha and tau, and 2) we use bernoulli_rng to generate choices instead of bernoulli_lpmf to score them.
 
  array[n] int choice_prior_rep; // Array of n simulated binary choices (0 or 1) generated from the prior.
  {
    real PE_prior; // Prediction error
    array[n] real EV_prior; // Expected value
    real p_prior; // Choice probability
    
    // First trial choice
    EV_prior[1] = 0.5; // Start with no prior belief about left or right.
    choice_prior_rep[1] = bernoulli_rng(0.5); // Randomly simulate first trial choice with equal probability.
    
    // Loop through trials 2 to n
    for (t in 2:n){
      PE_prior      = opp_choice[t-1] - EV_prior[t-1]; // Prediction error: OBS we are using the opponents choice data (it is contained within itself, as it is not impacted by the RL agents choice - simply biased with noise)
      EV_prior[t]   = EV_prior[t-1] + (alpha_prior * PE_prior); // Expected value
      p_prior       = exp(tau_prior * EV_prior[t]) / (exp(tau_prior * EV_prior[t]) + exp(tau_prior * (1 - EV_prior[t]))); // Softmax to get choice probability
      choice_prior_rep[t] = bernoulli_rng(p_prior); // Simulate a binary choice (0 or 1) based on predicted probability p_prior.
    }
  }

  // Calculate a summary statistic for this simulated prior dataset. This is so we can plot the prior predictive check to compare the simulated vs observed choices.
  int < lower = 0, upper = n > prior_rep_sum = sum(choice_prior_rep);
  


  // --------- Posterior Predictive Check Simulation ---------
  // Simulate data based on the *posterior* distribution of the parameter.
  // Purpose: check whether our model generates plausible choices after seeing the data.
  // The structure largely follows the model structure above, except we use alpha and tau instead of the posterior alpha_prior and tau_prior, as these are posteriors. However, we still use bernoulli_rng to generate choices instead of bernoulli_lpmf to score them.
 
  array[n] int choice_post_rep; // Array of n simulated binary choices (0 or 1) generated from the posterior.
  {
    real PE_post; // Prediction error
    array[n] real EV_post; // Expected value
    real p_post; // Choice probability

    // First trial choice
    EV_post[1] = 0.5; // Start with no prior belief about left or right.
    choice_post_rep[1] = bernoulli_rng(0.5); // Randomly simulate first trial choice with equal probability.

    // Loop through trials 2 to n
    for (t in 2:n) {
      PE_post      = opp_choice[t-1] - EV_post[t-1]; // Prediction error
      EV_post[t]   = EV_post[t-1] + (alpha * PE_post);  // Expected value. We are using posterior alpha from transformed parameters.
      p_post       = exp(tau * EV_post[t]) / (exp(tau * EV_post[t]) + exp(tau * (1 - EV_post[t]))); // Softmax to get choice probability.
      choice_post_rep[t] = bernoulli_rng(p_post); // Simulate a binary choice (0 or 1) based on predicted probability p_post.
      }
    }
  
  // Calculate a summary statistic for this simulated posterior dataset. This is so we can plot the posterior predictive check to compare the simulated vs observed choices.
  int<lower=0, upper=n> post_rep_sum = sum(choice_post_rep);


  
  // --------- Log-likelihood ---------
  // We compare simulated choices (generated by the model using posterior alpha and tau) against observed choices (what the agent actually did).
  // Purpose: We want to know how likely each real observed choice is compared to the model's predicted probability. 
  // So we ask e.g. my model predicted probability p of choosing 1. The participant actually chose choice[t]. How likely was that?
  // The structure largely follows the model structure above, except we use bernoulli_lpmf to score the choices instead of bernoulli_rng to generate them.

  vector[n] log_lik; // Vector of n log likelihoods, one per trial. Must be a vector (not array) so R's loo() function can read it for model comparison.
    {
    real PE_ll; // Prediction error 
    array[n] real EV_ll; // Expected value
    real p_ll; //  // Choice probability
    
    // First trial choice
    EV_ll[1] = 0.5; // Start with no prior belief about left or right.
    log_lik[1] = bernoulli_lpmf(choice[1] | 0); // Placeholder for trial 1 — there is no EV update yet so we cannot compute a meaningful log likelihood.
  
    // Loop through trials 2 to n
    for (t in 2:n) {
      PE_ll      = opp_choice[t-1] - EV_ll[t-1]; // Prediction error
      EV_ll[t]   = EV_ll[t-1] + (alpha * PE_ll);  // Expected value. We are using posterior alpha from transformed parameters.
      p_ll       = exp(tau * EV_ll[t]) / (exp(tau * EV_ll[t]) + exp(tau * (1 - EV_ll[t]))); // Softmax to get choice probability.
      log_lik[t] = bernoulli_lpmf(choice[t] | p_ll); // Per trial this computes the log probability of the observed choice under the model's predicted p_ll.
    }
  }

  // Joint log prior. Here we compute the combined log probability of the posterior alpha_logit and tau_logit under the Normal(0, 1.5) prior.
  real lprior = normal_lpdf(alpha_logit | 0, 1.5) + normal_lpdf(tau_logit | 0, 1.5); //
}
