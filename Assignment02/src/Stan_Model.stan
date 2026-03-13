//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {
  int < lower = 1 > players; // No. of players, must be at least 1
  int < lower = 1 > n; // No. of trials, must be at least 1
  array[n] int choice; // Array "h" of choices (0 or 1) with length "n" for no. trials, 
  array[n] int opp_choice; // Array "f" of feedback (0 or 1) with length "n" for no. trials
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real alpha_logit; // Define alpha_logit unbounded real no.
  real tau_logit; // Define tau_logit unbounded real no.
  
  // we use logit as we want stan to be able to sample for + to - inf!! 
  // That is why we use logit/inv_logit to transform between bounded and unbounded values e.g. log odds
}

transformed parameters { // Define how to transform paramteres back into shape the model requires.
  
  // define how to get from alpha logit to alpha and define alpha limits of parameter
  real <lower = 0, upper =1 > alpha = inv_logit(alpha_logit);
  // define how to get from tau logit to tau and define tau limits of parameter
  real < lower = 0, upper = 20 > tau = (inv_logit(tau_logit)*20); // * tau by 20 to get it bounded between 0-20
  
}

// The model to be estimated. We model the output
model {
  // Prior: Belief about alpha and tau before seeing the data
    target += normal_lpdf(alpha_logit | 0, 1.5 ); // normal distribution of alpha logit
    target += normal_lpdf(tau_logit | 0, 1.5); // normal distirbution of tau_logit
      
  // Likelihood: How the data depends on the parameters
  real PE; // define PE
  array[n] real EV;  // define Expected value as a real with bounds
  real p; // choice probability
  
  
  // !! We are modelling for the left hand (one hand) as they are implicitly linked, this makes it simple
  
  EV[1] = 0.5; // initial EV, could also be defined up by the data as something that can be changed with the data.
  
  // Loop through the trials 
  for (t in 2:n){
    
    // RL model
    PE = opp_choice[t-1] - EV[t-1];
    EV[t] = EV[t-1] + (alpha * PE);
    
    // softmax
    p = exp(tau * EV[t]) / (exp(tau * EV[t]) + exp(tau * (1-EV[t])));


    // bernoulli (0/1 choice) where choice is associated with p.
    target += bernoulli_lpmf(choice[t] | p); 
  }
}
// !! We have the data/choices, so we want to see how likely the choices are! Opposite way around to Assignment 01 !!
// How many times when you sample do you get the choice actually made.


generated quantities {
  // Create the prior for alpha and tau
  real alpha_logit_prior = normal_rng(0, 1.5);
  real tau_logit_prior   = normal_rng(0, 1.5);
  
  // --- Parameter Transformation ---
  real alpha_prior = inv_logit(alpha_logit_prior);
  real tau_prior   = inv_logit(tau_logit_prior) * 20;
  

  // --- Prior Predictive Check Simulation ---
  // Simulate data based *only* on the PRIOR distribution. - Model
  array[n] int choice_prior_rep;
  {
    real PE_prior;
    array[n] real EV_prior;
    real p_prior;
    
    // First trial choices and EV
    EV_prior[1] = 0.5;
    choice_prior_rep[1] = bernoulli_rng(0.5);
    
    for (t in 2:n){
      PE_prior      = opp_choice[t-1] - EV_prior[t-1]; // using the opponents choice data (it is contained within itself, as it is not impacted by the RL agents chocie - simply biased with noise)
      EV_prior[t]   = EV_prior[t-1] + (alpha_prior * PE_prior);
      p_prior       = exp(tau_prior * EV_prior[t]) / (exp(tau_prior * EV_prior[t]) + exp(tau_prior * (1 - EV_prior[t])));
      
      choice_prior_rep[t] = bernoulli_rng(p_prior);
    }
    
  }
  
  // Calculate a summary statistic for this PRIOR replicated dataset.
  int < lower = 0, upper = n > prior_rep_sum = sum(choice_prior_rep);
  


  // --- Posterior Predictive Check Simulation ---
  // Simulate data based on the *posterior* distribution of the parameter.
  array[n] int choice_post_rep;
  {
    real PE_post;
    array[n] real EV_post;
    real p_post;

    EV_post[1] = 0.5;
    choice_post_rep[1] = bernoulli_rng(0.5);

    for (t in 2:n) {
      PE_post      = opp_choice[t-1] - EV_post[t-1];
      EV_post[t]   = EV_post[t-1] + (alpha * PE_post);  // alpha from transformed parameters
      p_post       = exp(tau * EV_post[t]) /
                     (exp(tau * EV_post[t]) + exp(tau * (1 - EV_post[t])));
      choice_post_rep[t] = bernoulli_rng(p_post); // simulate binary choice
      }
    }
  
    // Calculate a summary statistic for this posterior dataset.
  int<lower=0, upper=n> post_rep_sum = sum(choice_post_rep);

  
  // --- Log-likelihood ---
  // Given these parameter values (alpha, tau), how probable was each choice the participant actually made?
  // For every trial - My model predicted probability p of choosing 1. The participant actually chose choice[t]. How likely was that?
  vector[n] log_lik;
  
    {
    real PE_ll;
    array[n] real EV_ll;
    real p_ll;
    
    EV_ll[1] = 0.5;
    log_lik[1] = bernoulli_lpmf(choice[1] | 0); // placeholder; t=1 has no EV update
  
  
    for (t in 2:n) {
      PE_ll      = opp_choice[t-1] - EV_ll[t-1];
      EV_ll[t]   = EV_ll[t-1] + (alpha * PE_ll);  // alpha from transformed parameters
      p_ll       = exp(tau * EV_ll[t]) /
                     (exp(tau * EV_ll[t]) + exp(tau * (1 - EV_ll[t])));
      log_lik[t] = bernoulli_lpmf(choice[t] | p_ll); 
      // per trial this computes the log probability of the observed choice under the model's predicted p
    }

  }

  // joint log prior
  real lprior = normal_lpdf(alpha_logit | 0, 1.5)
              + normal_lpdf(tau_logit | 0, 1.5);
}
