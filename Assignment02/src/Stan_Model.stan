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
  int < LOWER = 1 > players; # No. of players, must be at least 1
  int < LOWER = 1 > n; # No. of trials, must be at least 1
  array[n] int choice; # Array "h" of choices (0 or 1) with length "n" for no. trials, 
  array[n] int opp_choice; # Array "f" of feedback (0 or 1) with length "n" for no. trials
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real alpha_logit; # Define alpha_logit unbounded real no.
  real tau_logit; # Define tau_logit unbounded real no.
  
  # we use logit as we want stan to be able to sample for + to - inf!! 
  #That is why we use logit/inv_logit to transform between bounded and unbounded values e.g. log odds
}

transformed parameters { # Define how to transform paramteres back into shape the model requires.
  
  # define how to get from alpha logit to alpha and define alpha limits of parameter
  real <LOWER = 0, UPPER =1 > alpha = inv_logt(alpha_logit);
  # define how to get from tau logit to tau and define tau limits of parameter
  real < LOWER = 0, UPPER = 20 > tau = (inv_logt(tau_logit)*20); # * tau by 20 to get it bounded between 0-20
  
}

// The model to be estimated. We model the output
model {
  // Prior: Belief about alpha and tau before seeing the data
    target += normal_lpdf(alpha_logit | 0, 1.5 ); # normal distribution of alpha logit
    target += normal_lpdf(tau_logit | 0, 5); # normal distirbution of tau_logit
      
  // Likelihood: How the data depends on the parameters
  real PE; # define PE
  real < LOWER = 0, UPPER = 1 > EV; # define Expected value as a real with bounds
  real < LOWER = 0, UPPER = 1 > p; # choice probability
  
  
  # !! We are modelling for the left hand (one hand) as they are implicitly linked, this makes it simple
  
  EV[1] = 0.5; # initial EV, could also be defined up by the data as something that can be changed with the data.
  
  # Loop through the trials 
  for (t in 2:n){
    
    # RL model
    PE = opp_choice[t-1] - EV[t-1];
    EV[t] = EV[t-1] + (alpha * PE);
    
    # softmax
    p = exp(tau * EV[t]) / sum(exp(tau * EV[t]), exp(tau * (1-EV[t])));

    # bernoulli (0/1 choice) where choice is associated with p.
    target += bernoulli_lpmf(choice | p) 
}

# !! We have the data/choices, so we want to see how likely the choices are! Opposite way around to Assignment 01 !!
# How many times when you sample do you get the choice actually made.
