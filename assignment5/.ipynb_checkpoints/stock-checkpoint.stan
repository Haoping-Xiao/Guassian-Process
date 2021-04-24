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

functions {
#include basis.stan
}

data {
  int<lower=1> N; //number of observations
  vector[N] x; //univariate 'time' start from 1
  vector[N] y; //the highest price
  
  real<lower=0> c_f;   // factor c to determine the boundary value L
  int<lower=1> M_f;    // number of basis functions
}

transformed data {
  // Normalize data
  real xmean = mean(x);
  real ymean = mean(y);
  real xsd = sd(x);
  real ysd = sd(y);
  vector[N] xn = (x - xmean)/xsd;
  vector[N] yn = (y - ymean)/ysd;
  // Basis functions for f
  real L_f = c_f*max(xn);
  matrix[N,M_f] PHI_f = PHI_EQ(N, M_f, L_f, xn);
}


// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real intercept;
  vector[M_f] beta_f;          // the basis functions coefficients
  real<lower=0> lengthscale_f; // lengthscale of f
  real<lower=0> sigma_f;       // scale of f
  vector<lower=0>[N] sigma_y;         // scale of y
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  // spectral densities for f and g
  vector[M_f] diagSPD_f = diagSPD_EQ(sigma_f, lengthscale_f, L_f, M_f);
  // priors
  intercept ~ normal(0, 1);
  beta_f ~ normal(0, 1);
  lengthscale_f ~ normal(0, 1);
  sigma_f ~ normal(0, .5);
  sigma_y~normal(0,5);
  yn~normal(intercept + PHI_f * (diagSPD_f .* beta_f),sigma_y);
}

generated quantities{
  vector[N] f;
  {
    vector[M_f] diagSPD_f = diagSPD_EQ(sigma_f, lengthscale_f, L_f, M_f);
    f=(intercept + PHI_f * (diagSPD_f .* beta_f))*ysd+ymean;
  }
}


