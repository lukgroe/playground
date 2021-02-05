import numpy as np


size = 79;
number_of_samples = 3;
granularity = 10;
epsilon = 0.000000000000001;
infty= 1000000000000000;
distribution_support = np.linspace(-4, 4, granularity)





"""
  Let X_1, X_2,..., X_n be a sequence of independent random variables with distribution F. We throughout
  assume that the distribution F admits a Lebesgue density f. Furthermore assume that the random variables (we also
  call them events) are observed over time and the statistician needs to decide (as soon as possible) if the events
  belong to the distribution F or a alternative model G (admitting the density g). Hence we consider the situations

  H_0: X_1, X_2,..., X_n is iid with distribution F.
                                                                                       (I.)
  H_1: X_1, X_2,..., X_n is iid with distribution G.

  Our goal is to invent new test procedures for the given situation.

  Since F(X_i) is uniform distributed on the unit intervall, it is sufficient to consider the uniform distributed
  case. Note, under the alternative H_1 it follows that the distribution of the random variable F(X_i) admits the
  Lebesgue density g(F^{-1})*f^{-1}. Consequently, the situation in (II.) becomes

  H_0: F(X_1), F(X_2),..., F(X_n) is iid with uniform distribution on [0,1].
                                                                                       (II.)
  H_1: F(X_1), F(X_2),..., F(X_n) is iid with density g(F^{-1})*f^{-1}.

  The investigations lead to a stochastic process M with the properties

  P^0[ sup M > c ]= 1/c and P^1[sup M > c]>1/c.

  So the process M is a good framework for problem (II.)
"""