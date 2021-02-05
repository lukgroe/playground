import numpy as np
import matplotlib.pyplot as plt
import math

# First: Set the key parameters for the described scenario

mu_0 = 0
var_0 = 1
mu_1 = 0.2
var_1 = 0.8

# Some constants for numerical purposes
size = 50
number_of_samples = 3
granularity = 1500
epsilon = 0.000000000000001
infty= 1000000000000000
distribution_support = np.linspace(-4, 4, granularity)
uniform_support = np.linspace(0, 1, granularity)


def create_plot_to_function(numbers, support=uniform_support, show=True):
    plt.plot(support, numbers, 'ro', markersize=1, color='black')
    if show:
        plt.show()


def map_number_on_support(number, support=uniform_support):
    result = None
    for i in support:
        if number < i:
            result = i
            break
    return result


def map_array_on_support(numbers, support=uniform_support):
    result = np.array([])
    for i in numbers:
        result = np.concatenate((result, np.array([map_number_on_support(i, support)])))
    return result


def empirical_df(numbers, support=uniform_support):
    result = np.array([])
    for time in support:
        occur = [i for i in numbers if i <= time]
        result = np.concatenate((result, np.array([len(occur)/len(numbers)])))
    return result


def normal_density(support, mu=0, var=1):
    result = []
    for time in support:
        result += [(1 / (math.sqrt(2 * math.pi * var))) * (math.e ** (- ((time - mu) ** 2) / (2 * var)))]
    result = np.array(result)
    return result


def integrate_function(numbers, support=uniform_support):
    width = support[1]-support[0]
    result =[]
    old = 0
    for i in numbers:
        result = i
    return result


    return result


def create_sample(size, distribution="uniform"):
    if distribution == "uniform":
        result = np.random.uniform(size=size, low=0, high=1)
    return result


check = create_sample(size)
#print(sorted(check))
#print(map_array_on_support(sorted(check)))
#plt.ion()
#create_plot_to_function(empirical_df(check))



# plot sample, plot density
create_plot_to_function(normal_density(distribution_support), distribution_support, False)
create_plot_to_function(normal_density(distribution_support), distribution_support)


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