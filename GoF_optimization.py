import numpy as np
import matplotlib.pyplot as plt
import math
#import GetQuantiles as q


# Some constants for numerical purposes
size = 20

number_of_samples = 3
granularity = 2000
epsilon = 0.000000000000001
infty= 1000000000000000
distribution_support = np.linspace(-4, 4, granularity)
uniform_support = np.linspace(0, 1, granularity)
smooth_parameter = 1


def create_sample(size, distribution="uniform"):
    if distribution == "uniform":
        result = np.random.uniform(size=size, low=0, high=1)
    return result


def create_plot_to_function(numbers, support=uniform_support, show=True, col='black'):
    plt.plot(support, numbers, 'ro', markersize=.4, color=col)
   # plt.ylim(0, 2)
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


def normal_density(support=distribution_support, mu=0, var=1):
    result = []
    for time in support:
        result += [(1 / (math.sqrt(2 * math.pi * var))) * (math.e ** (- ((time - mu) ** 2) / (2 * var)))]
    result = np.array(result)
    return result


#def get_normal_quant_density(mu=0,var=1):
 #   result = []
  #  helper = list(normal_density(distribution_support, mu, var))
   # print(helper)
   # for i in helper:
    #    result += [1/i]
    #print(result)
 #   return result
#

def integrate_function(numbers, support=distribution_support):
    width = support[1]-support[0]
    result = []
    current = 0
    for i in numbers:
        current = current + i*width
        result += [current]
    result = np.array(result)
    return result


def differentiate_function(numbers, support=distribution_support):
    sup = support[1]-support[0]
    #result = [(numbers[1] - numbers[0]) / sup]
    result = []
    num_old = numbers[0]
    #for num in numbers:
    for i in range(granularity):
        if len(result)<smooth_parameter:
            result += [(numbers[i]-num_old)/sup]
            num_old = numbers[i]
        else:
            helper = 0
            for smh in range(smooth_parameter):
                helper += (numbers[i-smh]-numbers[i-smh-1])/sup
            helper = helper/smooth_parameter

            result +=[helper]      #[((numbers[i]-numbers[i-1])/sup + (numbers[i-1]-numbers[i-2])/sup + (numbers[i-2]-numbers[i-3])/sup + (numbers[i-3]-numbers[i-4])/sup +(numbers[i-4]-numbers[i-5])/sup)/5]
            num_old = numbers[i]


    """if num > num_old:
        result += [(num - num_old) / sup]
        num_old = num
    """

    result = np.array(result)
    return result

"""
def get_quantile_function(numbers, support=distribution_support):
    result = []
    numbers = list(numbers)
    numbers += [42]
    support = np.concatenate((support, np.array([support[len(support)-1]])))

    for i in uniform_support:
        last = i
        for j in range(granularity+1):
            if(last < numbers[j]):
                result += [support[j]]

                last = 42

    return result
"""


def get_quantil_at_point(point, distribution, support=distribution_support):
    new_sup = distribution
    new_map = support
    result = infty
    for i in range(granularity-1):
        if new_sup[i] < point and point < new_sup[i+1]:
            result = new_map[i] + (point - new_sup[i])*((new_map[i+1]-new_map[i])/(new_sup[i+1]-new_sup[i]))
            break
    return result


def get_quantile_function(distribution, support=distribution_support):
    result = []
    for i in uniform_support:
        result += [get_quantil_at_point(i, distribution)]
    result[0] = distribution_support[0]#result[1] -(result[2]-result[1])
    for i in range(granularity):
        if result[i]+1>infty:
            result[i]=distribution_support[granularity-1]
    return result


def get_quantil_dens(distribution):
    result = differentiate_function(get_quantile_function(distribution), uniform_support)
    help = 0
    for i in result:
        if i < 0:
            result[help]=0
        help +=1
    result[0] = result[1]
    return result


def compose_functions(support_2, numbers_1, numbers_2):
    result = []
    support_2 = list(support_2)
    support_2 += [infty]
    numbers_2 = list(numbers_2)
    numbers_2 += [numbers_2[len(numbers_2)-1]]

    for i in numbers_1:
        last = i
        for j in range(granularity+1):
            if support_2[j] > last:
                result += [numbers_2[j] ]
                last = infty
    result = np.array(result)
    return result


def prod(numbers1, numbers2):
    result = []
    for i,j in zip(numbers1,numbers2):
        result += [i*j]#

    return result


def smooth(numbers, times):
    result = numbers
    for i in range(times):
        result = differentiate_function(integrate_function(result, uniform_support), uniform_support)
    return result


def measure_preserving_opt(numbers, rev=True):
    result = []
    #result = [sorted(numbers).index(x) for x in numbers]
    list_to_check = list(numbers)
    location=0
    for check in range(granularity):
        help = 0
        for i in range(granularity):
            if help <= list_to_check[i]:
                help = list_to_check[i]
                location = i
        result += [location]
        list_to_check[location]=-1
    if rev:
        list.reverse(result)
    return result


def apply_measure_preserving_opt_to_numbers(numbers, measure):
    result = [numbers[i] for i in measure]
    return result


def apply_measure_opt_on_mapped_sample(sample, opt_uniform_support):
    result = []
    for i in sample:
        for check in range(granularity):
            if i == uniform_support[check]:
                result += [ uniform_support[opt_uniform_support[check]] ]
    return result


def apply_measure_opt_on_mapped_sample2(sample, measure, renormalized = False):
    result = [0]*granularity

    for i in sample:
        for check in range(granularity):
            if i == uniform_support[check]:
                #result[measure[check]] += 1
                result[check] += 1

    result = apply_measure_preserving_opt_to_numbers(result, measure)

    help = 0
    calc = 0
    for i in result:
        calc = calc + i
        result[help] = calc
        help +=1

    if not renormalized:
        for i in range(granularity):
            result[i] = result[i]/size

    return result


def transform_uniform_sample_in_any_dist(sample, quantil):
    result = []
    for i in sample:
        for check in range(granularity):
            if i < uniform_support[check]:
                result += [quantil[check]]
                i = infty
    result = map_array_on_support(result, distribution_support)
    return result


def transform_mapped_dist_support_sample_on_uniform_support_with_dist(sample, dist):
    result = []
    for i in sample:
        for check in range(granularity):
            if i < distribution_support[check]:
                result += [dist[check]]
                i=infty
    result = map_array_on_support(result, uniform_support)
    return result


def get_empirical_dist_intensity_uniform_sample(empirical_dist):
    result = []
    already_occured = 0
    for i in range(granularity):
        result += [(size - already_occured)/(1-uniform_support[i]-epsilon)]
        already_occured = empirical_dist[i]*size
    return result


def get_absolute_difference_null_to_alter(empirical_dist, dens_alter):
    result = []
    null_intensity2 = get_empirical_dist_intensity_uniform_sample(empirical_dist)
   # print(null_intensity)
    dist_alter = integrate_function(dens_alter, uniform_support)
    already_occured = 0
    for i in range(granularity):
        help = 0
        if empirical_dist[i]*size < size:

            help = (dens_alter[i]*(size-empirical_dist[i]*size))/(1-dist_alter[i])-null_intensity2[i]#- 1/(1-uniform_support[i])
        else:
            help=0
        result += [help]#- null_intensity[i]]
        already_occured = empirical_dist[i] * size

    return result


def get_stochastic_integral_to_strategy(empirical_dist):
    cond_intens = get_empirical_dist_intensity_uniform_sample(empirical_dist)
    result = []
    lebesgue_part = 0
    for i in range(granularity):
        lebesgue_part += cond_intens[i]*(uniform_support[1]-uniform_support[0])
        result += [empirical_dist[i]*size-lebesgue_part]
    return result
"""
mu_0 = 0
var_0 = 1
mu_1 = 0#0.05
var_1 = 1

dens_null = normal_density(mu=mu_0,var=var_0)
dens_alter = normal_density(mu=mu_1, var=var_1)

distribution_null = integrate_function(dens_null, distribution_support)
distribution_alter = integrate_function(dens_alter, distribution_support)

quant_null = get_quantile_function(distribution_null)
#quant_null = get_quantil_path(distribution_null)
quant_alter = get_quantile_function(distribution_alter)


quant_dens_null = differentiate_function(quant_null, uniform_support)#get_normal_quant_density(mu=mu_0,var=var_0)
#quant_dens_null = get_quantil_dens(quant_null)
quant_dens_alter = differentiate_function(quant_alter, uniform_support)
quant_dens_alter = get_quantil_dens(quant_alter)

compose_null =integrate_function(prod(compose_functions(distribution_support, quant_null, dens_null), quant_dens_null), uniform_support)
compose_alter =integrate_function(prod(compose_functions(distribution_support, quant_null, dens_alter), quant_dens_null), uniform_support)

compose_alter_dens = differentiate_function(compose_alter,uniform_support)

orders = measure_preserving_opt(compose_alter_dens)

density_final = apply_measure_preserving_opt_to_numbers(compose_alter_dens, orders)

create_plot_to_function(differentiate_function(integrate_function(density_final, uniform_support), uniform_support), uniform_support, show=False, col='blue')
create_plot_to_function(differentiate_function(compose_null,uniform_support), uniform_support, show=False)
create_plot_to_function(differentiate_function(integrate_function(compose_alter_dens, uniform_support), uniform_support), uniform_support, col='red')

"""
"""
create_plot_to_function(integrate_function(density_final, uniform_support), uniform_support, show=False, col='blue')
create_plot_to_function(compose_null, uniform_support, show=False)
create_plot_to_function(integrate_function(compose_alter_dens, uniform_support), uniform_support, col='red')
"""
"""
sample = map_array_on_support(create_sample(size))
sample_alter = transform_mapped_dist_support_sample_on_uniform_support_with_dist(transform_uniform_sample_in_any_dist(sample, quant_alter), distribution_null)


check_opt = apply_measure_opt_on_mapped_sample2(sample,orders)
check_opt2 = apply_measure_opt_on_mapped_sample2(sample_alter, orders)
"""
#print(check_opt2)
"""
create_plot_to_function(empirical_df(sample_alter), uniform_support, col='red', show=False)
create_plot_to_function(check_opt2, uniform_support, col='blue', show=False)
create_plot_to_function(check_opt, uniform_support)
"""

#check_opt2  density_final
#create_plot_to_function(get_absolute_difference_null_to_alter(check_opt2, dens_alter),uniform_support, col="red")
#create_plot_to_function(get_empirical_dist_intensity_uniform_sample(check_opt2),uniform_support)

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


