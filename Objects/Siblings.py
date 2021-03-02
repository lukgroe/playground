import playground.Objects.Numericals as Num
import playground.Objects.Explore as Exp
import numpy as np
import matplotlib.pyplot as plt
import math


N = Num.Numericals
E = Exp.Explore


class Sibling(N, E):

    def __init__(self,
                 density_alternative=N.normal_density(N.support_distribution, 0, 1),
                 density=N.normal_density(N.support_distribution, 0, 1),
                 support_distributions=N.support_distribution,
                 sample=None,
                 size=200,
                 optimization_forward=False
                 ):

        # Input
        self.density_alternative = density_alternative
        self.density = density
        self.support_distributions = support_distributions

        self.size = size
        self.sample = sample
        self.optimization_forward = optimization_forward

    @property
    def distribution_alternative(self):
        return N.integrate(self.density_alternative, self.support_distributions)

    @distribution_alternative.setter
    def distribution_alternative(self, new_distribution):
        self.density_alternative = N.differentiate(new_distribution, self.support_distributions)

    @property
    def distribution(self):
        return N.integrate(self.density, self.support_distributions)

    @distribution.setter
    def distribution(self, new_distribution):
        self.density = N.differentiate(new_distribution, self.support_distributions)

    @property
    def quantile(self):
        return N.get_quantile_function(self.distribution, self.support_distributions)

    @property
    def quantile_density(self):
        return N.get_quantile_dens(self.quantile)

    @property
    def quantile_alternative(self):
        return N.get_quantile_function(self.distribution_alternative, self.support_distributions)

    @property
    def sample_after_applying_null_distribution(self):
        result = N.transform_mapped_dist_support_sample_on_uniform_support_with_dist(self.sample,
                                                                                     self.distribution,
                                                                                     self.support_distributions)
        return result

    @property
    def density_after_applying_null_distribution(self):
        result = N.prod(
                        N.compose_functions(self.support_distributions,
                                            self.quantile,
                                            self.density_alternative),
                        self.quantile_density
                        )
        return result

    @property
    def distribution_after_applying_null_distribution(self):
        return N.integrate(self.density_after_applying_null_distribution, N.support_uniform)

    @property
    def distribution_optimized(self):
        return N.integrate(self.density_optimized, N.support_uniform)

    @property
    def optimization_measure(self):
        helper = 0
        for i,j in zip(self.density, self.density_alternative):
            helper += i-j
        if helper == 0:
            result = list(range(len(self.density)))
        else:
            result = N.measure_preserving_opt(self.density_after_applying_null_distribution, self.optimization_forward)
        return result

    @property
    def density_optimized(self):
        return N.apply_measure_preserving_opt_to_numbers(self.density_after_applying_null_distribution,
                                                         self.optimization_measure)

    @property
    def empirical_distributions(self):

        emp_distribution_raw = N.empirical_df(self.sample,
                                              self.support_distributions)

        emp_distribution_apply_null = N.empirical_df(self.sample_after_applying_null_distribution,
                                                     self.support_uniform)

        emp_distribution_opt = N.apply_measure_opt_on_mapped_sample2(self.sample_after_applying_null_distribution, self.optimization_measure)

        result = {'raw': emp_distribution_raw,
                  'apply_null': emp_distribution_apply_null,
                  'optimized': emp_distribution_opt}

        return result

    @property
    def sibling(self):
        result = []
        size = len(self.sample)
        last = 0
        last_value = 0
        times = list(np.linspace(0, 1, len(self.distribution)))
        for time, emp in zip(times, self.empirical_distributions['optimized']):
            if emp != 1:
                result += [(size-1)/size - emp + (1/size)*((1-last)/(1-time))**(size*(1-emp))]
            else:
                result += [0]
            if last_value != emp:
                last_value = emp
                last = time
        return result

    @property
    def conditional_intensity(self):
        result = []
        times = list(np.linspace(0, 1, len(self.distribution)))
        for dist, dens, emp in zip(times, [1]*len(times), self.empirical_distributions['optimized']):
            if emp != 1:
                result += [(self.size*(1-emp))*(dens/(1-dist))]
            else:
                result += [0]
        return result

    @property
    def sample_optimized(self):
        result = []
        times = list(np.linspace(0, 1, len(self.distribution)))
        current = 0
        for check, time in zip(self.empirical_distributions['optimized'], times):
            if check != current:
                result = result + [time]*(round((check-current)*self.size))
                current = check
        return result

    @property
    def sibling_optimized(self):
        result = [1]
        times = list(np.linspace(0, 1, len(self.distribution)))
        process_state = 1
        event_set = self.sample_optimized
        for emp, intensity, time in zip(self.empirical_distributions['optimized'], self.conditional_intensity, times):
            if time in self.sample_optimized:
                #result += [process_state - process_state - (-process_state)*self.conditional_intensity*1/len(self.intensity)]
                event_set.remove(time)
        return result

    @property
    def change_point(self):
        times = list(np.linspace(0, 1, len(self.distribution)))
        for time, check in zip(times, self.density_optimized):
            if self.optimization_forward:
                if check > 1:
                    return time
            else:
                if check < 1:
                    return time

    def create_sample(self, make_plot=False):
        result = list(np.random.uniform(size=self.size, low=0, high=1))
        result = N.transform_uniform_sample_in_any_dist(result,
                                                        self.quantile_alternative,
                                                        self.support_distributions)

        result = N.map_array_on_support(result, self.support_distributions)

        self.sample = result

        if make_plot:
            self.explore_origin()

"""
x = Sibling()
x.create_sample(make_plot=False)
#x.explore_origin()
#x.explore_optimized()
#x.explore_after_applying_null_distribution()
test =  N.integrate(x.conditional_intensity, N.support_uniform)
result = []
for i, j in zip(x.empirical_distributions['optimized'], test):
    result += [x.size*i-j]

N.make_plot( result, N.support_uniform)
"""
#N.make_plot(x.optimization_measure, N.support_uniform, show=True, col='green')

#N.make_plot(x.empirical_distributions['apply_null'], N.support_uniform, show=False)
#N.make_plot(x.distribution_after_applying_null_distribution, N.support_uniform, col='red')

#N.make_plot(x.empirical_distributions['optimized'], N.support_uniform, show=False)
#N.make_plot(x.distribution_optimized, N.support_uniform, col='red')



#N.make_plot(x.density_optimized, N.support_uniform, show=False, col='green')
#N.make_plot(x.density_after_applying_null_distribution, N.support_uniform, show=False, col='red')
#N.make_plot([1]*len(N.support_uniform), N.support_uniform, col='blue')

#N.make_plot(x.empirical_distributions['optimized'], N.support_uniform)