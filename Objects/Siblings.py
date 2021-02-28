import playground.Objects.Numericals as Num
import numpy as np
import matplotlib.pyplot as plt
import math


N = Num.Numericals


class Sibling(N):

    def __init__(self,
                 density_alternative=N.normal_density(N.support_distribution, 0, .5),
                 density=N.normal_density(N.support_distribution, 0, 1),
                 support_distributions=N.support_distribution,
                 sample=None,
                 size=1000,
                 optimization_forward=True
                 ):

        # Input
        self.density_alternative = density_alternative
        self.density = density
        self.support_distributions = support_distributions

        self.size = size
        self.sample = sample
        self.optimization_direction = optimization_forward

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
    def optimization_measure(self):
        return N.measure_preserving_opt(self.density_after_applying_null_distribution, self.optimization_direction)

    @property
    def density_optimized(self):
        return 4

    @property
    def empirical_distributions(self):

        emp_distribution_raw = N.empirical_df(self.sample,
                                              self.support_distributions)

        emp_distribution_apply_null = N.empirical_df(self.sample_after_applying_null_distribution,
                                                     self.support_uniform)

        emp_distribution_opt = N.apply_measure_opt_on_mapped_sample2(self.sample_after_applying_null_distribution,self.optimization_measure)

        result = {'raw': emp_distribution_raw,
                  'apply_null': emp_distribution_apply_null,
                  'optimized': emp_distribution_opt}

        return result

    def create_sample(self, make_plot=False):
        result = list(np.random.uniform(size=self.size, low=0, high=1))

        result = N.transform_uniform_sample_in_any_dist(result,
                                                        self.quantile_alternative,
                                                        self.support_distributions)

        result = N.map_array_on_support(result, self.support_distributions)

        self.sample = result

        if make_plot:
            N.make_plot(self.distribution_alternative,
                        self.support_distributions,
                        show=False,
                        col="red")

            N.make_plot(self.empirical_distributions['raw'], self.support_distributions, show=True)


x = Sibling()
x.create_sample(make_plot=False)

N.make_plot(x.empirical_distributions['raw'], N.support_uniform)
N.make_plot(x.empirical_distributions['apply_null'], N.support_uniform)
N.make_plot(x.empirical_distributions['optimized'], N.support_uniform)


#N.make_plot(x.empirical_distributions['optimized'], N.support_uniform)

#x.optimization_direction=False
print(x.sample_after_applying_null_distribution)