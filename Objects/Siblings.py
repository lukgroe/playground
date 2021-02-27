import playground.Objects.Numericals as Num
import numpy as np
import matplotlib.pyplot as plt
import math


N = Num.Numericals


class Sibling(N):

    def __init__(self, sample=None, density_alternative=N.normal_density(N.support_distribution, 0, 1), support_distribution=N.support_distribution, density=[1]*N.granularity, support=N.support_uniform, size=4):

        # Input
        self.density_alternative = density_alternative
        self.density = density
        self.support_distributions = support_distribution

        self.size =size
        self.sample = sample

    @property
    def distribution_alternative(self):
        return N.integrate(self.density_alternative, self.support_distribution)

    @distribution_alternative.setter
    def distribution_alternative(self, distribution_a):
        self.density_alternative = N.differentiate(distribution_a, self.support_distribution)

    @property
    def quantil_alternative(self):
        return N.get_quantile_function(self.distribution_alternative, self.support_distribution)

    @property
    def quantil_density_alternative(self):
        return N.get_quantil_dens(self.quantil_alternative)

    def create_sample(self, support=0, quantil=0):
        result = list(np.random.uniform(size=self.size, low=0, high=1))
        result = N.map_array_on_support(result)
        return result





x = Sibling()

#print(x.create_sample())

#print(x.quantil_density_alternative)

N.make_plot(x.quantil_density_alternative, x.support_distribution)



