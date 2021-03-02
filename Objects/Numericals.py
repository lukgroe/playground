import numpy as np
import math
import matplotlib.pyplot as plt
import random


class Numericals(object):

    # Default-Values
    granularity = 1500
    support_distribution = list(np.linspace(-4, 4, granularity))
    support_uniform = list(np.linspace(0, 1, granularity))



    # Fixed-Values
    infty = 1000000000000000

    @staticmethod
    def empirical_df(numbers, support):
        result = np.array([])
        for time in support:
            occur = [i for i in numbers if i!=None if i <= time]
            result = np.concatenate((result, np.array([len(occur) / len(numbers)])))
        result = list(result)
        return result

    @staticmethod
    def normal_density(support, mu=0, var=1):
        result = []
        for time in support:
            result += [(1 / (math.sqrt(2 * math.pi * var))) * (math.e ** (- ((time - mu) ** 2) / (2 * var)))]
        result = list(np.array(result))
        return result

    @staticmethod
    def integrate(numbers, support):
        width = support[1]-support[0]
        result = []
        current = 0
        for i in numbers:
            current = current + i*width
            result += [current]
        result = list(result)
        return result

    @staticmethod
    def differentiate(numbers, support, smooth_parameter=1):
        sup = support[1] - support[0]
        # result = [(numbers[1] - numbers[0]) / sup]
        result = []
        num_old = numbers[0]
        # for num in numbers:
        for i in range(len(support)):
            if len(result) < smooth_parameter:
                result += [(numbers[i] - num_old) / sup]
                num_old = numbers[i]
            else:
                helper = 0
                for smh in range(smooth_parameter):
                    helper += (numbers[i - smh] - numbers[i - smh - 1]) / sup
                helper = helper / smooth_parameter

                result += [helper]
                num_old = numbers[i]

        result = list(result)
        result[0]=result[1]
        return result

    @staticmethod
    def get_quantil_at_point(point, distribution, support):
        new_sup = distribution
        new_map = support
        result = Numericals.infty
        for i in range(len(distribution) - 1):
            if new_sup[i] < point < new_sup[i + 1]:
                result = new_map[i] + (point - new_sup[i]) * (
                            (new_map[i + 1] - new_map[i]) / (new_sup[i + 1] - new_sup[i]))
                break
        return result

    @staticmethod
    def get_quantile_function(distribution, support_distribution):
        result = []
        for i in Numericals.support_uniform:
            result += [Numericals.get_quantil_at_point(i, distribution, support_distribution)]
        result[0] = support_distribution[0]
        for i in range(len(distribution)):
            if result[i] + 1 > Numericals.infty:
                result[i] = support_distribution[Numericals.granularity - 1]
        return result

    @staticmethod
    def map_number_on_support(number, support=support_uniform):
        result = None
        for i in support:
            if number is not None:
                if number < i:
                    result = i
                    break
        return result

    @staticmethod
    def map_array_on_support(numbers, support=support_uniform):
        result = np.array([])
        for i in numbers:
            result = list(np.concatenate((result, np.array([Numericals.map_number_on_support(i, support)]))))
        return result

    @staticmethod
    def get_quantile_dens(quantil):
        result = Numericals.differentiate(quantil, Numericals.support_uniform)
        help = 0
        for i in result:
            if i < 0:
                result[help] = 0
            help += 1
        result[0] = result[1]
        return result

    @staticmethod
    def transform_mapped_dist_support_sample_on_uniform_support_with_dist(sample, distribution, support_distribution):
        result = []
        for i in sample:
            for check in range(len(distribution)):
                if i is not None:
                    if i < support_distribution[check]:
                        result += [distribution[check]]
                        i = Numericals.infty
        result = Numericals.map_array_on_support(result, Numericals.support_uniform)
        return result

    @staticmethod
    def transform_uniform_sample_in_any_dist(sample, quantile, distribution_support):
        result = []
        for i in sample:
            for check in range(Numericals.granularity):
                if i < Numericals.support_uniform[check]:
                    result += [quantile[check]]
                    i = Numericals.infty
        result = Numericals.map_array_on_support(result, distribution_support)
        return result

    @staticmethod
    def compose_functions(support_2, numbers_1, numbers_2):
        result = []
        support_2 = list(support_2)
        support_2 += [Numericals.infty]
        numbers_2 = list(numbers_2)
        numbers_2 += [numbers_2[len(numbers_2) - 1]]

        for i in numbers_1:
            last = i
            for j in range(Numericals.granularity + 1):
                if support_2[j] > last:
                    result += [numbers_2[j]]
                    last = Numericals.infty
        result = list(result)
        return result

    @staticmethod
    def prod(numbers1, numbers2):
        result = []
        for i, j in zip(numbers1, numbers2):
            result += [i * j]  #

        return result

    @staticmethod
    def measure_preserving_opt(numbers, rev=True):

        result = []
        list_to_check = list(numbers)
        location = 0
        for check in range(len(numbers)):
            help = 0
            for i in range(len(numbers)):
                if help <= list_to_check[i]:
                    help = list_to_check[i]
                    location = i
            result += [location]
            list_to_check[location] = -1
        if rev:
            list.reverse(result)
        return result

    @staticmethod
    def apply_measure_opt_on_mapped_sample2(sample, measure, renormalized=False):
        result = [0] * len(measure)

        for i in sample:
            for check in range(len(measure)):
                if i == Numericals.support_uniform[check]:
                    result[check] += 1

        result = [result[i] for i in measure]

        help = 0
        calc = 0
        for i in result:
            calc = calc + i
            result[help] = calc
            help += 1

        if not renormalized:
            for i in range(len(measure)):
                result[i] = result[i] / len(sample)
        return result

    @staticmethod
    def make_plot(numbers, support, show=True, col='black', label=None):
        plt.plot(0, 0, markersize=.4, color=col, label=label)
        if label !=None:
            plt.legend()
        plt.plot(support, numbers, 'ro', markersize=.4, color=col)
        if show:
            plt.show()

    @staticmethod
    def apply_measure_preserving_opt_to_numbers(numbers, measure):
        result = [numbers[i] for i in measure]
        return result
