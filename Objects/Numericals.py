import numpy as np
import math
import matplotlib.pyplot as plt


class Numericals(object):

    # Default-Values
    granularity = 3000
    support_distribution = list(np.linspace(-4, 4, granularity))
    support_uniform = list(np.linspace(0, 1, granularity))
    mu=1
    var=1


    # Fixed-Values
    infty = 1000000000000000

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
    def get_quantil_dens(quantil):
        result = Numericals.differentiate(quantil, Numericals.support_uniform)
        help = 0
        for i in result:
            if i < 0:
                result[help] = 0
            help += 1
        result[0] = result[1]
        return result

    @staticmethod
    def make_plot(numbers, support=support_uniform, show=True, col='black'):
        plt.plot(support, numbers, 'ro', markersize=.4, color=col)
        # plt.ylim(0, 2)
        if show:
            plt.show()