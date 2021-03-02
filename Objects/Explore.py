import matplotlib.pyplot as plt
import numpy as np


class Explore(object):

    @classmethod
    def make_plot(cls,numbers, support, show=True, col='black', label=None):
        plt.plot(0, 0, markersize=.4, color=col, label=label)
        if label !=None:
            plt.legend()
        plt.plot(support, numbers, 'ro', markersize=.4, color=col)
        if show:
            plt.show()

    def explore_origin(self):
        Explore.make_plot(self.distribution,
                    self.support_distributions,
                    show=False,
                    col="blue",
                    label="$F^{H_0}$")

        Explore.make_plot(self.distribution_alternative,
                    self.support_distributions,
                    show=False,
                    col="red",
                    label="$F^{H_1}$")

        Explore.make_plot(self.empirical_distributions['raw'],
                    self.support_distributions,
                    show=True,
                    label='$F_{n}$')

    def explore_optimized(self):
        Explore.make_plot(list(np.linspace(0, 1, len(self.distribution))),
                          list(np.linspace(0, 1, len(self.distribution))),
                          show=False,
                          col="blue",
                          label="$\mu_{opt}*F^{H_0}$"
                          )

        Explore.make_plot(self.distribution_optimized,
                          list(np.linspace(0, 1, len(self.distribution))),
                          show=False,
                          col="red",
                          label="$\mu_{opt}*F^{H_1}$"
                          )

        Explore.make_plot(self.empirical_distributions['optimized'],
                          list(np.linspace(0, 1, len(self.distribution))),
                          show=True,
                          label='$F_n^{\mu_{opt}}$'
                          )

    def explore_after_applying_null_distribution(self):
        Explore.make_plot(list(np.linspace(0, 1, len(self.distribution))),
                          list(np.linspace(0, 1, len(self.distribution))),
                          show=False,
                          col="blue",
                          label="$F*F^{H_0}$"
                          )

        Explore.make_plot(self.distribution_after_applying_null_distribution,
                          list(np.linspace(0, 1, len(self.distribution))),
                          show=False,
                          col="red",
                          label="$F*F^{H_1}$"
                          )

        Explore.make_plot(self.empirical_distributions['apply_null'],
                          list(np.linspace(0, 1, len(self.distribution))),
                          show=True,
                          label='$F_n^{F}$'
                          )

    def explore_sibling(self):
        label = True
        show = True
        plt.plot(0, 0, markersize=.4, label='Test-process', color='black')
        if label != None:
            plt.legend()
        plt.plot(list(np.linspace(0, 1, len(self.distribution))),
                 self.sibling,
                 markersize=.4,
                 color='black'
                 )
        if show:
            plt.show()

    def explore_density_optimized(self):
        Explore.make_plot(self.density_optimized,
                          list(np.linspace(0, 1, len(self.distribution))),
                          show=True,
                          col="blue",
                          label="Optimized density comparison"
                          )
