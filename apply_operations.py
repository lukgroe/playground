import GoF_optimization as gof




""" STEP 1: Specify the alternative/null distribution. """
size = 1000
mu_0 = 0
var_0 = 1
mu_1 = 0
var_1 = 1

# THE DENSITIES
density_null = gof.normal_density(mu=mu_0,var=var_0)
density_alternative = gof.normal_density(mu=mu_1, var=var_1)

"""
gof.create_plot_to_function(dens_null, gof.uniform_support, show=False)
gof.create_plot_to_function(dens_alter, gof.uniform_support, col='red')
"""




""" STEP 2: Generate Sample following the distribution under the alternative """
# needed quantities in this step:
distribution_alternative = gof.integrate_function(density_alternative)
quantile_alternative = gof.get_quantile_function(distribution_alternative)

# THE RAW SAMPLE:
sample_under_alternative = gof.transform_uniform_sample_in_any_dist(gof.map_array_on_support(gof.create_sample(size)), quantile_alternative)

# plot to compare empirical distribution with underlying distribution
"""
# compare empirical distribution with the underlying distribution
gof.create_plot_to_function(distribution_alternative, gof.distribution_support, show=False, col="red")
gof.create_plot_to_function(gof.empirical_df(sample_under_alternative, gof.distribution_support), gof.distribution_support)
"""




""" STEP 3: Apply the null-distribution on the underlying data set. Compute the distribution of the resulting transformed sample """
distribution_null = gof.integrate_function(density_null)
sample_under_alternative_apply_null = gof.transform_mapped_dist_support_sample_on_uniform_support_with_dist(sample_under_alternative, distribution_null)


gof.create_plot_to_function(gof.empirical_df(sample_under_alternative_apply_null, gof.uniform_support), gof.uniform_support, show=False)


quantile_null = gof.get_quantile_function(distribution_null)
quantile_density_null = gof.differentiate_function(quantile_null, gof.uniform_support)
distribution_alternative_transformed = gof.integrate_function(gof.prod(gof.compose_functions(gof.distribution_support, quantile_null, density_alternative), quantile_density_null), gof.uniform_support)

gof.create_plot_to_function(distribution_alternative_transformed, gof.uniform_support, col="red")
