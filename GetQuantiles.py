import GoF_optimization as GoF

granularity = GoF.granularity

uni_sup = GoF.uniform_support
dist_sup = GoF.distribution_support
distribution = GoF.distribution_null
quantil_old = GoF.quant_null


def get_quantil_at_point(point, distribution, support=dist_sup):
    new_sup = distribution
    new_map = support
    result = GoF.infty


    for i in range(granularity-1):
        if new_sup[i] < point and point < new_sup[i+1]:
            result = new_map[i] + (point - new_sup[i])*((new_map[i+1]-new_map[i])/(new_sup[i+1]-new_sup[i]))
            break
    return result


def get_quantil_path(distribution, support=dist_sup):
    result = []
    for i in uni_sup:
        result += [get_quantil_at_point(i, distribution)]
    result[0] = dist_sup[0]#result[1] -(result[2]-result[1])
    for i in range(granularity):
        if result[i]+1>GoF.infty:
            result[i]=dist_sup[granularity-1]
    return result


def get_quantil_dens(distribution):
    result = GoF.differentiate_function(get_quantil_path(distribution), uni_sup)
    help = 0
    for i in result:
        if i < 0:
            result[help]=0
        help +=1
    result[0] = 1
    result[granularity-1]=1
    return result



GoF.create_plot_to_function(get_quantil_path(distribution),uni_sup, show = True)
GoF.create_plot_to_function(get_quantil_dens(distribution), uni_sup, show = True)

#GoF.create_plot_to_function(distribution, dist_sup, show = True)
