import numpy as np
from scipy import stats


def _two_sided_zstat_generic(value1, value2, std_diff, diff):
    zstat = (value1 - value2 - diff) / std_diff
    pvalue = stats.norm.sf(np.abs(zstat)) * 2
    return zstat, pvalue


def _two_sided_tstat_generic(value1, value2, std_diff, dof, diff):
    """The test statistic is :
        tstat = (value1 - value2 - diff) / std_diff
    and is assumed to be t-distributed with ``dof`` degrees of freedom.
    """
    tstat = (value1 - value2 - diff) / std_diff
    pvalue = stats.t.sf(np.abs(tstat), dof) * 2
    return tstat, pvalue


def two_sided_test(x1, x2, test='t_test', value=0):

    """two sample mean two-sided t-test based on normal distribution
        the samples are assumed to be independent.
        Parameters
        ----------
        x1 : array_like, 1-D or 2-D
            first of the two independent samples
        x2 : array_like, 1-D or 2-D
            second of the two independent samples
        value : float
            In the one sample case, value is the mean of x1 under the Null
            hypothesis.
            In the two sample case, value is the difference between mean of x1 and
            mean of x2 under the Null hypothesis. The test statistic is
            `x1_mean - x2_mean - value`
    """

    x1 = np.asarray(x1)
    nobs1 = x1.shape[0]
    x1_mean = x1.mean(0)
    x1_var = x1.var(0) # sample variance
    x2 = np.asarray(x2)
    nobs2 = x2.shape[0]
    x2_mean = x2.mean(0)
    x2_var = x2.var(0) # sample variance
    var_pooled = nobs1 * x1_var + nobs2 * x2_var

    if test == 't_test':
        dof = nobs1 + nobs2 - 2
    elif test == 'z_test':
        dof = nobs1 + nobs2
    else:
        raise ValueError('Either t_test or z_test!')

    var_pooled = var_pooled/dof
    var_pooled = var_pooled*(1.0 / nobs1 + 1.0 / nobs2)
    std_diff = np.sqrt(var_pooled)

    if test == 't_test:':
        stat, pval = _two_sided_tstat_generic(x1_mean, x2_mean, std_diff, dof, diff=value)
    else:
        stat, pval = _two_sided_zstat_generic(x1_mean, x2_mean, std_diff, diff=value)

    return stat, pval


if __name__ == '__main__':

    x1 = np.random.randn(500)
    x2 = np.random.randn(500)
    stat, p = two_sided_test(x1, x2, 'z_test')
    print(f'Performing Z-test, the statistic is {stat} and p-value is {p}')
