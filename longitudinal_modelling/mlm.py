import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.regression.mixed_linear_model as mlm
from longitudinal_modelling.longitudinal_utils import *


def make_smf_formula(target, covariates=None, timestamp=None):
    if covariates is None:
        return target + ' ~ ' + timestamp

    covariates = to_list(covariates)
    str_cov = ' + '.join(covariates)
    if timestamp is not None:
        str_cov = timestamp + ' + ' + str_cov
        add_slope = ' + ' + timestamp + ' * '
        str_cov += add_slope + add_slope.join(covariates)
    return target + ' ~ ' + str_cov


def fit_mlm(df, group, target, covariates, timestamp, rdn_slope=True, method=['lbfgs']):
    df_local = df[to_list(group)+to_list(target)+to_list(covariates)+to_list(timestamp)].copy()
    df_local.dropna(inplace=False)
    print(len(df)-len(df_local), 'na values dropped')
    df_local[target] = df_local[target].values.reshape(-1, 1)
    print('target shape:', df_local[target].shape, 'covariates shape:', df_local[covariates].shape if covariates is not None else 0)
    r_formula = make_smf_formula(target=target, covariates=covariates, timestamp=timestamp)
    print('running', r_formula)
    if rdn_slope and (timestamp is not None):
        # random intercept, and random slope (with respect to time)
        md = smf.mixedlm(r_formula, df_local, groups=df_local[group], re_formula='~' + timestamp)
    else:
        # random intercept only
        md = smf.mixedlm(r_formula, df_local, groups=df_local[group])
    mdf = md.fit(method=method, reml=True)  # other methods lbfgs bfgs cg
    coeffs = mdf.summary().tables[1]
    print(coeffs.loc[pd.to_numeric(coeffs['P>|z|']) <= 0.05])
    del df_local
    return {'model': mdf, 'stats': mdf.summary().tables[0], 'coeffs': coeffs}


def fit_mlm_constraint(df, group, target, covariates, timestamp, rdn_slope=True, method=['lbfgs']):
    # fit a model in which the two random effects are constrained to be uncorrelated:
    r_formula = make_smf_formula(target=target, covariates=covariates, timestamp=timestamp)
    md = smf.mixedlm(r_formula, df, groups=df[group], re_formula='~' + timestamp)
    free = sm.regression.mixed_linear_model.MixedLMParams.from_components(np.ones(2), np.eye(2))

    mdf = md.fit(free=free, method=method)
    print(mdf.summary())
    return mdf.summary()
