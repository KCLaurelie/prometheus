import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu

default_subcols = ('age_bucket_baseline', 'gender', 'ethnicity', 'marital_status', 'education', 'first_language',
                   'imd_bucket_baseline', 'smoking_status_baseline', 'cvd_problem_baseline',
                   'dementia_medication_baseline', 'antipsychotic_medication_baseline', 'antidepressant_medication_baseline')


def p_value_sig(p_val):
    p_val = float(p_val)
    if p_val <= 0.001: res = '***'
    elif p_val <= 0.01: res = '**'
    elif p_val <= 0.05:res = '*'
    else: res = ''
    return res


def cohens_d(df, value='score_combined', group_col='patient_group'):
    groups = df[group_col].value_counts()
    print('warning, this is a paired ttest, it will use the 2 groups with most data:\n',
          groups.index[0], groups.index[1])
    rvs1 = df.loc[df[group_col] == groups.index[0], [value]]
    rvs2 = df.loc[df[group_col] == groups.index[1], [value]]
    res = (np.mean(rvs1) - np.mean(rvs2)) / (np.sqrt((np.stdev(rvs1) ** 2 + np.stdev(rvs2) ** 2) / 2))
    return res


def ttest_ind(df, value='score_combined', group_col='patient_group', equal_var=False):
    groups = df[group_col].value_counts()
    print('warning, this is a paired ttest, it will use the 2 groups with most data:\n',
          groups.index[0], groups.index[1])
    rvs1 = df.loc[df[group_col] == groups.index[0], [value]]
    rvs2 = df.loc[df[group_col] == groups.index[1], [value]]
    res = stats.ttest_ind(rvs1, rvs2, equal_var=equal_var)
    return res


def mannwhitneyu_ind(df, value='score_combined', group_col='patient_group'):
    groups = df[group_col].value_counts()
    print('warning, this is a paired, it will use the 2 groups with most data:\n',
          groups.index[0], groups.index[1])
    rvs1 = df.loc[df[group_col] == groups.index[0], [value]]
    rvs2 = df.loc[df[group_col] == groups.index[1], [value]]
    res = mannwhitneyu(rvs1, rvs2)
    return res


def ttest(df, value='score_combined',
          group_col='patient_group',
          groups_to_study=['organic only', 'smi+organic'],
          subgroup_col='age_bucket_baseline'):
    if group_col is not None: df = df[df[group_col].isin(groups_to_study)]
    subgroups = list(df[subgroup_col].unique())+['all'] if subgroup_col is not None else ['all']
    subgroups.sort()

    res = pd.DataFrame(columns=['t (ttest)', 'p (ttest)', 'sig (ttest)', 'stat (utest)', 'p (utest)', 'sig (utest)'])
    for sub in subgroups:
        df_tmp = df.loc[df[subgroup_col] == sub] if sub != 'all' else df
        g1 = df_tmp.loc[df_tmp[group_col] == groups_to_study[0], value]
        g2 = df_tmp.loc[df_tmp[group_col] == groups_to_study[1], value]
        t, p = stats.ttest_ind(g1, g2, equal_var=False)
        p_sig = p_value_sig(p)
        stat, pu = mannwhitneyu(g1, g2)
        pu_sig = p_value_sig(pu)
        res.loc[sub] = [t, p, p_sig, stat, pu, pu_sig]
        print('variable=', sub, 't (ttest)=', t, 'p (ttest)=', p)
    return res


def chisq(df, key='brcid',
          group_col='patient_group',
          groups_to_study=['organic only', 'SMI+organic'],
          subgroup_cols=default_subcols,
          complete_case=True):
    if group_col is not None: df = df[df[group_col].isin(groups_to_study)]

    res = pd.DataFrame()
    for sub in subgroup_cols:
        df_to_pivot = df.loc[(df[sub] != 'not known') & (df[sub] != 'unknown')] if complete_case else df
        df_tmp = df_to_pivot.pivot_table(values=key, index=sub, columns=group_col, aggfunc=pd.Series.nunique, margins=False, fill_value=0)
        chi2, p, dof, ex = stats.chi2_contingency(df_tmp)
        df_tmp['variable'] = sub
        df_tmp['chi2'] = chi2
        df_tmp['p'] = p
        df_tmp['dof'] = dof
        df_tmp['p_significance'] = p_value_sig(p)
        res = pd.concat([res, df_tmp])
    return res


def data_stats(df, key='brcid', value='score_combined', group_col='patient_group', subgroup_cols=default_subcols
               ):
    res = pd.DataFrame()
    for sub in subgroup_cols:
        df_countunique = df.pivot_table(values=key, index=sub, columns=group_col, aggfunc=pd.Series.nunique, margins=True, fill_value=0)
        df_countunique.columns = [x + '(1. N)' for x in df_countunique.columns]
        for col in df_countunique.columns:
            df_countunique[col.replace('N', '%')] = (df_countunique[col] / (df_countunique[col].sum()/2) * 100)
        df_count = df.pivot_table(values=key, index=sub, columns=group_col, aggfunc='count', margins=True, fill_value=0)
        df_count.columns = [x + '(2. obs)' for x in df_count.columns]
        df_mean = df.pivot_table(values=value, index=sub, columns=group_col, aggfunc=np.mean, margins=True)
        df_mean.columns = [x + '(3. M)' for x in df_mean.columns]
        df_sd = df.pivot_table(values=value, index=sub, columns=group_col, aggfunc=np.std, margins=True)
        df_sd.columns = [x + '(4. SD)' for x in df_sd.columns]
        res_tmp = pd.concat([df_mean, df_count, df_sd, df_countunique], axis=1)
        res_tmp = res_tmp.reindex(sorted(res_tmp.columns), axis=1)
        res_tmp['variable'] = sub
        res = pd.concat([res, res_tmp])

    return res
