from longitudinal_modelling.mlm import *
from longitudinal_modelling.regression import *

###############################################
## LOAD DATA
###############################################
try:
    xls = pd.ExcelFile('/Users/aurelie/PycharmProjects/prometheus/longitudinal_modelling/cris_traj_20210322.xlsx', engine='openpyxl')
except:
    xls = pd.ExcelFile('/Users/k1774755/PycharmProjects/prometheus/longitudinal_modelling/cris_traj_20210322.xlsx', engine='openpyxl')

# pre-built Sample C
df = pd.read_excel(xls, 'traj')
diag_cols = ['abuse_neglect','adhd','dementia','depressive','eating','learning','mania_bipolar','mood_other','nervous_syst','other_organic','personality','psychotic','sexual','sleep','stress','substance_abuse']

# to rebuild samples
include_unknown = True # include patients with unknown diagnosis
df_nlp_ci = pd.read_excel(xls, 'nlp_ci').dropna(subset=['score_year'])
df_pop = pd.read_excel(xls, 'pop')
df_diag = pd.read_excel(xls, 'diag')
df_honos = pd.read_excel(xls, 'honos')
df_ward = pd.read_excel(xls, 'ward')

df_diag = pd.pivot_table(df_diag, values='diag_date', index='brcid', columns='diag', aggfunc=np.max, fill_value=0)
diag_cols = list(df_diag.columns)
if include_unknown:
    df = pd.merge(df_pop, df_diag, how='left', left_on=['brcid'], right_index=True, suffixes=[None, '_tomerge'])
    df['diag_unknown'] = 1 - np.minimum(1, df[diag_cols].sum(axis=1))
    if len(df['diag_unknown'].unique()) > 1: diag_cols += ['diag_unknown']
else:
    df = pd.merge(df_pop, df_diag, how='inner', left_on=['brcid'], right_index=True, suffixes=[None, '_tomerge'])
del df_pop, df_diag

###############################################
# CLEAN SOCIODEM DATA AND ADD NEW VARIABLES
###############################################
df = dummyfy(df, cols_to_dummyfy=['gender', 'ethnicity', 'employment', 'marital_status', 'education_level', 'housing_status'], drop=False)
df['ethnicity_not_white'] = 1 - df['ethnicity_white']
df['education_above_gcse'] = 1 - df['education_level_no_education']
df['not_homeless'] = 1 - df['housing_status_homeless_or_not_fixed']
cov_sociodem_minus = ['gender_male', 'education_level_no_education', 'ethnicity_not_white', 'marital_status_single_separated', 'employment_unemployed', 'housing_status_homeless_or_not_fixed']
cov_sociodem_plus = ['gender_female', 'education_above_gcse', 'ethnicity_white', 'marital_status_married_cohabitating', 'employment_employed', 'not_homeless']
cov_sociodem_plus_nlp = ['education_level_no_education' if x == 'education_above_gcse' else x for x in cov_sociodem_plus]
for col in diag_cols:
    if col in df.columns:
        df[col] = np.where(df[col].fillna(0) > 0, 1, 0)
        df[col+'_no'] = 1 - df[col]
diag_no = [x+'_no' for x in diag_cols]

#  MERGING WITH HONOS AND WARD TABS  ###############################
df_honos = pd.merge(df_nlp_ci, df_honos,  how='right', on=['brcid', 'score_year'], suffixes=[None, '_tomerge'])
df_honos = pd.merge(df, df_honos,  how='inner', on=['brcid'], suffixes=[None, '_tomerge'])

df_ward = pd.merge(df_nlp_ci, df_ward,  how='right', left_on=['brcid', 'score_year'], right_on=['brcid', 'ward_start_year'], suffixes=[None, '_tomerge'])
df_ward = pd.merge(df, df_ward,  how='inner', on=['brcid'], suffixes=[None, '_tomerge'])
df_ward['readmission'] = 1 * (df_ward['num_ward_entries'] > 1)

df_nlp_ci = pd.merge(df, df_nlp_ci,  how='inner', on=['brcid'], suffixes=[None, '_tomerge'])
#####################################################################

# clean age values and drop missing
del df
df = df_nlp_ci # ***** which df to use ******
if 'age_at_score' not in df.columns: df['age_at_score'] = np.maximum(10, df.score_year - df.dob.dt.year.fillna(1900))
df = df.loc[(df.score_year >= 2000) & (df.score_year < 2021)].copy()
df = df.dropna(subset=['age_at_score']).copy()

# Dummyfy and standardize data
cov_honos = ['Daily_Living_Problems_Score_ID_absent', 'Relationship_Problems_Score_ID_absent', 'Hallucinations_Score_ID_absent', 'Depressed_Mood_Score_ID_absent']
cov_scores = ['Cognitive_Problems_Score_ID', 'honos_adjusted_total', 'nlp_ci', 'attention', 'cognition', 'emotion', 'exec_function', 'memory', 'ward_len', 'num_ward_entries']
# df['nlp_CI_adj'] = np.maximum(1, df['cognition'].fillna(0))+np.maximum(1, df['exec_function'].fillna(0))+np.maximum(1, df['memory'].fillna(0))
df['nlp_ci'] = df['nlp_ci'].fillna(0)
for col in cov_scores:
    if col in df.columns: df[col + '_bool'] = np.where(df[col].fillna(0) > 0, 1, 0)
for col in [col for col in df.columns if 'Score_ID' in col and 'bool' not in col]:
    df[col + '_absent'] = np.where(df[col].fillna(0) > 1, 0, 1)

#   CREATE BASELINE
df = df.sort_values(by=['brcid', 'score_year'], ascending=[True, True]).reset_index()
df['counter'] = df.groupby(['brcid']).cumcount()+1
df_baseline = df.loc[df.counter == 1][['brcid', 'age_at_score']]  # 'antipsychotic']
df = df.join(df_baseline.set_index('brcid'), on='brcid', rsuffix='_baseline')
df['age_centered'] = 1 + df.age_at_score - df.age_at_score_baseline
df['age_rounded'] = np.floor(df.age_at_score / 10) * 10
df['fifty_older'] = np.where(df.age_at_score_baseline >= 50, 1, 0)
df['fifty_younger'] = 1 - df['fifty_older']

###############################################
# TRAJECTORY ANALYSIS
###############################################
traj_brcid = df.groupby('brcid', as_index=False).size()
df_mlm = pd.merge(df, traj_brcid.loc[traj_brcid['size'] >= 3],  how='inner', on=['brcid'], suffixes=[None, '_tomerge'])
res = fit_mlm(df_mlm, group='brcid', target='nlp_ci', covariates=cov_sociodem_plus+diag_no, timestamp='age_centered', rdn_slope=True, method=['lbfgs'])
res = fit_mlm(df_mlm, group='brcid', target='nlp_ci', covariates=['age_rounded']+cov_sociodem_plus+diag_cols, timestamp='counter', rdn_slope=True, method=['lbfgs'])
(res['stats']).to_clipboard(index=False, header=False)
(res['coeffs']).to_clipboard()


def run_all_diag_traj(df, target='nlp_ci', covariates=cov_sociodem_plus, timestamp='age_centered', rdn_slope=True):
    to_paste = pd.DataFrame()
    for col in diag_cols:
        df_tmp = df.loc[df[col] == 1]
        print('************', col, '\n')
        title = col + '_' + str(len(df_tmp))
        try:
            res = fit_mlm(df_tmp, group='brcid', target=target, covariates=covariates, timestamp=timestamp, rdn_slope=rdn_slope, method=['lbfgs'])['coeffs']
            to_paste = to_paste.append(pd.DataFrame([[title]], columns=[res.columns[0]])).append(res)
        except:
            pass
    return to_paste

###############################################
# REGRESSION ANALYSIS
###############################################
target='Cognitive_Problems_Score_ID_bool'
target='nlp_ci'
res_reg = fit_reg(df, target=target, covariates=cov_sociodem_plus+diag_no, timestamp='age_centered', reg_fn=sm.OLS, dummyfy_non_num=True, intercept=False)
res_reg['stats'].to_clipboard(index=False, header=False)
res_reg['coeffs'].to_clipboard()


def run_all_diag_reg(df, target='ward_len', score='nlp_ci_bool', covariates='cov_sociodem_plus_nlp', timestamp='age_centered', intercept=False):
    to_paste = pd.DataFrame()
    reg_fn = sm.OLS if df[target].max() > 1 else sm.Logit
    for col in diag_cols:
        df_tmp = df.loc[df[col] == 1]
        print('************', col, len(df_tmp), ' patients\n')
        title = col + '_' + str(len(df_tmp))
        try:
            res = fit_reg(df_tmp, target=target, covariates=[score], timestamp=None, reg_fn=reg_fn, dummyfy_non_num=True, intercept=intercept)['coeffs']
            to_paste = to_paste.append(pd.DataFrame([[title]], columns=[res.columns[0]])).append(res)
            to_paste = to_paste.append(fit_reg(df_tmp, target=target, covariates=[score], timestamp=timestamp, reg_fn=reg_fn,dummyfy_non_num=True, intercept=intercept)['coeffs'])
            to_paste = to_paste.append(fit_reg(df_tmp, target=target, covariates=[score] + covariates, timestamp=timestamp, reg_fn=reg_fn,dummyfy_non_num=True, intercept=intercept)['coeffs'])
        except:
            pass
    return to_paste


res = run_all_diag_reg(df, target='ward_len', score='nlp_ci', covariates=cov_sociodem_plus, timestamp='age_centered', intercept=False)
res = run_all_diag_reg(df, target='num_ward_entries', score='nlp_ci', covariates=cov_sociodem_plus, timestamp='age_centered', intercept=False)
res[res.index.isin(['nlp_ci', 0])].to_clipboard()