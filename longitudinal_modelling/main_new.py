from longitudinal_modelling.mlm import *
from longitudinal_modelling.regression import *

###############################################
## LOAD DATA
###############################################
try:
    xls = pd.ExcelFile('/Users/aurelie/PycharmProjects/prometheus/longitudinal_modelling/cris_traj_20210322.xlsx', engine='openpyxl')
except:
    xls = pd.ExcelFile('/Users/k1774755/PycharmProjects/prometheus/longitudinal_modelling/cris_traj_20210322.xlsx', engine='openpyxl')

df_nlp_ci = pd.read_excel(xls, 'nlp_ci').dropna(subset=['score_year'])
df_pop = pd.read_excel(xls, 'pop')
df_diag = pd.read_excel(xls, 'diag')
df_honos = pd.read_excel(xls, 'honos')
df_ward = pd.read_excel(xls, 'ward')
df_docs = pd.read_excel(xls, 'f20_docs')
df_meds = pd.read_excel(xls, 'f20_meds').fillna('no')
#   REMOVE PATIENTS WITHOUT STATIC DATA
df_diag = pd.pivot_table(df_diag, values='diag_date', index='brcid', columns='diag', aggfunc=np.max, fill_value=0)
diag_cols = list(df_diag.columns)+['had_schizo']
df_pop = pd.merge(df_pop, df_diag, how='inner', left_on=['brcid'], right_index=True, suffixes=[None, '_tomerge'])
df = pd.merge(df_pop['brcid'], df_nlp_ci,  how='inner', on=['brcid'], suffixes=[None, '_tomerge'])
# to count absence of doc as absence of symptom
df_nlp_ci = pd.merge(df_docs, df_nlp_ci,  how='outer', on=['brcid', 'year'], suffixes=[None, '_tomerge']).fillna(0)
# merge the rest
df = pd.merge(df, df_honos,  how='right', on=['brcid', 'score_year'], suffixes=[None, '_tomerge'])
df = pd.merge(df, df_ward,  how='left', left_on=['brcid', 'score_year'], right_on=['brcid', 'ward_start_year'], suffixes=[None, '_tomerge'])
df['year_rounded'] = np.floor(df['score_year']).astype(int)
df = pd.merge(df, df_meds,  how='left', left_on=['brcid', 'year_rounded'], right_on=['brcid', 'year'], suffixes=[None, '_tomerge'])
df = pd.merge(df_pop, df,  how='inner', on=['brcid'], suffixes=[None, '_tomerge'])

del df_pop, df_nlp_ci, df_diag, df_docs, df_honos, df_ward, df_meds


## CLEAN AND ADD NEW VARIABLES
df['nlp_ci'] = df['nlp_ci'].fillna(0)
df = dummyfy(df, cols_to_dummyfy=['gender', 'ethnicity', 'employment', 'marital_status', 'education_level', 'housing_status'], drop=False)
df['ethnicity_not_white'] = 1 - df['ethnicity_white']
df['education_above_gcse'] = 1 - df['education_level_no_education']
df['not_homeless'] = 1 - df['housing_status_homeless_or_not_fixed']
df['nlp_CI_adj'] = np.maximum(1, df['cognition'].fillna(0))+np.maximum(1, df['exec_function'].fillna(0))+np.maximum(1, df['memory'].fillna(0))
df['nlp_CI_adj_bool'] = np.minimum(1, df['nlp_CI_adj'])
cov_sociodem_minus = ['gender_male', 'education_level_no_education', 'ethnicity_not_white', 'marital_status_single_separated', 'employment_unemployed', 'housing_status_homeless_or_not_fixed']
cov_sociodem_plus = ['gender_female', 'education_above_gcse', 'ethnicity_white', 'marital_status_married_cohabitating', 'employment_employed', 'not_homeless']
cov_honos = ['Daily_Living_Problems_Score_ID_absent', 'Relationship_Problems_Score_ID_absent', 'Hallucinations_Score_ID_absent', 'Depressed_Mood_Score_ID_absent']
cov_meds = ['antidementia', 'antidepressant', 'antipsychotic']
cov_scores = ['Cognitive_Problems_Score_ID', 'honos_adjusted_total', 'nlp_ci', 'attention', 'cognition', 'emotion', 'exec_function', 'memory', 'ward_len', 'num_ward_entries']
df['readmission'] = 1 * (df['num_ward_entries'] > 1)
for col in cov_meds:
    df[col + '_bool'] = np.where(df[col].str.lower().isin(['no', np.nan, 'null', '#n/a', 0]), 0, 1)
for col in cov_scores:
    if col in df.columns: df[col + '_bool'] = np.where(df[col].fillna(0) > 0, 1, 0)
for col in diag_cols:
    if col in df.columns: df[col] = np.where(df[col].fillna(0) > 0, 1, 0)
for col in [col for col in df.columns if 'Score_ID' in col and 'bool' not in col]:
    df[col + '_absent'] = np.where(df[col].fillna(0) > 1, 0, 1)

# clean age values and drop missing
df['age_at_score'] = np.maximum(10, df.score_year - df.dob.dt.year.fillna(1900))
df = df.loc[(df.score_year >= 2000) & (df.score_year < 2021)].copy()
df = df.dropna(subset=['age_at_score']).copy()
#   CREATE BASELINE
df = df.sort_values(by=['brcid', 'score_year'], ascending=[True, True]).reset_index()
df['counter'] = df.groupby(['brcid']).cumcount()+1
print('score at baseline', df.loc[df.counter == 1]['nlp_ci'].mean())

df_baseline = df.loc[df.counter == 1][['brcid', 'age_at_score', 'age_bucket']]  # 'antipsychotic']
df = df.join(df_baseline.set_index('brcid'), on='brcid', rsuffix='_baseline')
df['score_centered'] = 1 + df.age_at_score - df.age_at_score_baseline
df['age_rounded'] = np.floor(df.age_at_score / 10) * 10

df['fifty_older'] = np.where(df.age_at_score_baseline >= 50, 1, 0)


###############################################
# TRAJECTORY ANALYSIS
###############################################
# df_sampleB = df.loc[df.sample_B == 'yes'].copy()
traj_brcid = df.groupby('brcid', as_index=False).size()
df_mlm = pd.merge(df, traj_brcid.loc[traj_brcid['size'] >= 3],  how='inner', on=['brcid'], suffixes=[None, '_tomerge'])


res = fit_mlm(df_mlm, group='brcid', target='nlp_ci', covariates=['fifty_older']+cov_sociodem_plus+diag_cols, timestamp='score_centered', rdn_slope=True, method=['lbfgs'])
res = fit_mlm(df_mlm, group='brcid', target='nlp_ci', covariates=['age_rounded']+cov_sociodem_plus+diag_cols, timestamp='counter', rdn_slope=True, method=['lbfgs'])
(res['stats']).to_clipboard(index=False, header=False)
(res['coeffs']).to_clipboard()


def run_all_diag():
    to_paste = pd.DataFrame()
    for col in diag_cols:
        df_tmp = df_mlm.loc[df_mlm[col]==1]
        print('************', col, '\n')
        title = pd.DataFrame([df_tmp.shape[1] * [col + '_' + str(len(df_tmp))]], columns=df_tmp.columns)
        try:
            res = fit_mlm(df_tmp, group='brcid', target='nlp_ci', covariates=['age_rounded']+cov_sociodem_plus, timestamp='counter', rdn_slope=True, method=['lbfgs'])['coeffs']
            to_paste = to_paste.append(title).append(res)
        except:
            to_paste = to_paste.append(title)


###############################################
# REGRESSION ANALYSIS
###############################################
res_reg = fit_reg(df, target='nlp_ci', covariates=cov_sociodem_plus+diag_cols, timestamp='age_at_score', reg_fn=sm.OLS, dummyfy_non_num=True, intercept=False)
res_reg['stats'].to_clipboard(index=False, header=False)
res_reg['coeffs'].to_clipboard()

