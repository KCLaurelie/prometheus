from longitudinal_modelling.mlm import *
from longitudinal_modelling.regression import *

###############################################
## LOAD DATA
###############################################
xls = pd.ExcelFile('/Users/aurelie/PycharmProjects/prometheus/longitudinal_modelling/f20_traj_consolidated_20210318.xlsx', engine='openpyxl')
df_pop = pd.read_excel(xls, 'pop')
df_nlp_ci = pd.read_excel(xls, 'nlp_ci')
df_docs = pd.read_excel(xls, 'docs')
df_honos = pd.read_excel(xls, 'honos')
df_ward = pd.read_excel(xls, 'ward')
df_meds = pd.read_excel(xls, 'meds').fillna('no')
#   MERGE ALL PARTS TOGETHER
df = pd.merge(df_docs, df_nlp_ci,  how='outer', on=['brcid', 'year'], suffixes=[None, '_tomerge']).fillna(0)
df = pd.merge(df_pop, df,  how='inner', on=['brcid'], suffixes=[None, '_tomerge'])
df = pd.merge(df, df_honos,  how='outer', on=['brcid', 'year'], suffixes=[None, '_tomerge'])
df = pd.merge(df, df_ward,  how='left', on=['brcid', 'year'], suffixes=[None, '_tomerge'])
df['year_rounded'] = np.floor(df['score_year']).astype(int)
df = pd.merge(df, df_meds,  how='left', left_on=['brcid', 'year_rounded'], suffixes=[None, '_tomerge'])
# df = pd.merge(df_pop, df_docs,  how='inner', on=['brcid'], suffixes=[None, '_tomerge'])
# del df_pop, df_docs
#   CREATE BASELINE
df['baseline'] = 1*(df.brcid.diff() > 0)
df.at[0, 'baseline'] = 1
del df_pop, df_docs, df_nlp_ci, df_honos, df_ward, df_meds
for col in [col for col in df.columns if '_tomerge' in col]:
    df[col.replace('_tomerge','')].fillna(df[col], inplace=True)

## ADD NEW VARIABLES
df['diagnosis_schizo_only'] = np.where(df.has_dementia == 'dementia', 0, 1)
df = dummyfy(df, cols_to_dummyfy=['gender', 'diagnosis', 'ethnicity', 'employment', 'marital_status', 'education'], drop=False)
df['ethnicity_not_white'] = 1 - df['ethnicity_white']
df['education_above_gcse'] = 1 - df['education_no_education']
df['nlp_CI_adj'] = np.maximum(1, df['cognition'].fillna(0))+np.maximum(1, df['exec_function'].fillna(0))+np.maximum(1, df['memory'].fillna(0))
df['nlp_CI_adj_bool'] = np.minimum(1, df['nlp_CI_adj'])
cov_sociodem_minus = ['gender_male', 'diagnosis_schizo_dementia', 'education_no_education', 'ethnicity_not_white', 'marital_status_single_separated', 'employment_unemployed']
cov_sociodem_plus = ['gender_female', 'diagnosis_schizo_only', 'education_above_gcse', 'ethnicity_white', 'marital_status_married_cohabitating', 'employment_employed']
cov_honos = ['Daily_Living_Problems_Score_ID_absent', 'Relationship_Problems_Score_ID_absent', 'Hallucinations_Score_ID_absent', 'Depressed_Mood_Score_ID_absent']
cov_meds = ['antidementia', 'antidepressant', 'antipsychotic']
cov_scores = ['Cognitive_Problems_Score_ID', 'honos_adjusted_total', 'nlp_ci', 'attention', 'cognition', 'emotion', 'exec_function', 'memory', 'ward_len', 'num_ward_entries']
df['readmission'] = 1 * (df['num_ward_entries'] > 1)
for col in cov_meds:
    df[col + '_bool'] = np.where(df[col].str.lower().isin(['no', np.nan, 'null', '#n/a', 0]), 0, 1)
for col in cov_scores:
    df[col + '_bool'] = np.where(df[col].fillna(0) > 0, 1, 0)
for col in [col for col in df.columns if 'Score_ID' in col and 'bool' not in col]:
    df[col + '_absent'] = np.where(df[col].fillna(0) > 1, 0, 1)
df['age_at_score'] = np.maximum(10, df.year - df.dob.dt.year)
df['score_year_centered'] = 1 + np.where(df['baseline'] == 1, 0, df.year.diff())

###############################################
# TRAJECTORY ANALYSIS
###############################################
df = df.dropna(subset=['age_at_score']).copy()  # df.dropna(subset=cov_honos).copy()
df_sampleB = df.loc[df.sample_B == 'yes'].copy()
baseline_cols = ['brcid', 'antipsychotic', 'age_at_score']
df_baseline = df.loc[df.score_year_centered == 1][baseline_cols]
df = df.join(df_baseline.set_index('brcid'), on='brcid', rsuffix='_baseline')
df['fifty_older'] = np.where(df.age_at_score_baseline >= 50, 1, 0)

res = fit_mlm(df, group='brcid', target='nlp_ci', covariates=['fifty_older']+cov_sociodem_plus+['antipsychotic_baseline'], timestamp='score_year_centered', rdn_slope=True, method=['lbfgs'])
(res['stats']).to_clipboard(index=False, header=False)
(res['coeffs']).to_clipboard()
###############################################
# REGRESSION ANALYSIS
###############################################
