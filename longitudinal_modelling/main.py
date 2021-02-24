# https://www.statsmodels.org/stable/examples/notebooks/generated/mixed_lm_example.html
from longitudinal_modelling.mlm import *
from longitudinal_modelling.regression import *


mmse_obj = LongitudinalDataset(
    data=r'/Users/aurelie/PycharmProjects/prometheus/longitudinal_modelling/trajectories_synthetic.xlsm'
    , sheet_name='data', target='score_inv', timestamp='date', group='brcid', covariates=['age', 'med', 'gender'])
honos_obj = LongitudinalDataset(
    data=r'/Users/aurelie/PycharmProjects/prometheus/longitudinal_modelling/f20_all_classification_by_year_20200209.xlsx'
    , sheet_name='honos', target='honos_adjusted_total', timestamp='score_year_centered', group='brcid')
obj = honos_obj
df = obj.load_data()

# DERIVED FIELDS
df['ethnicity_bool'] = np.where(df['ethnicity'] == 'white', 'white', 'not_white')
df['first_language_bool'] = np.where(df['first_language'] == 'english', 'english', 'not_english')
df['marital_status_bool'] = np.where(df['marital_status'] == 'married_cohabitating', 'married', 'alone')
df['nlp_sc_bool'] = np.where(df['nlp_sc'] > 0, 1, 0)
df['patient_in_ward'] = np.where(df['ward_len'] > 0, 1, 0)
df['patient_discharged'] = np.where(df['num_ward_discharges'] > 0, 1, 0)
for col in ['antidementia_medication_baseline', 'antidepressant_medication_baseline', 'antipsychotic_medication_baseline'
    , 'antidementia_medication', 'antidepressant_medication', 'antipsychotic_medication']:
    df[col] = np.where(df[col].str.lower().isin(['no', np.nan, 'null', '#n/a']), 'no', 'yes')
df_reg = df.loc[(df.honos_adjusted_total > 0) & (df.honos_adjusted_total <= 30)]
df_reg = df.loc[(df.age_at_score >= 18) & (df.age_at_score <= 90)]

# TRAJECTORIES MODELLING
cov_mlm = ['Cognitive_Problems_Score_ID_baseline', 'ward_baseline'
    , 'nlp_sc_baseline'#, 'nlp_sc_before_honos_baseline'
    , 'diagnosis', 'education', 'gender', 'ethnicity_bool', 'first_language_bool', 'marital_status_bool'
    , 'antidementia_medication_baseline', 'antidepressant_medication_baseline', 'antipsychotic_medication_baseline']
res = fit_mlm(df, group=obj.group, target=obj.target, covariates=cov_mlm, timestamp=obj.timestamp, rdn_slope=True, method=['lbfgs'])
(res.tables[0]).to_clipboard(index=False, header=False)
(res.tables[1]).to_clipboard()

# REGRESSION
cov_reg = ['nlp_sc_bool', 'Cognitive_Problems_Score_ID'
     , 'num_ward_entries', 'num_ward_discharges'  # , 'patient_in_ward'
    , 'diagnosis', 'education', 'gender', 'ethnicity_bool', 'first_language_bool', 'marital_status_bool'
    , 'antidementia_medication', 'antidepressant_medication', 'antipsychotic_medication']
res_reg = fit_reg(df, target=obj.target, covariates=cov_reg, timestamp='age_at_score', reg_type='ols', dummyfy_non_num=True, intercept=False, round_stats=5)
res_reg = fit_reg(df, target='num_ward_entries', covariates=cov_reg, timestamp='age_at_score', reg_type='ols', dummyfy_non_num=True, intercept=False, round_stats=5)
res_reg = fit_reg(df, target='patient_in_ward', covariates=cov_reg, timestamp='age_at_score', reg_type='logit', dummyfy_non_num=True, intercept=False)
res_reg = fit_reg(df, target='patient_discharged', covariates=cov_reg, timestamp='age_at_score', reg_type='logit', dummyfy_non_num=True, intercept=False)
res_reg = fit_reg(df, target='nlp_num_symptoms', covariates=cov_reg, timestamp='age_at_score', reg_type='ols', dummyfy_non_num=True, intercept=False)


(res_reg['model'].summary2().tables[0]).to_clipboard(index=False, header=False)
(res_reg['model'].summary2().tables[1]).to_clipboard()

## MANUAL ANALYSIS
# r_formula = 'score ~  date + age + diagnosis + gender + date * age + date * diagnosis + date * gender'
r_formula = make_smf_formula(target=obj.target, covariates=cov_mlm, timestamp=obj.timestamp)
md = smf.mixedlm(r_formula, df, groups=df[obj.group], re_formula='~'+obj.timestamp)
mdf = md.fit(method=['lbfgs'], reml=True)




