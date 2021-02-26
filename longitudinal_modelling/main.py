# https://www.statsmodels.org/stable/examples/notebooks/generated/mixed_lm_example.html
from longitudinal_modelling.mlm import *
from longitudinal_modelling.regression import *


mmse_obj = LongitudinalDataset(
    data=r'/Users/aurelie/PycharmProjects/prometheus/longitudinal_modelling/trajectories_synthetic.xlsm'
    , sheet_name='data', target='score_inv', timestamp='date', group='brcid', covariates=['age', 'med', 'gender'])
honos_obj = LongitudinalDataset(
    data=r'/Users/aurelie/PycharmProjects/prometheus/longitudinal_modelling/f20_all_classification_by_year_20210209.xlsx'
    , sheet_name='honos', target='honos_adjusted_total', timestamp='score_year_centered', group='brcid')
obj = honos_obj
df = obj.load_data()

# DERIVED FIELDS
df['ethnicity_white'] = np.where(df['ethnicity'] == 'white', 1, 0)
df['first_language_EN'] = np.where(df['first_language'] == 'not english', 0, 1)
df['married_or_cohab'] = np.where(df['marital_status'] == 'married_cohabitating', 1, 0)
df['employed_or_active'] = np.where(df['employment'].isin(['employed', 'retired_or_at_home', 'student_or_volunteer']), 1, 0)
df['nlp_sc_bool'] = np.where(df['nlp_sc'] > 0, 1, 0)
df['patient_in_ward'] = np.where(df['ward_len'] > 0, 1, 0)
df['patient_discharged'] = np.where(df['num_ward_discharges'] > 0, 1, 0)
df['ward_len_normalized'] = df['ward_len'] / df['ward_len'].max()
df['has_dementia'] = np.where(df['diagnosis'] == 'schizo+dementia', 1, 0)
df['gender_male'] = np.where(df['gender'] == 'male', 1, 0)

for col in ['antidementia_medication_baseline', 'antidepressant_medication_baseline', 'antipsychotic_medication_baseline'
    , 'antidementia_medication', 'antidepressant_medication', 'antipsychotic_medication']:
    df[col] = np.where(df[col].str.lower().isin(['no', np.nan, 'null', '#n/a', 0]), 0, 1)
# df_reg = df.loc[(df.honos_adjusted_total > 0) & (df.honos_adjusted_total <= 30)]
# df_reg = df.loc[(df.age_at_score >= 18) & (df.age_at_score <= 90)]

# TRAJECTORIES MODELLING
cov_mlm = ['Cognitive_Problems_Score_ID_baseline', 'ward_baseline'
    , 'nlp_sc_baseline'#, 'nlp_sc_before_honos_baseline'
    , 'has_dementia', 'education', 'gender_male', 'ethnicity_white', 'first_language_EN', 'married_or_cohab', 'employed_or_active'
    , 'antidementia_medication_baseline', 'antidepressant_medication_baseline', 'antipsychotic_medication_baseline']
res = fit_mlm(df, group=obj.group, target=obj.target, covariates=cov_mlm, timestamp=obj.timestamp, rdn_slope=True, method=['lbfgs'])
(res.tables[0]).to_clipboard(index=False, header=False)
(res.tables[1]).to_clipboard()

# REGRESSION
cov_reg = ['nlp_num_symptoms', 'Cognitive_Problems_Score_ID', 'Daily_Living_Problems_Score_ID', 'Hallucinations_Score_ID', 'Living_Conditions_Problems_Score_ID', 'Relationship_Problems_Score_ID'
    , 'has_dementia', 'education', 'gender_male', 'ethnicity_white', 'first_language_EN', 'married_or_cohab', 'employed_or_active'
    , 'antidementia_medication', 'antidepressant_medication', 'antipsychotic_medication']
res_reg = fit_reg(df, target=obj.target, covariates=cov_reg, timestamp='age_at_score', reg_type='ols', dummyfy_non_num=True, intercept=False, round_stats=5)
res_reg = fit_reg(df, target='ward_len', covariates=cov_reg, timestamp='age_at_score', reg_type='ols', dummyfy_non_num=True, intercept=False, round_stats=5)
res_reg = fit_reg(df, target='patient_in_ward', covariates=cov_reg, timestamp='age_at_score', reg_type='logit', dummyfy_non_num=True, intercept=False)
res_reg = fit_reg(df, target='patient_discharged', covariates=cov_reg, timestamp='age_at_score', reg_type='logit', dummyfy_non_num=True, intercept=False)
res_reg = fit_reg(df, target='nlp_num_symptoms', covariates=cov_reg, timestamp='age_at_score', reg_type='ols', dummyfy_non_num=True, intercept=False)


res_reg['model'].summary2().tables[0].append(res_reg['model'].summary2().tables[2]).to_clipboard(index=False, header=False)  # regression goodness of fit stats
res_reg['model'].summary2().tables[1].to_clipboard()  # coeffs

## MANUAL ANALYSIS
# r_formula = 'score ~  date + age + diagnosis + gender + date * age + date * diagnosis + date * gender'
r_formula = make_smf_formula(target=obj.target, covariates=cov_mlm, timestamp=obj.timestamp)
md = smf.mixedlm(r_formula, df, groups=df[obj.group], re_formula='~'+obj.timestamp)
mdf = md.fit(method=['lbfgs'], reml=True)


X = df[cov_reg + ['age_at_score']]
model = sm.OLS(df['ward_len'], df[['age_at_score', 'nlp_num_symptoms']]).fit()  # length of admission
model = sm.OLS(df['ward_len'], df[cov_reg].fillna(0)).fit()  # length of admission
model = sm.Logit(df['patient_in_ward'], df[['age_at_score', 'nlp_num_symptoms']]).fit()  # admission
model = sm.Logit(1 * (df['num_ward_entries'] > 1), df[['age_at_score', 'nlp_num_symptoms']]).fit()  # readmission

model = sm.GLM(np.exp(df['ward_len_normalized']), sm.add_constant(X), family=sm.families.Gamma(link=sm.families.links.log())).fit()
print(model.summary())
model.summary2().tables[0].append(model.summary2().tables[2]).to_clipboard(index=False, header=False)
model.summary2().tables[1].to_clipboard()  # coeffs



