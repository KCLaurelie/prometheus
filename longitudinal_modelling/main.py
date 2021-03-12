# https://www.statsmodels.org/stable/examples/notebooks/generated/mixed_lm_example.html
from longitudinal_modelling.mlm import *
from longitudinal_modelling.regression import *


mmse_obj = LongitudinalDataset(
    data=r'/Users/aurelie/PycharmProjects/prometheus/longitudinal_modelling/trajectories_synthetic.xlsm'
    , sheet_name='data', target='score_inv', timestamp='date', group='brcid', covariates=['age', 'med', 'gender'])
nlp_ci = LongitudinalDataset(
    data=r'/Users/aurelie/PycharmProjects/prometheus/longitudinal_modelling/nlp_ci_traj_20200303.xlsx'
    , sheet_name='traj', target='nlp_num_symptoms', timestamp='score_year_centered', group='brcid')
honos_obj = LongitudinalDataset(
    data=r'/Users/aurelie/PycharmProjects/prometheus/longitudinal_modelling/f20_all_classification_by_year_20210209.xlsx'
    , sheet_name='honos', target='honos_adjusted_total', timestamp='score_year_centered', group='brcid')
obj = nlp_ci
obj = honos_obj
df = obj.load_data()

# DERIVED FIELDS
df = obj.dummyfy(cols_to_dummyfy=['gender', 'diagnosis', 'ethnicity', 'employment', 'marital_status', 'education'], drop=False)
df['ethnicity_not_white'] = 1 - df['ethnicity_white']
df['education_above_gcse'] = 1 - df['education_no_education']
cov_sociodem = ['gender_male', 'diagnosis_schizo_dementia', 'education_no_education', 'ethnicity_not_white', 'marital_status_single_separated', 'employment_unemployed']
cov_sociodem_plus = ['gender_female', 'diagnosis_schizo_only', 'education_above_gcse', 'ethnicity_white', 'marital_status_married_cohabitating', 'employment_employed']
cov_meds = ['antidementia_medication', 'antidepressant_medication', 'antipsychotic_medication']
cov_scores = ['Cognitive_Problems_Score_ID', 'honos_adjusted_total', 'nlp_num_symptoms', 'nlp_attention', 'nlp_cognition', 'nlp_emotion', 'nlp_exec_function', 'nlp_memory', 'ward_len', 'num_ward_entries', 'num_ward_discharges']
df['age_at_score'] = np.maximum(df['age_at_score'], 10)
df['age_rounded'] = np.floor(df.age_at_score / 10) * 10
df['readmission'] = 1 * (df['num_ward_entries'] > 1)

for col in cov_meds:
    if col in df.columns: df[col + '_bool'] = np.where(df[col].str.lower().isin(['no', np.nan, 'null', '#n/a', 0]), 0, 1)
for col in cov_scores:
    if col in df.columns: df[col + '_bool'] = np.minimum(df[col], 1)

# #####################################
# TRAJECTORIES MODELLING
baseline_cols = [x for x in df.columns if x in ['age_at_score', 'age_bucket'] + [col + '_bool' for col in cov_meds] + [col + '_bool' for col in cov_scores]]
df_baseline = df.loc[df.score_year_centered == 1][[obj.group] + baseline_cols]
df = df.join(df_baseline.set_index(obj.group), on=obj.group, rsuffix='_baseline')
cov_meds_baseline = [col + '_bool_baseline' for col in cov_meds]

res = fit_mlm(df, group=obj.group, target='Cognitive_Problems_Score_ID', covariates=cov_sociodem_plus+['antipsychotic_medication_bool_baseline'], timestamp=obj.timestamp, rdn_slope=True, method=['lbfgs'])
res = fit_mlm(df, group=obj.group, target='nlp_num_symptoms', covariates=cov_sociodem_plus+['antipsychotic_medication_bool_baseline'], timestamp=obj.timestamp, rdn_slope=True, method=['lbfgs'])
(res['stats']).to_clipboard(index=False, header=False)
(res['coeffs']).to_clipboard()

# #####################################
# REGRESSION
cov_reg = cov_sociodem_plus + ['antipsychotic_medication_bool_baseline']
res_reg = fit_reg(df, target=obj.target, covariates=cov_reg, timestamp='age_at_score', reg_fn=sm.OLS, dummyfy_non_num=True, intercept=False, round_stats=5)
res_reg = fit_reg(df, target='ward_len', covariates=cov_reg, timestamp='age_at_score', reg_fn=sm.OLS, dummyfy_non_num=True, intercept=False, round_stats=5)
res_reg = fit_reg(df, target='num_ward_entries_bool', covariates=cov_reg, timestamp='age_at_score', reg_fn=sm.Logit, dummyfy_non_num=True, intercept=False)
res_reg = fit_reg(df, target='num_ward_discharges_bool', covariates=cov_reg, timestamp='age_at_score', reg_fn=sm.Logit, dummyfy_non_num=True, intercept=False)
res_reg = fit_reg(df, target='nlp_num_symptoms', covariates=cov_reg, timestamp='age_at_score', reg_fn=sm.OLS, dummyfy_non_num=True, intercept=False)
res_reg = fit_reg(df, target='nlp_num_symptoms_bool', covariates=['age_bucket']+cov_reg, timestamp=None, reg_fn=sm.Logit, dummyfy_non_num=True, intercept=False)

res_reg['stats'].to_clipboard(index=False, header=False)  # regression goodness of fit stats
res_reg['coeffs'].to_clipboard()  # coeffs

target, timestamp, intercept = ['num_ward_entries_bool', 'age_at_score', True]
target, timestamp, intercept = ['num_ward_entries', 'age_at_score', False]
reg_fn = sm.OLS if df[target].max() > 1 else sm.Logit
for var in ['nlp_num_symptoms', 'nlp_num_symptoms_bool', 'Cognitive_Problems_Score_ID', 'Cognitive_Problems_Score_ID_bool']:
    if var == var[0]: to_paste = pd.DataFrame()
    res_reg = fit_reg(df, target=target, covariates=var, timestamp=None, reg_fn=reg_fn,dummyfy_non_num=True, intercept=intercept, round_stats=5)
    size = res_reg['coeffs'].shape[1]
    to_paste = to_paste.append(pd.DataFrame([size*[var]], columns=res_reg['coeffs'].columns))
    to_paste = to_paste.append(res_reg['coeffs']).append(pd.DataFrame([size*['']], columns=to_paste.columns))
    res_reg = fit_reg(df, target=target, covariates=var, timestamp=timestamp, reg_fn=reg_fn,dummyfy_non_num=True, intercept=intercept, round_stats=5)
    to_paste = to_paste.append(res_reg['coeffs']).append(pd.DataFrame([size*['']], columns=to_paste.columns))
    res_reg = fit_reg(df, target=target, covariates=[var]+ [cov_sociodem_plus[0]], timestamp=timestamp, reg_fn=reg_fn,dummyfy_non_num=True, intercept=intercept, round_stats=5)
    to_paste = to_paste.append(res_reg['coeffs']).append(pd.DataFrame([size*['']], columns=to_paste.columns))
    res_reg = fit_reg(df, target=target, covariates=[var]+ cov_sociodem_plus, timestamp=timestamp, reg_fn=reg_fn,dummyfy_non_num=True, intercept=intercept, round_stats=5)
    to_paste = to_paste.append(res_reg['coeffs']).append(pd.DataFrame([size*['']], columns=to_paste.columns))
    res_reg = fit_reg(df, target=target, covariates=[var]+ cov_sociodem_plus + ['antipsychotic_medication_bool_baseline'], timestamp=timestamp, reg_fn=reg_fn,dummyfy_non_num=True, intercept=intercept, round_stats=5)
    to_paste = to_paste.append(res_reg['coeffs']).append(pd.DataFrame([size*['']], columns=to_paste.columns))
to_paste.to_clipboard()

# #####################################
# MANUAL ANALYSIS
# r_formula = 'score ~  date + age + diagnosis + gender + date * age + date * diagnosis + date * gender'
df = pd.read_excel(r'/Users/aurelie/PycharmProjects/prometheus/longitudinal_modelling/nlp_ci_traj_20200303.xlsx', sheet_name='traj', engine='openpyxl')
res = fit_mlm(df, group='brcid', target='nlp_num_symptoms', covariates=cov_sociodem, timestamp=obj.timestamp, rdn_slope=True, method=['lbfgs'])

r_formula = make_smf_formula(target=obj.target, covariates=cov_reg, timestamp=obj.timestamp)
md = smf.mixedlm(r_formula, df, groups=df[obj.group], re_formula='~'+obj.timestamp)
mdf = md.fit(method=['lbfgs'], reml=True)

X = df[cov_reg + ['age_at_score']]
model = sm.Logit(1 * (df['num_ward_entries'] > 1), df[['age_at_score', 'nlp_num_symptoms']]).fit()  # readmission
model = sm.GLM(np.exp(df['ward_len_normalized']), sm.add_constant(X), family=sm.families.Gamma(link=sm.families.links.log())).fit()
print(model.summary())
model.summary2().tables[0].append(model.summary2().tables[2]).to_clipboard(index=False, header=False)
model.summary2().tables[1].to_clipboard()  # coeffs



