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
df['ethnicity_white'] = np.where(df['ethnicity'] == 'white', 1, 0)
df['first_language_EN'] = np.where(df['first_language'] == 'english', 1, 0)
df['married_or_cohab'] = np.where(df['marital_status'] == 'married_cohabitating', 1, 0)
df['employed_or_active'] = np.where(df['employment'].isin(['employed', 'retired_or_at_home', 'student_or_volunteer']), 1, 0)
df['employed'] = np.where(df['employment'] == 'employed', 1, 0)
df['no_dementia'] = np.where(df['diagnosis'] == 'schizo+dementia', 0, 1)
df['gender_male'] = np.where(df['gender'] == 'male', 1, 0)
df['gender_female'] = np.where(df['gender'] == 'male', 0, 1)
df['age_at_score'] = np.minimum(df['age_at_score'], 10)
df['readmission'] = 1 * (df['num_ward_entries'] > 1)
df['ward_len_normalized'] = df['ward_len'] / df['ward_len'].max()
df['nlp_num_symptoms_adj'] = np.minimum(df['nlp_cognition'].fillna(0), 1) + np.minimum(df['nlp_exec_function'].fillna(0), 1) + np.minimum(df['nlp_memory'].fillna(0), 1)
cov_sociodem = ['gender_male', 'no_dementia', 'education', 'ethnicity_white', 'first_language_EN', 'married_or_cohab', 'employed']
cov_meds = ['antidementia_medication', 'antidepressant_medication', 'antipsychotic_medication']
cov_scores = ['Cognitive_Problems_Score_ID', 'Daily_Living_Problems_Score_ID', 'Hallucinations_Score_ID', 'Living_Conditions_Problems_Score_ID', 'Relationship_Problems_Score_ID', 'honos_adjusted_total', 'nlp_num_symptoms', 'nlp_attention', 'nlp_cognition', 'nlp_emotion', 'nlp_exec_function', 'nlp_memory', 'num_ward_entries', 'num_ward_discharges']
for col in cov_meds:
    df[col + '_bool'] = np.where(df[col].str.lower().isin(['no', np.nan, 'null', '#n/a', 0]), 0, 1)
for col in cov_scores:
    try: df[col + '_bool'] = np.minimum(df[col], 1) # np.where(df[col] >= 1, 1, 0)
    except: print(col, 'not in df')

# #####################################
# TRAJECTORIES MODELLING
baseline_cols = [x for x in df.columns if x in ['age_at_score'] + [col + '_bool' for col in cov_meds] + [col + '_bool' for col in cov_scores]]
df_baseline = df.loc[df.score_year_centered == 1][[obj.group] + baseline_cols]
df = df.join(df_baseline.set_index(obj.group), on=obj.group, rsuffix='_baseline')
# df_rest = df.loc[(df.age_at_score >= 40) & (df.age_at_score <= 90)]

cov_honos_baseline = [col for col in df.columns if 'Score_ID_bool_baseline' in col]
cov_meds_baseline = [col + '_bool_baseline' for col in cov_meds]
res = fit_mlm(df, group=obj.group, target=obj.target, covariates=['nlp_sc_bool_baseline']+cov_sociodem+cov_meds_baseline, timestamp=obj.timestamp, rdn_slope=True, method=['lbfgs'])
res = fit_mlm(df, group=obj.group, target='nlp_num_symptoms', covariates=['age_at_score']+cov_sociodem+cov_meds_baseline, timestamp=obj.timestamp, rdn_slope=True, method=['lbfgs'])
(res['stats']).to_clipboard(index=False, header=False)
(res['coeffs']).to_clipboard()

# #####################################
# REGRESSION
cov_honos = ['honos_adjusted_total_bool'] + [col for col in df.columns if ('Score_ID_bool' in col) and ('baseline' not in col)]
cov_nlp = ['nlp_attention_bool', 'nlp_cognition_bool', 'nlp_emotion_bool', 'nlp_exec_function_bool', 'nlp_memory_bool']
cov_reg = ['Cognitive_Problems_Score_ID'] + cov_sociodem + [col + '_bool' for col in cov_meds]
res_reg = fit_reg(df, target=obj.target, covariates=cov_reg, timestamp='age_at_score', reg_fn=sm.OLS, dummyfy_non_num=True, intercept=False, round_stats=5)
res_reg = fit_reg(df, target='ward_len', covariates=cov_reg, timestamp='age_at_score', reg_fn=sm.OLS, dummyfy_non_num=True, intercept=False, round_stats=5)
res_reg = fit_reg(df, target='num_ward_entries_bool', covariates=cov_reg, timestamp='age_at_score', reg_fn=sm.Logit, dummyfy_non_num=True, intercept=False)
res_reg = fit_reg(df, target='num_ward_discharges_bool', covariates=cov_reg, timestamp='age_at_score', reg_fn=sm.Logit, dummyfy_non_num=True, intercept=False)
res_reg = fit_reg(df, target='nlp_num_symptoms', covariates=cov_reg, timestamp='age_at_score', reg_fn=sm.OLS, dummyfy_non_num=True, intercept=False)

res_reg['stats'].to_clipboard(index=False, header=False)  # regression goodness of fit stats
res_reg['coeffs'].to_clipboard()  # coeffs

to_paste = pd.DataFrame()
target = 'ward_len'
reg_fn = sm.OLS if target == 'ward_len' else sm.Logit
for var in ['nlp_num_symptoms', 'nlp_num_symptoms_bool', 'Cognitive_Problems_Score_ID', 'Cognitive_Problems_Score_ID_bool']:
    res_reg = fit_reg(df, target=target, covariates=var, timestamp=None, reg_fn=reg_fn,dummyfy_non_num=True, intercept=False, round_stats=5)
    size = res_reg['coeffs'].shape[1]
    to_paste = to_paste.append(pd.DataFrame([size*[var]], columns=res_reg['coeffs'].columns))
    to_paste = to_paste.append(res_reg['coeffs']).append(pd.DataFrame([size*['']], columns=to_paste.columns))
    res_reg = fit_reg(df, target=target, covariates=var, timestamp='score_year_centered', reg_fn=reg_fn,dummyfy_non_num=True, intercept=False, round_stats=5)
    to_paste = to_paste.append(res_reg['coeffs']).append(pd.DataFrame([size*['']], columns=to_paste.columns))
    res_reg = fit_reg(df, target=target, covariates=[var]+ [cov_sociodem[0]], timestamp='score_year_centered', reg_fn=reg_fn,dummyfy_non_num=True, intercept=False, round_stats=5)
    to_paste = to_paste.append(res_reg['coeffs']).append(pd.DataFrame([size*['']], columns=to_paste.columns))
    res_reg = fit_reg(df, target=target, covariates=[var]+ cov_sociodem, timestamp='score_year_centered', reg_fn=reg_fn,dummyfy_non_num=True, intercept=False, round_stats=5)
    to_paste = to_paste.append(res_reg['coeffs']).append(pd.DataFrame([size*['']], columns=to_paste.columns))
    res_reg = fit_reg(df, target=target, covariates=[var]+ cov_sociodem + [col + '_bool' for col in cov_meds], timestamp='score_year_centered', reg_fn=reg_fn,dummyfy_non_num=True, intercept=False, round_stats=5)
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



