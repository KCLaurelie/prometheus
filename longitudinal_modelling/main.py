# https://www.statsmodels.org/stable/examples/notebooks/generated/mixed_lm_example.html
from longitudinal_modelling.longitudinal_utils import *
from longitudinal_modelling.data_stats import *
from longitudinal_modelling.mlm import *
from longitudinal_modelling.regression import *


mmse_obj = LongitudinalDataset(
    data=r'/Users/aurelie/PycharmProjects/prometheus/longitudinal_modelling/trajectories_synthetic.xlsm'
    , sheet_name='data', target='score_inv', timestamp='date', group='brcid', covariates=['age', 'med', 'gender'])
honos_obj = LongitudinalDataset(
    data=r'/Users/aurelie/PycharmProjects/prometheus/longitudinal_modelling/f20_all_classification_by_year_20200209.xlsx'
    , sheet_name='traj', target='honos_adjusted_total', timestamp='score_year_centered', group='brcid')

obj = honos_obj
df = obj.load_data()
for col in ['antidementia_medication_baseline', 'antidepressant_medication_baseline', 'antipsychotic_medication_baseline'
    , 'education_level', 'ward_total_len', 'nlp_sc', 'nlp_sc_baseline', 'nlp_sc_baseline_cum', 'nlp_sc_anytime'
    , 'Cognitive_Problems_Score_ID', 'Cognitive_Problems_Score_ID_baseline', 'Cognitive_Problems_Score_ID_anytime']:
    if df[col].dtype == 'object':
        df[col] = np.where(df[col] != 'no', 'yes', 'no')
    else:
        df[col] = np.where(pd.to_numeric(df[col]) > 0, 'yes', 'no')

cov = ['age_at_score_baseline', 'nlp_sc_baseline', 'Cognitive_Problems_Score_ID_baseline', 'ward_total_len'
        , 'diagnosis', 'education_level', 'gender', 'ethnicity', 'first_language', 'doa', 'marital_status'
        , 'antidementia_medication_baseline', 'antidepressant_medication_baseline', 'antipsychotic_medication_baseline']
res = fit_mlm(df, group=obj.group, target=obj.target, covariates=cov, timestamp=obj.timestamp, rdn_slope=True, method=['lbfgs'])
(res.tables[0]).to_clipboard(index=False, header=False)
(res.tables[1]).to_clipboard()

## MANUAL ANALYSIS
# r_formula = 'score ~  date + age + diagnosis + gender + date * age + date * diagnosis + date * gender'
r_formula = make_smf_formula(target=obj.target, covariates=cov, timestamp=obj.timestamp)
md = smf.mixedlm(r_formula, df, groups=df[obj.group], re_formula='~'+obj.timestamp)
mdf = md.fit(method=['lbfgs'], reml=True)




