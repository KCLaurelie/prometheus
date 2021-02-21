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

for col in ['antidementia_medication_baseline', 'antidepressant_medication_baseline', 'antipsychotic_medication_baseline'
    , 'antidementia_medication', 'antidepressant_medication', 'antipsychotic_medication'
    , 'education_level', 'ward_len_baseline'
    , 'nlp_sc_baseline', 'nlp_sc_baseline_cum', 'nlp_sc_before_honos'
    , 'Cognitive_Problems_Score_ID_baseline', 'Cognitive_Problems_Score_ID_before_honos']:
    try:
        if df[col].dtype == 'object':
            df[col] = np.where(df[col].str.lower().isin([np.nan, 'no', 'null', 'na', '#n/a']), 'no', 'yes')
        else:
            df[col] = np.where(pd.to_numeric(df[col]) > 0, 'yes', 'no')
    except:
        print('error for col', col)

cov = ['nlp_sc_baseline', 'Cognitive_Problems_Score_ID_baseline', 'ward_len_baseline'
        , 'diagnosis', 'education_level', 'gender', 'ethnicity', 'first_language', 'marital_status'
        , 'antidementia_medication_baseline', 'antidepressant_medication_baseline', 'antipsychotic_medication_baseline']
res = fit_mlm(df, group=obj.group, target=obj.target, covariates=cov, timestamp=obj.timestamp, rdn_slope=True, method=['lbfgs'])
(res.tables[0]).to_clipboard(index=False, header=False)
(res.tables[1]).to_clipboard()

cov_reg = ['nlp_sc', 'Cognitive_Problems_Score_ID', 'ward_len'
    , 'diagnosis', 'education_level', 'gender', 'ethnicity', 'first_language', 'marital_status'
    , 'antidementia_medication', 'antidepressant_medication', 'antipsychotic_medication']
res2 = fit_reg(df, target=obj.target, covariates=cov_reg, timestamp=obj.timestamp
               , reg_type=LogisticRegression(), dummyfy=True)

## MANUAL ANALYSIS
# r_formula = 'score ~  date + age + diagnosis + gender + date * age + date * diagnosis + date * gender'
r_formula = make_smf_formula(target=obj.target, covariates=cov, timestamp=obj.timestamp)
md = smf.mixedlm(r_formula, df, groups=df[obj.group], re_formula='~'+obj.timestamp)
mdf = md.fit(method=['lbfgs'], reml=True)




