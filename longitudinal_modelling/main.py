# https://www.statsmodels.org/stable/examples/notebooks/generated/mixed_lm_example.html

from longitudinal_modelling.longitudinal_utils import *
from longitudinal_modelling.data_stats import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.regression.mixed_linear_model as mlm


mmse_obj = LongitudinalDataset(
    data=r'/Users/aurelie/PycharmProjects/prometheus/longitudinal_modelling/trajectories_synthetic.xlsm'
    , sheet_name='data', target='score_new', timestamp='date', group='brcid', covariates=['age', 'med', 'gender'])
honos_obj = LongitudinalDataset(
    data=r'/Users/aurelie/PycharmProjects/prometheus/longitudinal_modelling/f20_all_classification_by_year_20200209.xlsx'
    , sheet_name='fake', target='honos_adjusted_total', timestamp='score_year', group='brcid'
    , covariates=['age_at_score_baseline', 'nlp_sc', 'Cognitive_Problems_Score_ID', 'diagnosis', 'education_level', 'gender'])
data_obj = LongitudinalDataset(
    data=r'/Users/aurelie/PycharmProjects/prometheus/longitudinal_modelling/f20_all_classification_by_year_20200209.xlsx'
    , sheet_name='traj', target='honos_adjusted_total', timestamp='score_year', group='brcid'
    , covariates=['age_at_score_baseline', 'nlp_sc_anytime', 'Cognitive_Problems_Score_ID_anytime'
        , 'diagnosis', 'education_level', 'gender', 'ethnicity', 'first_language', 'doa', 'marital_status'
        , 'antidementia_medication_baseline', 'antidepressant_medication_baseline', 'antipsychotic_medication_baseline'])

obj = data_obj
df = obj.load_data()
for col in ['antidementia_medication_baseline', 'antidepressant_medication_baseline', 'antipsychotic_medication_baseline']:
    df[col] = np.where(df[col] != 'no', 'yes', 'no')
for col in ['education_level', 'nlp_sc', 'Cognitive_Problems_Score_ID'
    , 'nlp_sc_baseline', 'Cognitive_Problems_Score_ID_baseline'
    , 'nlp_sc_anytime', 'Cognitive_Problems_Score_ID_anytime']:
    if col in df.columns: df[col] = np.where(pd.to_numeric(df[col]) > 0, 'yes', 'no')

# r_formula = 'score ~  date + age + diagnosis + gender + date * age + date * diagnosis + date * gender'
r_formula = make_smf_formula(target=obj.target, covariates=obj.covariates, timestamp=obj.timestamp)

# random intercept only
md = smf.mixedlm(r_formula, df, groups=df[obj.group])

# random intercept, and random slope (with respect to time)
md = smf.mixedlm(r_formula, df, groups=df[obj.group], re_formula='~'+obj.timestamp)

mdf = md.fit(method=['lbfgs'], reml=True)  # other methods lbfgs bfgs cg
print(mdf.summary())
(mdf.summary().tables[0]).to_clipboard()
(mdf.summary().tables[1]).to_clipboard()

# fit a model in which the two random effects are constrained to be uncorrelated:
md = smf.mixedlm(r_formula, df, groups=df['brcid'], re_formula='~date')
free = sm.regression.mixed_linear_model.MixedLMParams.from_components(np.ones(2), np.eye(2))

mdf = md.fit(free=free, method=['lbfgs'])
print(mdf.summary())
