# https://www.statsmodels.org/stable/examples/notebooks/generated/mixed_lm_example.html

from longitudinal_modelling.longitudinal_utils import *
from longitudinal_modelling.data_stats import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.regression.mixed_linear_model as mlm

data_obj = LongitudinalDataset(
    data=r'/Users/aurelie/PycharmProjects/prometheus/longitudinal_modelling/trajectories_synthetic.xlsm',
    sheet_name='data',
    target='score',
    group='brcid',
    covariates=['diagnosis', 'date', 'score', 'gender', 'med'])
data_obj2 = LongitudinalDataset(
    data=r'/Users/aurelie/PycharmProjects/prometheus/longitudinal_modelling/f20_all_classification_by_year.xlsx'
    , sheet_name='honos'
    , target='honos_adjusted_total'
    , timestamp='score_year'
    , group='brcid'
    , covariates=['age_at_score_baseline', 'nlp_sc', 'Cognitive_Problems_Score_ID'
        , 'diagnosis', 'education_level', 'gender', 'ethnicity', 'first_language', 'doa', 'marital_status'
        , 'antidementia_medication_baseline', 'antidepressant_medication_baseline', 'antipsychotic_medication_baseline']
)
df = data_obj2.load_data()
for col in ['antidementia_medication_baseline', 'antidepressant_medication_baseline',
            'antipsychotic_medication_baseline']:
    df[col] = np.where(df[col] != 'no', 'yes', 'no')
for col in ['nlp_sc', 'Cognitive_Problems_Score_ID', 'education_level']:
    try:
        df[col] = np.where(pd.to_numeric(df[col]) > 0, 'yes', 'no')
    except:
        print('could no convert', col, type(df[col]))
# r_formula = 'score ~  date + age + diagnosis + gender + date * age + date * diagnosis + date * gender'
r_formula = make_smf_formula(target=data_obj2.target
                             , covariates=data_obj2.covariates
                             , timestamp=data_obj2.timestamp)

# random intercept only
md = smf.mixedlm(r_formula, df, groups=df['brcid'])

# random intercept, and random slope (with respect to time)
md = smf.mixedlm(r_formula, df, groups=df['brcid'], re_formula='~score_year')

mdf = md.fit(method=['lbfgs'], reml=True)  # other methods lbfgs bfgs cg
print(mdf.summary())
(mdf.summary().tables[1]).to_clipboard()

# fit a model in which the two random effects are constrained to be uncorrelated:
md = smf.mixedlm(r_formula, df, groups=df['brcid'], re_formula='~date')
free = sm.regression.mixed_linear_model.MixedLMParams.from_components(np.ones(2), np.eye(2))

mdf = md.fit(free=free, method=['lbfgs'])
print(mdf.summary())
