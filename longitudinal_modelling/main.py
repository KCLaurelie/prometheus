#https://www.statsmodels.org/stable/examples/notebooks/generated/mixed_lm_example.html

from longitudinal_modelling.longitudinal_utils import *
from longitudinal_modelling.data_stats import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.regression.mixed_linear_model as mlm


data_obj = LongitudinalDataset(data=r'C:\Users\aurelie\PycharmProjects\prometheus\longitudinal_modelling\trajectories_synthetic.xlsm',
                               sheet_name='data',
                               target='score',
                               group='brcid',
                               covariates=['diagnosis', 'date', 'score', 'gender', 'med'])
df = data_obj.load_data()
#r_formula = 'score ~  date + age + diagnosis + gender + date * age + date * diagnosis + date * gender'
r_formula = make_smf_formula(target='score', covariates=['age', 'diagnosis', 'gender'], timestamp='date')

# random intercept only
md = smf.mixedlm(r_formula, df, groups=df['brcid'])

# random intercept, and random slope (with respect to time)
md = smf.mixedlm(r_formula, df, groups=df['brcid'], re_formula='~date')

mdf = md.fit(method=['cg'], reml=True)  # other methods lbfgs bfgs cg
print(mdf.summary())

# fit a model in which the two random effects are constrained to be uncorrelated:
md = smf.mixedlm(r_formula, df, groups=df['brcid'], re_formula='~date')
free = sm.regression.mixed_linear_model.MixedLMParams.from_components(np.ones(2), np.eye(2))

mdf = md.fit(free=free, method=['lbfgs'])
print(mdf.summary())
