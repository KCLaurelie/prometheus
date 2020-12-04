#https://www.statsmodels.org/stable/examples/notebooks/generated/mixed_lm_example.html

from longitudinal_modelling.longitudinal_utils import *
from longitudinal_modelling.data_stats import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.regression.mixed_linear_model as mlm

df = pd.read_excel(r'C:\Users\K1774755\PycharmProjects\prometheus\longitudinal_modelling\trajectories_synthetic.xlsm',
                   sheet_name='data')
# ['brcid', 'diagnosis', 'date', 'score', 'gender', 'med']
r_formula = 'score ~  date + age + diagnosis + gender + date * age + date * diagnosis + date * gender'

# random intercept only
md = smf.mixedlm(r_formula, df, groups=df['brcid'])

# random intercept, and random slope (with respect to time)
md = smf.mixedlm(r_formula, df, groups=df['brcid'], re_formula="~date")

mdf = md.fit(method=["lbfgs"])
print(mdf.summary())

# fit a model in which the two random effects are constrained to be uncorrelated:
md = smf.mixedlm(r_formula, df, groups=df['brcid'], re_formula="~date")
free = sm.regression.mixed_linear_model.MixedLMParams.from_components(np.ones(2),
                                                                      np.eye(2))

mdf = md.fit(free=free, method=["lbfgs"])
print(mdf.summary())
