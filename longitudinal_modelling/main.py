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
