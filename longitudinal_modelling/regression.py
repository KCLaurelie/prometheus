from sklearn.model_selection import train_test_split
from longitudinal_modelling.longitudinal_utils import *
from sklearn.linear_model import LinearRegression, LogisticRegression
from statsmodels.api import OLS
from mord import LogisticAT

# logistic regression: nlp_sc_anytime versus covariates
# ordinal regression: honos score_baseline (cog + total) versus covariates


def fit_reg(df, target, covariates, timestamp
            , reg_type=LogisticRegression(solver='sag', max_iter=1000)
            , test_size=0.1, random_state=911
            , dummyfy=False,  cols_to_dummyfy=None):
    if dummyfy:
        if cols_to_dummyfy is None:
            cols_to_dummyfy = [col for col in covariates if df[col].dtype == 'object']
        print('columns dummyfied', cols_to_dummyfy)
        dummyfied_cols = pd.get_dummies(df[cols_to_dummyfy])
        covariates = dummyfied_cols.columns
        df = pd.concat([df, dummyfied_cols], axis=1, sort=True)
    x_train, x_test, y_train, y_test = train_test_split(df[list(covariates) + [timestamp]], df[target]
                                                        , test_size=test_size, random_state=random_state)
    if reg_type == 'OLS':
        reg_type = OLS(df[target], df[list(covariates) + [timestamp]]).fit()
    else:
        reg_type.fit(x_train, y_train)

    #acc_log = round(reg_type.score(x_train, y_train) * 100, 2)
    #print('accuracy', acc_log)
    return reg_type
