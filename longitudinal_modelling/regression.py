from sklearn.model_selection import train_test_split
from longitudinal_modelling.longitudinal_utils import *
from sklearn.linear_model import LinearRegression, LogisticRegression
import statsmodels.api as sm
from sklearn import metrics
import numpy as np
from mord import LogisticAT

# logistic regression: nlp_sc_anytime versus covariates
# ordinal regression: honos score_baseline (cog + total) versus covariates


def round_down(num, divisor):
    return num - (num % divisor)


def fit_reg(df, target, covariates, timestamp
            , reg_type=LogisticRegression(solver='sag', max_iter=1000)
            , test_size=0.1, random_state=911
            , dummyfy_non_num=False,  cols_to_dummyfy=None):
    if dummyfy_non_num:
        if cols_to_dummyfy is None:
            cols_to_dummyfy = [col for col in covariates if df[col].dtype == 'object']
        print('columns dummyfied', cols_to_dummyfy)
        dummyfied_cols = pd.get_dummies(df[cols_to_dummyfy], drop_first=True)
        covariates = list(dummyfied_cols.columns) + [col for col in covariates if df[col].dtype != 'object']
        df = pd.concat([df, dummyfied_cols], axis=1, sort=True)
    #x_train, x_test, y_train, y_test = train_test_split(df[list(covariates) + [timestamp]], df[target], test_size=test_size, random_state=random_state)
    x_train, y_train = [df[list(covariates) + [timestamp]], df[target]]
    if reg_type == 'ols':
        reg_type = sm.OLS(y_train, x_train).fit()
    elif reg_type == 'logit':
        reg_type = sm.Logit(y_train, x_train).fit()
    else:
        reg_type.fit(x_train, y_train)

    preds_train = reg_type.predict(x_train)
    print('training scores:\n',
          metrics.classification_report(round_down(preds_train, 5), round_down(y_train, 5)))

    return {'model': reg_type, 'preds': preds_train, 'y': y_train}
