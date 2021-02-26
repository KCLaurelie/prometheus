from sklearn.model_selection import train_test_split
from longitudinal_modelling.longitudinal_utils import *
from sklearn.linear_model import LinearRegression, LogisticRegression
import sklearn.metrics as metrics
import statsmodels.api as sm
import statsmodels.miscmodels as mm
from sklearn import metrics
from mord import LogisticAT

# logistic regression: nlp_sc_anytime versus covariates
# ordinal regression: honos score_baseline (cog + total) versus covariates


def round_down(num, divisor):
    return num - (num % divisor)


def fit_reg(df, target, covariates, timestamp
            , reg_type=LogisticRegression(solver='sag', max_iter=1000)
            , intercept=True
            , test_size=0, random_state=911, round_stats=None
            , dummyfy_non_num=False,  cols_to_dummyfy=None):
    df_local = df.copy()
    # GENERATE DUMMY VARIABLES
    if dummyfy_non_num:
        if cols_to_dummyfy is None:
            cols_to_dummyfy = [col for col in covariates if df_local[col].dtype == 'object']
        if len(cols_to_dummyfy) > 0:
            print('columns dummyfied', cols_to_dummyfy)
            dummyfied_cols = pd.get_dummies(df_local[cols_to_dummyfy], drop_first=True)
            covariates = list(dummyfied_cols.columns) + [col for col in covariates if df_local[col].dtype != 'object']
            df_local = pd.concat([df_local, dummyfied_cols], axis=1, sort=True)
    # SPLIT TEST/TRAINING SETS
    if test_size > 0:
        x_train, x_test, y_train, y_test = train_test_split(df_local[list(covariates) + [timestamp]], df_local[target], test_size=test_size, random_state=random_state)
    else:
        x_train, y_train = [df_local[list(covariates) + [timestamp]], df_local[target]]
    # FIT REGRESSION
    if intercept: x_train = sm.add_constant(x_train)
    if reg_type == 'ols':  # linear regression from statsmodel
        reg_type = sm.OLS(y_train, x_train).fit()
    elif reg_type == 'logit':  # logistic regression from statsmodel
        reg_type = sm.Logit(y_train, x_train).fit()
    elif reg_type == 'ordinal':  # ordinal regression, only available in mord at the moment
        if x_train.ndim == 1:  # need to reshape if only 1 feature
            x_train = np.array(x_train).reshape(-1, 1)
        reg_type = LogisticAT(alpha=0).fit(x_train, y_train)
    else:  # will work to call any model from sklearn
        reg_type.fit(x_train, y_train)

    preds_train = reg_type.predict(x_train)

    # GENERATE STATS REPORT
    if round_stats is not None:
        report = metrics.classification_report(round_down(preds_train, round_stats), round_down(y_train, round_stats))
    else:
        report = metrics.classification_report(preds_train.astype(int), y_train.astype(int))
    print('training scores:\n', report)

    del df_local
    return {'model': reg_type, 'y_pred': preds_train, 'y_true': y_train, 'report': report}


def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    res = {
        'Explained_variance: ': round(explained_variance, 4),
        'Mean_squared_log_error: ': round(mean_squared_log_error, 4),
        'R-squared: ': round(r2, 4),
        'Mean_Abs_Err: ': round(mean_absolute_error, 4),
        'Mean_Sq_Err: ': round(mse, 4),
        'Root_MSE: ': round(np.sqrt(mse), 4)
    }
    return res
