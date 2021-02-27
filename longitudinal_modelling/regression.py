from sklearn.model_selection import train_test_split
from longitudinal_modelling.longitudinal_utils import *
import statsmodels.api as sm
from sklearn import metrics

# logistic regression: nlp_sc_anytime versus covariates
# ordinal regression: honos score_baseline (cog + total) versus covariates


def round_down(num, divisor):
    return num - (num % divisor)


def fit_reg(df, target, covariates, timestamp=None
            , reg_fn=sm.OLS
            , intercept=True
            , test_size=0, random_state=911, round_stats=None
            , dummyfy_non_num=False,  cols_to_dummyfy=None):
    df_local = df.copy()
    covariates, target, timestamp = [to_list(covariates), to_list(target), to_list(timestamp)]
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
        x_train, x_test, y_train, y_test = train_test_split(df_local[covariates + timestamp], df_local[target], test_size=test_size, random_state=random_state)
    else:
        x_train, y_train = [df_local[covariates + timestamp], df_local[target]]
    # FIT REGRESSION
    if intercept: x_train = sm.add_constant(x_train)
    fitted_reg = reg_fn(y_train, x_train).fit()
    preds_train = fitted_reg.predict(x_train)

    # GENERATE STATS REPORT
    if round_stats is not None:
        report = metrics.classification_report(round_down(preds_train, round_stats), round_down(y_train, round_stats))
    else:
        report = metrics.classification_report(preds_train.astype(int), y_train.astype(int))
    print('training scores:\n', report)

    del df_local
    return {'model': fitted_reg, 'y_pred': preds_train, 'y_true': y_train, 'report': report}


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
