import pandas as pd
import numpy as np


def make_smf_formula(covariates, timestamp=None):
    str_cov = ' + '.join(covariates)
    if timestamp is not None:
        str_cov = timestamp + ' + ' + str_cov
        add_slope = ' + ' + timestamp + ' * '
        str_cov += add_slope + add_slope.join(covariates)
    return str_cov


def cut_with_na(to_bin, bins, labels, na_category='not known'):
    to_bin = pd.to_numeric(to_bin, errors='coerce')
    res = pd.cut(pd.Series(to_bin),
                 bins=bins,
                 labels=labels
                 ).values.add_categories(na_category)
    res = res.fillna(na_category)
    return res


def round_nearest(x, intv=0.5, direction='down'):
    """
    rounds number or series of numbers to nearest value given interval
    :param x: number or pd>Series of numbers to round
    :param intv: interval to round to
    :param direction: up (ceil) or down (floor)
    :return: rounded number or series of numbers
    """
    if direction == 'down':
        res = np.floor(x / intv) * intv
    else:
        res = np.ceil(x / intv) * intv
    return res


def to_list(x):
    """
    converts variable to a list
    :param x: variable to convert
    :return: variable converted to list
    """
    if x is None:
        res = []
    elif isinstance(x, str):
        res = [x]
    elif isinstance(x, list):
        res = x
    else:
        res = list(x)
    return res


# source: https://www.nhs.uk/common-health-questions/lifestyle/what-is-the-body-mass-index-bmi/
def bmi_category(score, na_category='not known'):
    res = cut_with_na(to_bin=score,
                      bins=[-np.inf, 18.5, 25, 30, np.inf],
                      labels=['underweight', 'normal', 'preobese', 'obese'],
                      na_category=na_category)
    return res


# source: https://www.cdc.gov/bloodpressure/measure.htm
def blood_pressure(systolic, diastolic, na_category='not known'):
    ratio = pd.to_numeric(systolic, errors='coerce') / pd.to_numeric(diastolic, errors='coerce')
    res = cut_with_na(to_bin=ratio,
                      bins=[-np.inf, 120. / 80, 140. / 90, np.inf],
                      labels=['normal', 'prehypertension', 'hypertension'],
                      na_category=na_category)
    return res


# source: https://www.diabetes.co.uk/fasting-plasma-glucose-test.html
def diabetes(plasma_glucose, na_category='not known'):
    res = cut_with_na(to_bin=plasma_glucose,
                      bins=[-np.inf, 5.5, 7, np.inf],
                      labels=['normal', 'prediabetic', 'diabetic'],
                      na_category=na_category)
    return res


def convert_num_to_bucket(nb, bucket_size=0.5, convert_to_str=True):
    lower_bound = np.ceil(nb / bucket_size) * bucket_size - bucket_size
    upper_bound = np.ceil(nb / bucket_size) * bucket_size
    res = [lower_bound, upper_bound]
    if convert_to_str: res = str(res)
    return res


def bucket_data(df, to_bucket, key='brcid', bucket_min=None, bucket_max=None, interval=None, cols_to_exclude=None,
                na_values='unknown', min_obs=3, timestamp_cols='score_year'):
    print("bucketting data (method from parent class)")
    cols_to_keep = [x for x in df.columns if
                    x not in to_list(cols_to_exclude)] if cols_to_exclude is not None else df.columns
    # only use data within bucket boundaries
    mask_bucket = (df[to_bucket] >= bucket_min) & (df[to_bucket] <= bucket_max)
    df = df.loc[mask_bucket, cols_to_keep]
    if na_values is not None: df.fillna(na_values, inplace=True)
    # transform bool cols to "yes"/"no" so they are not averaged out in the groupby
    bool_cols = [col for col in df.columns if df[col].value_counts().index.isin([0, 1]).all()]
    if len(bool_cols) > 0: df[bool_cols] = df[bool_cols].replace({0: 'no', 1: 'yes'})
    # detect numerical and categorical columns
    categoric_col = [col for col in df.select_dtypes(include=['object', 'category']).columns if (key not in col)]
    numeric_col = [col for col in df._get_numeric_data().columns if (col != key)]
    # group by buckets
    bucket_col = to_bucket + '_upbound'
    df[bucket_col] = round_nearest(df[to_bucket], interval, 'up')
    # we aggregate by average for numeric variables and baseline value for categorical variables
    keys = categoric_col + numeric_col
    values = ['first'] * len(categoric_col) + ['mean'] * len(numeric_col)
    grouping_dict = dict(zip(keys, values))

    df_grouped = df.groupby([key] + [bucket_col], as_index=False).agg(grouping_dict)
    df_grouped = df_grouped.sort_values([key, to_bucket])

    df_grouped['occur'] = df_grouped.groupby(key)[key].transform('size')
    df_grouped = df_grouped[(df_grouped['occur'] >= min_obs)]
    df_grouped['counter'] = df_grouped.groupby(key).cumcount() + 1
    if timestamp_cols is not None:
        for x in to_list(timestamp_cols):
            df_grouped[x + '_upbound'] = round_nearest(df_grouped[x], interval, 'up')
            df_grouped[x + '_centered'] = df_grouped[x + '_upbound'] - df_grouped[x + '_upbound'].min()

    return df_grouped
