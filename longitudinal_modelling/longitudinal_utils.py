import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class LongitudinalDataset:
    def __init__(self, data, sheet_name=0,  group='brcid', timestamp='date',
                 target='score', covariates=None,  # for regression model
                 to_bucket=None, bucket_min=50, bucket_max=90, interval=0.5, min_obs=3,  # to create groups
                 ):
        """
        create dataset object for trajectories modelling
        :param data: string or pandas DataFrame. filepath or DataFrame object containing longitudinal data.
        :param sheet_name: integer or string, default 0. if data provided via excel file, name of sheet to load
        :param group: string, default 'brcid'. column name from data with group identification (e.g. brcid)
        :param timestamp: string, default 'date'. key used as time measure (for baseline values, the oldest/smallest timestamp will be used)
        :param target: string, default 'score'. measure to predict
        :param covariates: string, default None. list of covariates for prediction modelling
        :param to_bucket: string, default None (e.g. 'age'}. on what variable to bucket the data if applicable (will groupby based on this variable)
        :param bucket_min: float (e.g. 50). min cutoff value for bucketting
        :param bucket_max: float (e.g. 90). max cutoff value for bucketting
        :param interval: float (e.g. 0.5). interval to use for bucketting (needs to be between bucket_min and bucket_max)
        :param min_obs: integer, default 3. remove individuals having less than min_obs observations
        """
        self.sheet_name = sheet_name
        self.data = data
        self.group = group
        self.timestamp = timestamp
        self.target = target
        self.covariates = to_list(covariates) if covariates is not None else None
        self.to_bucket = str(to_bucket) if to_bucket is not None else None
        self.bucket_min = bucket_min
        self.bucket_max = bucket_max
        self.interval = interval
        self.min_obs = min_obs

    def load_data(self):
        if isinstance(self.data, str): # load data if needed
            if 'csv' in self.data :
                self.data = pd.read_csv(self.data)
            else:
                self.data = pd.read_excel(self.data, sheet_name=self.sheet_name, engine='openpyxl')
        if self.to_bucket is not None:
            cols_to_keep = self.covariates + [self.group] + [self.target] + [self.timestamp] + [self.to_bucket]
        elif self.covariates is not None:
            cols_to_keep = self.covariates + [self.group] + [self.target] + [self.timestamp]
        else:
            cols_to_keep = self.data.columns
        # remove duplicates
        cols_to_keep = list(set(cols_to_keep))
        raise_missing = [x for x in cols_to_keep if x not in self.data.columns]
        if len(raise_missing) > 0:
            print('columns not found:', raise_missing, 'all original columns loaded')
        else:
            self.data = self.data[cols_to_keep]
        return self.data

    def normalize(self, cols_to_normalize=None):
        if cols_to_normalize is None:
            cols_to_normalize = [col for col in self.data[self.covariates]._get_numeric_data().columns]
        scaler = MinMaxScaler()
        x = self.data[cols_to_normalize].values
        scaled_values = scaler.fit_transform(x)
        self.data[cols_to_normalize] = scaled_values
        return self.data

    def dummyfy(self, cols_to_dummyfy=None):
        if cols_to_dummyfy is None:
            cols_to_dummyfy = self.data.select_dtypes(include=['object', 'category']).columns
        dummyfied_df = pd.get_dummies(self.data[cols_to_dummyfy])
        self.data = pd.concat([self.data.drop(columns=cols_to_dummyfy), dummyfied_df], axis=1, sort=True)
        return self.data

    def bucket_data(self):
        # only use data within bucket boundaries
        df = self.clean_data()
        mask_bucket = (df[self.to_bucket] >= self.bucket_min) & (df[self.to_bucket] <= self.bucket_max)
        df = df.loc[mask_bucket, :]
        # transform bool cols to "yes"/"no" so they are not averaged out in the groupby
        bool_cols = [col for col in df.columns if df[col].value_counts().index.isin([0, 1]).all()]
        if len(bool_cols) > 0: df[bool_cols] = df[bool_cols].replace({0: 'no', 1: 'yes'})
        # detect numerical and categorical columns
        categoric_col = [col for col in df.select_dtypes(include=['object', 'category']).columns
                         if (self.group not in col)]
        numeric_col = [col for col in df._get_numeric_data().columns if (col != self.group)]
        # group by buckets
        bucket_col = self.to_bucket + '_upbound'
        df[bucket_col] = round_nearest(df[self.to_bucket], self.interval, 'up')
        # we aggregate by average for numeric variables and baseline value for categorical variables
        keys = categoric_col + numeric_col
        values = ['first'] * len(categoric_col) + ['mean'] * len(numeric_col)
        grouping_dict = dict(zip(keys, values))

        df_grouped = df.groupby([self.group] + [bucket_col], as_index=False).agg(grouping_dict)
        df_grouped = df_grouped.sort_values([self.group, self.to_bucket])

        df_grouped['occur'] = df_grouped.groupby(self.group)[self.group].transform('size')
        df_grouped = df_grouped[(df_grouped['occur'] >= self.min_obs)]
        df_grouped['counter'] = df_grouped.groupby(self.group).cumcount() + 1

        return df_grouped


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
