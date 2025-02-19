from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs):
        ...

    def fit(self, X, y=None):


        return self

    def transform(self, X):
        new_X = X.copy()

        # Feature engineering
        new_X['CREDIT_INCOME_PERCENT'] = new_X['AMT_CREDIT'] / new_X['AMT_INCOME_TOTAL']
        new_X['ANNUITY_INCOME_PERCENT'] = new_X['AMT_ANNUITY'] / new_X['AMT_INCOME_TOTAL']
        new_X['CREDIT_TERM'] = new_X['AMT_ANNUITY'] / new_X['AMT_CREDIT']
        new_X['DAYS_EMPLOYED_PERCENT'] = new_X['DAYS_EMPLOYED'] / new_X['DAYS_BIRTH']

        return new_X