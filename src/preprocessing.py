import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

### CLEANING DATA

class DataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = X.copy()

        # Numeric conversion
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')

        # Binary mappings (NO Churn here!)
        binary_map = {
            'gender': {'Female': 1, 'Male': 0},
            'Partner': {'Yes': 1, 'No': 0},
            'Dependents': {'Yes': 1, 'No': 0},
            'PhoneService': {'Yes': 1, 'No': 0},
            'PaperlessBilling': {'Yes': 1, 'No': 0},
        }

        for col, mapping in binary_map.items():
            if col in X:
                X[col] = X[col].map(mapping)

        # MultipleLines
        if 'MultipleLines' in X:
            X['MultipleLines'] = (X['MultipleLines'] == 'Yes').astype(int)

        # Internet-dependent features
        internet_features = [
            'OnlineSecurity','OnlineBackup','DeviceProtection',
            'TechSupport','StreamingTV','StreamingMovies'
        ]

        for col in internet_features:
            if col in X:
                X[col] = (X[col] == 'Yes').astype(int)

        return X
    


### FEATURE ENGINEERING

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.median_charge_ = X['MonthlyCharges'].median()
        return self

    def transform(self, X):
        X = X.copy()

        # Groups
        X['tenure_group'] = pd.cut(
            X['tenure'],
            bins=[0, 12, 24, 48, 72],
            labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr']
        )

        X['avg_revenue'] = X['TotalCharges'] / (X['tenure'] + 1)

        return X
    
class FlagBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, flag_configs):
        self.flag_configs = flag_configs

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = X.copy()
        for new_col, func in self.flag_configs.items():
            X[new_col] = func(X).astype(int)
        return X


class HighMonthlyFlag(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.median_ = X['MonthlyCharges'].median()
        return self

    def transform(self, X):
        X = X.copy()
        X['high_monthly_flag'] = (X['MonthlyCharges'] > self.median_).astype(int)
        return X
    

class ServiceCounter(BaseEstimator, TransformerMixin):
    def __init__(self, service_cols):
        self.service_cols = service_cols

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = X.copy()
        X['num_services'] = (X[self.service_cols] == 'Yes').sum(axis=1)
        return X
    

class InteractionBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, interactions):
        self.interactions = interactions

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = X.copy()
        for new_col, (c1, c2) in self.interactions.items():
            X[new_col] = X[c1] * X[c2]
        return X


def is_new_customer(X):
    return X['tenure'] <= 12
def is_long_term(X):
    return X['Contract'].isin(['One year', 'Two year'])
def auto_pay_flag(X):
    return X['PaymentMethod'].isin(['Bank transfer (automatic)', 'Credit card (automatic)'])
def family_flag(X):
    return (X['Partner'] == 1) | (X['Dependents'] == 1)  # assuming binary mapping already applied
def fiber_flag(X):
    return X['InternetService'] == 'Fiber optic'
def electronic_check_flag(X):
    return X['PaymentMethod'] == 'Electronic check'

class FlagBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, flag_funcs):
        """
        flag_funcs: dict, keys are new column names, values are functions X -> series
        """
        self.flag_funcs = flag_funcs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, func in self.flag_funcs.items():
            X[col] = func(X).astype(int)
        return X
    
flag_builder = FlagBuilder({
    'is_new_customer': is_new_customer,
    'is_long_term': is_long_term,
    'auto_pay_flag': auto_pay_flag,
    'family_flag': family_flag,
    'fiber_flag': fiber_flag,
    'electronic_check_flag': electronic_check_flag
})

feature_pipeline = Pipeline([
    ('basic', FeatureEngineer()),
    ('flags', flag_builder),
    ('high_monthly', HighMonthlyFlag()),
    ('services', ServiceCounter([
        'OnlineSecurity','OnlineBackup','DeviceProtection',
        'TechSupport','StreamingTV','StreamingMovies'])),
    ('interactions', InteractionBuilder({
        'new_echeck_interaction': ('is_new_customer', 'electronic_check_flag'),
        'fiber_highcharge_interaction': ('fiber_flag', 'high_monthly_flag'),
        'loyal_engaged_interaction': ('is_long_term', 'num_services'),}))])


## REAUSABLE PREPROCESSING COMPONENTS

median_imputer = SimpleImputer(strategy='median')
most_frequent_imputer = SimpleImputer(strategy='most_frequent')
missing_cat_imputer = SimpleImputer(strategy='constant', fill_value='Missing')

scaler = StandardScaler()

ohe_drop_first = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
ohe_all = OneHotEncoder(handle_unknown='ignore', sparse_output=False)


### LINEAR AND TREE PREPROCESSING


def get_preprocessor(numeric_features, categorical_features, engineered_features):
    # Linear model preprocessing
    num_pipeline_linear = Pipeline([
        ('imputer', median_imputer),
        ('scaler', scaler)])
    cat_pipeline_linear = Pipeline([
        ('imputer', missing_cat_imputer),
        ('encoder', ohe_drop_first)])
    linear_preprocessor = ColumnTransformer([
        ('num', num_pipeline_linear, numeric_features + engineered_features),
        ('cat', cat_pipeline_linear, categorical_features)])

    # Tree-based preprocessing
    num_pipeline_tree = Pipeline([
        ('imputer', median_imputer)])
    cat_pipeline_tree = Pipeline([
        ('imputer', most_frequent_imputer),
        ('encoder', ohe_all)])
    tree_preprocessor = ColumnTransformer([
        ('num', num_pipeline_tree, numeric_features + engineered_features),
        ('cat', cat_pipeline_tree, categorical_features)])

    ## PREPROCESSOR PIPELINES
    full_pipeline_linear = Pipeline([
        ('cleaning', DataCleaner()),
        ('feature_engineering', feature_pipeline),
        ('preprocessing', linear_preprocessor)])
    full_pipeline_tree = Pipeline([
        ('cleaning', DataCleaner()),
        ('feature_engineering', feature_pipeline),
        ('preprocessing', tree_preprocessor)])
    return full_pipeline_linear, full_pipeline_tree