
import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split

from sklearn.pipeline import  Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,QuantileTransformer


from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import MeanMedianImputer
from feature_engine.imputation import CategoricalImputer

if __name__ == '__main__':
    # load training data
    data = pd.read_csv('train.csv')

    # split training data into train and test

    X_train, X_test, y_train, y_test = train_test_split(data.drop(
        ['Id', 'SalePrice'], axis=1),
        data['SalePrice'],
        test_size=0.05,
        random_state=0,
        stratify=None)

    #get numerical labels
    numerical_labels=list(X_train._get_numeric_data().columns)
    categorical_labels = X_train.select_dtypes(include=['object']).columns.tolist()

    #moving 'MSSubClass' feature from numerical to categorical
    numerical_labels.remove('MSSubClass')
    categorical_labels.append('MSSubClass')

    print(f'Numerical labels are (contains ordinal cat):{numerical_labels}')
    print(f'Categorical labels are:{categorical_labels}')
    #print(X_train.head())

    num_pipeline=Pipeline([
        ('imputer',MeanMedianImputer(imputation_method='median'))#,
        #('std_scaler',StandardScaler())
    ])
    cat_pipeline=Pipeline([
        ('imputer',CategoricalImputer(imputation_method='missing', fill_value='Missing')),
        ('one_hot', OneHotEncoder(top_categories=None,drop_last=False))
    ])

    full_pipeline=ColumnTransformer([
        ('num',num_pipeline,numerical_labels),
        ('cat',cat_pipeline,categorical_labels)
    ])

    X_converted=cat_pipeline.fit_transform(X_train)
    print(X_converted.head())