import pandas as pd
import numpy as np

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from Transformers import QuantileTransformerDf, IterativeImputerDf

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    columns = ["F1", "F2", "F3", "F4"]

    X = [[1.0, np.nan, 0.2, 1.1], [2.0, 0.35, np.nan, 1.12], [2.5, 3, 2.32, 8.2],
         [0.5, 1.2, 0.8, 2.2], [0.3, np.nan, 4.5, np.nan]]
    df = pd.DataFrame(data=X, columns=columns)

    print(df.head())

    q_trans = QuantileTransformerDf()
    imp_mean = IterativeImputerDf(min_value=0,  # values from 0 to 1 for categorical for numeric
                                  max_value=1,
                                  random_state=0,
                                  max_iter=100,
                                  tol=1e-6,
                                  verbose=1, dataframe_as_output=True)

    pipe = Pipeline([
        ('q_trans', q_trans),
        ('imputer', imp_mean)
    ])

    # imp_mean.fit(df)
    # df_imputed = imp_mean.transform(df)

    pipe.fit(df)
    df_imputed = pipe.transform(df)
    print(f'df_original=\n{df}')
    print(f'df_imputed=\n{df_imputed}')
