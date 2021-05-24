import pandas as pd
import numpy as np


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    columns=["Sex", "Country"]
    X = [[np.nan, 'Germany'], ['Male', np.nan], ['Female', 'Poland'],['Female', 'Brasil'],[np.nan, 'Brasil']]
    df=pd.DataFrame(data=X,columns=columns)
    print(df.head())

    drop_columns=True
    for column in columns:
        labels=df[column].unique().tolist()
        labels = [str(x) for x in labels]  # converting nan to 'nan
        try:
            labels.remove('nan') #remove nan labels
        except:
            pass
        for label in labels:

            new_label=str(column)+'_'+str(label)
            print(f"New label: {new_label} Column: {column}, label: {label}")
            df[new_label] = np.where(df[column] == label, 1, 0)
            df.loc[df[column].isna(), new_label] = np.nan
    if drop_columns:
        df=df.drop(columns=columns) #drop encoded columns


    print(df.head())
    new_columns=df.columns.to_list()
    print(new_columns)

    # explicitly require this experimental feature
    from sklearn.experimental import enable_iterative_imputer  # noqa
    # now you can import normally from sklearn.impute
    from sklearn.impute import IterativeImputer

    imp_mean = IterativeImputer(min_value=0,
                                max_value=1,
                                random_state=0,
                                max_iter=100,
                                tol=1e-6,
                                verbose=2)

    df_imputed=imp_mean.fit_transform(df)
    df_imputed=pd.DataFrame(df_imputed,columns=new_columns)
    print(df_imputed.head())