import pandas as pd
import numpy as np


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    columns=["Number","Sex", "Country"]
    cat_features = ["Sex", "Country"]
    num_features=["Number"]

    X = [[1.0,np.nan, 'Germany'], [1.0,'Male', np.nan], [1.99,'Female', 'Poland'],[np.nan,'Female', 'Brasil'],[3.0,np.nan, 'Brasil']]
    df=pd.DataFrame(data=X,columns=columns)
    df_num=df[num_features]
    print(df_num.head())

    drop_columns=True
    for cat_feature in cat_features:
        labels=df[cat_feature].unique().tolist()
        labels = [str(x) for x in labels]  # converting nan to 'nan
        try:
            labels.remove('nan') #remove nan labels
        except:
            pass
        for label in labels:

            new_label=str(cat_feature)+'_'+str(label)
            print(f"New label: {new_label} Column: {cat_features}, label: {label}")
            df[new_label] = np.where(df[cat_feature] == label, 1, 0)
            df.loc[df[cat_feature].isna(), new_label] = np.nan
    if drop_columns:
        df=df.drop(columns=columns) #drop encoded columns

    df=pd.concat([df_num,df])
    print(df.head())
    new_columns=df.columns.to_list()
    print(new_columns)

    # explicitly require this experimental feature
    from sklearn.experimental import enable_iterative_imputer  # noqa
    # now you can import normally from sklearn.impute
    from sklearn.impute import IterativeImputer

    imp_mean = IterativeImputer(min_value=-np.inf,  # values from 0 to 1 for categorical for numeric
                                max_value=np.inf,
                                random_state=0,
                                max_iter=100,
                                tol=1e-6,
                                verbose=2)

    df_imputed=imp_mean.fit_transform(df)
    df_imputed=pd.DataFrame(df_imputed,columns=new_columns)
    print(df_imputed.head())