import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
#https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156

if __name__ == '__main__':
    num_features=["Number1","Number2", "Number3"]
    X = [[1.0,np.nan, 0.2], [1.0,0.2, np.nan], [1.99,0.7, 5.0],[np.nan,0.9, 2.0],[3.0,np.nan, 1.2]]
    df=pd.DataFrame(data=X,columns=num_features)

    print(df.head())

    for feature in num_features:
        x=df[feature]
        idx=df.loc[pd.isna(df[feature]), :].index
        ecdf = ECDF(x)
        output=ecdf(x)
        output[idx]=np.nan
        print(f'idx={idx}')
        print(f' ECDF:{output}')

    #ecdf = ECDF([3, 3, 1, 4])
    #print(ecdf([3, 55, 0.5, 1.5]))
