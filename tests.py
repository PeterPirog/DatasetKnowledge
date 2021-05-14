
from sklearn.preprocessing import OneHotEncoder
import numpy as np


enc = OneHotEncoder(handle_unknown='ignore')
X = [['Male', 1], ['Female', 3], ['Female', 2]]
enc.fit(X)

enc.categories_

output=enc.transform([[np.nan, 1], ['Male', 4]]).toarray()
print(f'output= {output}')

enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])


enc.get_feature_names(['gender', 'group'])