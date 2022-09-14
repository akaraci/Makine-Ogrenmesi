

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd


X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [5, 6],[5, 6], [7, 8], [3, 4], [20, 6]])

  
mms=MinMaxScaler(feature_range=(-1,1))
X_normalized_with_min_max=mms.fit_transform(X)
df=pd.DataFrame(X_normalized_with_min_max) #convert data frame
print("\n-------MinMax Normalized data\n",df)

inverse=mms.inverse_transform(X_normalized_with_min_max)
df=pd.DataFrame(inverse)
print("\n-------MinMax Inverse Data\n",df)


sss=StandardScaler()
X_normalized_with_standart_scalar=sss.fit_transform(X)
df=pd.DataFrame(X_normalized_with_standart_scalar)
print("\n-------Standart Scalar Normalized data\n",df)

inverse=sss.inverse_transform(X_normalized_with_standart_scalar)
df=pd.DataFrame(inverse)
print("\n-------Standart Scalar Inverse Data\n",df)