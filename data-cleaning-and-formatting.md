# Data Cleaning & formatting

```python
import pandas as pd
import numpy as np
```

## Load the data into a pandas DataFrame
```python
data = pd.read_csv('data.csv')
```

## Remove any rows with missing values
```python
data = data.dropna()
```

## Remove any outliers using the interquartile range method
```python
q1 = data.quantile(0.25)
q3 = data.quantile(0.75)
iqr = q3 - q1
data = data[~((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).any(axis=1)]
```

## Convert the data into a format that can be used with a specific machine learning library
## In this example, we will convert the data into numpy arrays
```python
X = data.drop('weather_variable', axis=1).values
y = data['weather_variable'].values
```

## Split the data into training and testing sets
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
## Load the data, remove missing values and outliers and convert the data into a format that can be used with a specific machine learning library(in this case scikit-learn), split the data into training and testing sets. 
## Additional code

## Handle categorical variables
## One hot encoding
```python
data = pd.get_dummies(data, columns=['categorical_variable'])
```

## Label encoding
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['categorical_variable'] = le.fit_transform(data['categorical_variable'])
```

## Scale numerical variables
## Min-max scaling
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data[['numerical_variable1','numerical_variable2']] = scaler.fit_transform(data[['numerical_variable1','numerical_variable2']])
```

## Standardization 
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[['numerical_variable1','numerical_variable2']] = scaler.fit_transform(data[['numerical_variable1','numerical_variable2']])
```
The first block of code shows how to handle categorical variables. In the first line, using the Pandas "get_dummies" function to one hot encode the categorical variable 'categorical_variable'. In the second block, I am using scikit-learn's LabelEncoder class to label encode the categorical variable 'categorical_variable'.
The second block of code shows how to scale numerical variables. Here, using the MinMaxScaler to scale the numerical variables 'numerical_variable1' and 'numerical_variable2' to a range of [0,1]. In the second line of this block, using the StandardScaler to standardize the numerical variables 'numerical_variable1' and 'numerical_variable2'
Data preprocessing is an iterative process, so will require multiple rounds of cleaning, and formatting
Code for Splitting the data into training and testing sets. This will be used to train and evaluate the performance of the model.
In Python use the train_test_split function from the sklearn.model_selection module. This his function used from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
In this example, X is the feature data and y is the target variable data. The test_size parameter is used to specify the proportion of the data that should be allocated to the testing set (in this case, 20% of the data). The random_state parameter is used to ensure that the data is split in a reproducible way.
We also have other ways to split data into training and test set, such as k-fold cross-validation, Stratified k-fold cross-validation and so on, the choice of the method depends on the dataset and the problem at hand.
It's important to make sure that the training and testing set have similar distribution of the target variable, this achieved by using the stratify parameter.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
Here  assumption is that we have already loaded and pre-processed the data correctly and have it in the form of X,y. We use the train_test_split function for splitting your data into training and testing sets. 
Complete data cleaning, formatting and splitting process might be implemented in Python, using the Pandas and Numpy library:
```python
import pandas as pd
import numpy as np
# Load the data into a pandas DataFrame
data = pd.read_csv('data.csv')
# Remove any rows with missing values
data = data.dropna()
# Remove any outliers using the interquartile range method
q1 = data.quantile(0.25)
q3 = data.quantile(0.75)
iqr = q3 - q1
data = data[~((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).any(axis=1)]
# Handle categorical variables
# One hot encoding
data = pd.get_dummies(data, columns=['categorical_variable'])
# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['categorical_variable'] = le.fit_transform(data['categorical_variable'])
# Scale numerical variables
# Min-max scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data[['numerical_variable1','numerical_variable2']] = scaler.fit_transform(data[['numerical_variable1','numerical_variable2']])
#Standardization 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[['numerical_variable1','numerical_variable2']] = scaler.fit_transform(data[['numerical_variable1','numerical_variable2']])
# Convert the data into a format that can be used with a specific machine learning library
X = data.drop('weather_variable', axis=1).values
y = data['weather_variable'].values
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
This code loads the data from a csv file using the Pandas library, removes any missing values and outliers, handles categorical variables using one hot encoding, label encoding and scale numerical variables using Min-Max Scaling and Standardization. Then, it converts the data into a format that can be used with a machine learning library and splits it into training and testing sets using the train_test_split function from the sklearn.model_selection module and stratifying the data.
This is our code to show the complete data cleaning, formatting and splitting process might be implemented, with additional pre-processing steps, and adjusting code to match the names of the variables in the dataset, and the method of splitting the data:
import pandas as pd
import numpy as np
# Load the data into a pandas DataFrame
data = pd.read_csv('data.csv')
# Remove any rows with missing values
data = data.dropna()
# Remove any outliers using the interquartile range method
q1 = data.quantile(0.25)
q3 = data.quantile(0.75)
iqr = q3 - q1
data = data[~((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).any(axis=1)]
# Handle categorical variables
# One hot encoding
data = pd.get_dummies(data, columns=['categorical_variable1','categorical_variable2'])
# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['categorical_variable3'] = le.fit_transform(data['categorical_variable3'])
# Scale numerical variables
# Min-max scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data[['numerical_variable1','numerical_variable2']] = scaler.fit_transform(data[['numerical_variable1','numerical_variable2']])
#Standardization 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[['numerical_variable3','numerical_variable4']] = scaler.fit_transform(data[['numerical_variable3','numerical_variable4']])

# Remove correlated features
correlated_features = set()
correlation_matrix = data.corr()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
data.drop(correlated_features, axis=1, inplace=True)

# Convert the data into a format that can be used with a specific machine learning library
X = data.drop('weather_variable', axis=1).values
y = data['weather_variable'].values

# Split the data into training and testing sets using k-fold cross-validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=42, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #train the model using X_train, y_train
    #test the model using X_test, y_test
```
