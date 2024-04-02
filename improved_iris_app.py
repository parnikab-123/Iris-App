import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the dataset into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


@st.cache()
def prediction(model, sepal_length, sepal_width, petal_length, petal_width):
  species = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
  species = species[0]
  if species == 0:
    return "Iris-setosa"
  elif species == 1:
    return "Iris-virginica"
  else:
    return "Iris-versicolor"

st.sidebar.title("Iris Species Classifer")
sepal_len= st.sidebar.slider('Sepal Length', float(iris_df['SepalLengthCm'].min()), float(iris_df['SepalLengthCm'].max()))
sepal_wid= st.sidebar.slider('Sepal Width', float(iris_df['SepalWidthCm'].min()), float(iris_df['SepalWidthCm'].max()))
petal_len= st.sidebar.slider('Petal Length', float(iris_df['PetalLengthCm'].min()), float(iris_df['PetalLengthCm'].max()))
petal_wid= st.sidebar.slider('Petal Width', float(iris_df['PetalWidthCm'].min()), float(iris_df['PetalWidthCm'].max()))
st.sidebar.subheader("Choose a Classifer")
select_box= st.sidebar.selectbox("Classifer", ('RandomForestClassifier', 'SupportVectorClassifer', 'LogisticRegression'))
if st.sidebar.button('Predict'):
	if select_box=='RandomForestClassifier':
		rf_clf = RandomForestClassifier(n_jobs = -1, n_estimators = 100)
		rf_clf.fit(X_train, y_train)

		species_name= prediction(rf_clf, sepal_len, sepal_wid, petal_len, petal_wid)
		score= rf_clf.score(X_train, y_train)
	elif select_box=='SupportVectorClassifer':
		svc_model = SVC(kernel = 'linear')
		svc_model.fit(X_train, y_train)

		species_name= prediction(svc_model, sepal_len, sepal_wid, petal_len, petal_wid)
		score= svc_model.score(X_train, y_train)
	else:
		log_reg = LogisticRegression(n_jobs = None)
		log_reg.fit(X_train, y_train)
		species_name= prediction(log_reg, sepal_len, sepal_wid, petal_len, petal_wid)
		score= log_reg.score(X_train, y_train)

	st.text(f'The species is: {species_name}')
	st.text(f'The score is: {score}')