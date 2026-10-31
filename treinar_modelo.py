import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import TrainingArguments, Trainer

df = pd.read_csv('data/imdb_limpo.csv')

x = df['review_limpa']
y = df['sentiment']

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=50)

vetorizador = TfidfVectorizer(max_features=5000)

x_treino_tfidf = vetorizador.fit_transform(x_treino)
x_teste_tfidf = vetorizador.transform(x_teste)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(x_treino_tfidf, y_treino)

previsoes = modelo.predict(x_teste_tfidf)

print("Acurácia:", accuracy_score(y_teste, previsoes))
print("\nRelatório de classificação:\n", classification_report(y_teste, previsoes))
print("\nMatriz de confusão:\n", confusion_matrix(y_teste, previsoes))

x_teste_original = df.loc[x_teste.index, 'review']
erros = x_teste_original[previsoes != y_teste.values]
print(erros.head())