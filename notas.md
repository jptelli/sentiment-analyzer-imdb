O dataset veio do Kaggle -> IMDB Dataset of 50K Movie Reviews
50 000 reviews totais, sendo 25 000 positivas e 25 000 negativas
TF-IDF analisa palavra por palavra, não consegue compreender o contexto delas na frase, então quando uma review positiva contém palavras como hate, bad, worst, etc, o modelo tende a classifcar como uma análise negativa. Isso se repete em casos de análises negativas que contenham palavras positivas, casos de sarcasmo, ironia e reviews mistas. Apesar disso obteve uma precisão de quase 89% nas análises.
Vou fazer utilizando transformers, pois tem uma capacidade melhor de compreender o contexto das reviews
Rodei no Google Colab utilizando a GPU T4, com os arquivos IMDB Dataset.csv e treinar_distilbert.py
Precisão aumentou em cerca de 3,6%, obtendo 92,11%
