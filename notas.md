# Notas do projeto

## Dataset
IMDB Dataset of 50K Movie Reviews (Kaggle): 50.000 reviews, 25.000 positivas e 25.000 negativas (classes balanceadas).

## Baseline: TF-IDF + Regressão Logística
- Acurácia de 88,5% no conjunto de teste (10.000 reviews).
- Limitação: o TF-IDF analisa palavra por palavra e não entende o contexto. Reviews positivas com palavras como "hate", "bad" ou "worst" tendem a ser classificadas como negativas, e o mesmo ocorre ao contrário. Também falha com sarcasmo, ironia e reviews mistas.
- Exemplos reais de erro: "not that bad" (a negação é ignorada) e "if you hated the book, you will likely hate the movie" (a estrutura condicional se perde).

## Upgrade: DistilBERT (fine-tuning)
- Transformers entendem o contexto, porque processam o texto em sequência.
- Treino no Google Colab (GPU T4), 2 épocas, max_length=256, aproximadamente 10 minutos.
- Acurácia de 92,11% no mesmo conjunto de teste, um ganho de 3,6 pontos percentuais.
- Em erros absolutos, de 1.146 para 789 em 10.000 reviews (31% menos erros).

## Decisões
- Mantive "not", "no" e "nor" fora da lista de stopwords no baseline, para preservar as negações.
- Mesmo split (random_state=50) nos dois modelos, para a comparação ser justa.
- O conjunto de teste foi usado apenas para monitorar o treino, e o modelo final (época 2) foi o avaliado, sem escolher a melhor época.