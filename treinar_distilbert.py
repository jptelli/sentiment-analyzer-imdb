import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import TrainingArguments, Trainer

df = pd.read_csv("data/IMDB Dataset.csv")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

modelo = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels = 2
)

def limpar_texto(texto):
    texto_limpo = re.sub(r'<.*?>', ' ', texto)
    return texto_limpo

df['review_limpa'] = df['review'].apply(limpar_texto)


mappings = {
    'positive': 1,
    'negative': 0
}

df['sentiment_binary'] = df['sentiment'].map(mappings)

x = df['review_limpa']
y = df['sentiment_binary']

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=50)

# print(len(x_treino), len(x_teste))
# print(y_treino.value_counts())

lista_reviews_treino = x_treino.tolist()
lista_reviews_teste = x_teste.tolist()
labels_treino = y_treino.tolist()
labels_teste = y_teste.tolist()

tokens_treino = tokenizer(lista_reviews_treino, truncation = True, padding = True, max_length = 256)
tokens_teste = tokenizer(lista_reviews_teste, truncation = True, padding = True, max_length = 256)

dataset_treino = Dataset.from_dict({
    "input_ids": tokens_treino["input_ids"],
    "attention_mask": tokens_treino["attention_mask"],
    "labels": labels_treino
})

dataset_teste = Dataset.from_dict({
    "input_ids": tokens_teste["input_ids"],
    "attention_mask": tokens_teste["attention_mask"],
    "labels": labels_teste
})

print(dataset_treino)
print(dataset_teste)

def calcular_metricas(eval_pred):
    logits, labels = eval_pred
    previsoes = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, previsoes)}

args = TrainingArguments(
    output_dir="resultados",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="no",
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model = modelo,
    args = args,
    train_dataset = dataset_treino,
    eval_dataset = dataset_teste,
    compute_metrics = calcular_metricas
)

trainer.train()
trainer.save_model("modelo_distilbert")