import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

tqdm.pandas()

df = pd.read_csv('data/IMDB Dataset.csv')

stop_words = set(stopwords.words('english'))
negacoes = {'not', 'no', 'nor'}
stop_words_sem_negacoes = stop_words - negacoes

def limpar_texto(texto):
    texto_limpo = re.sub(r'<.*?>', ' ', texto)
    texto_lower = texto_limpo.lower()
    texto_sem_pontuacao = re.sub(r'[,.()\'"!?;:-]', '', texto_lower)
    palavras = word_tokenize(texto_sem_pontuacao, language=('english'))
    palavras_filtradas = [p for p in palavras if p not in stop_words_sem_negacoes]
    return " ".join(palavras_filtradas)

df['review_limpa'] = df['review'].progress_apply(limpar_texto)

print(df[['review', 'review_limpa']].head())

df.to_csv('data/imdb_limpo.csv', index=False)