# Sentiment Analyzer IMDB

Classificação de sentimentos em avaliações de filmes: um baseline clássico (**TF-IDF + Regressão Logística**) comparado a um **DistilBERT** ajustado (*fine-tuning*) sobre o mesmo conjunto de teste.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-baseline-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers)
[![PyTorch](https://img.shields.io/badge/PyTorch-backend-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Acurácia](https://img.shields.io/badge/DistilBERT-92%2C11%25-brightgreen)](#resultados)
[![Acurácia baseline](https://img.shields.io/badge/Baseline-88%2C5%25-blue)](#resultados)

## Sumário

- [Visão geral](#visão-geral)
- [Resultados](#resultados)
- [Funcionalidades](#funcionalidades)
- [Tecnologias](#tecnologias)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Uso](#uso)
- [Decisões de projeto](#decisões-de-projeto)
- [Limitações](#limitações)
- [Como contribuir](#como-contribuir)
- [Licença](#licença)

## Visão geral

O objetivo é classificar uma avaliação de filme em **positiva** ou **negativa**. O projeto usa o [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), com 50.000 avaliações em inglês e classes balanceadas (25.000 positivas e 25.000 negativas).

O trabalho foi conduzido em duas etapas, para mostrar *por que* migrar de uma abordagem para outra:

1. **Baseline (TF-IDF + Regressão Logística).** Rápido de treinar e fácil de interpretar. Porém o TF-IDF trata o texto como uma sequência de palavras desconexas: ignora ordem e contexto. A análise de erros mostrou falhas em frases como *"not that bad"* (a negação perde o efeito) e *"if you hated the book, you will likely hate the movie"* (a estrutura condicional se perde).
2. **DistilBERT (fine-tuning).** Um Transformer pré-treinado lê o texto em sequência e usa o contexto de cada palavra, o que reduz esse tipo de erro.

Os dois modelos usam **o mesmo split treino/teste** (80/20, `random_state=50`), então a comparação é feita nas mesmas 10.000 avaliações.

## Resultados

Avaliação em 10.000 avaliações de teste:

| Modelo                       | Acurácia | Erros (de 10.000) | Treino                         |
| ---------------------------- | -------: | ----------------: | ------------------------------ |
| TF-IDF + Regressão Logística |   88,54% |             1.146 | segundos, em CPU               |
| DistilBERT (2 épocas)        |   92,11% |               789 | ~10 min, GPU T4 (Google Colab) |

O DistilBERT ganhou **3,6 pontos percentuais** de acurácia, o que equivale a cerca de **31% menos erros** que o baseline.

Matriz de confusão do baseline (linhas = real, colunas = previsto):

|              | previsto negativo | previsto positivo |
| ------------ | ----------------: | ----------------: |
| **negativo** |             4.426 |               610 |
| **positivo** |               536 |             4.428 |

## Funcionalidades

- Exploração inicial do dataset (balanceamento de classes e tamanho médio das avaliações).
- Pré-processamento de texto: remoção de tags HTML, normalização para minúsculas, remoção de pontuação e de *stopwords* (preservando as negações).
- Baseline com TF-IDF (5.000 atributos) e Regressão Logística, com acurácia, relatório de classificação e matriz de confusão.
- Listagem de exemplos que o baseline errou, para análise qualitativa.
- Fine-tuning do `distilbert-base-uncased` com a API `Trainer` da Hugging Face e salvamento do modelo treinado.

## Tecnologias

| Área                    | Ferramentas                          |
| ----------------------- | ------------------------------------ |
| Linguagem               | Python                               |
| Dados                   | pandas                               |
| ML clássico             | scikit-learn                         |
| Processamento de texto  | NLTK, expressões regulares (`re`)    |
| Deep learning           | PyTorch, Hugging Face Transformers, `datasets`, `accelerate` |
| Ambiente de treino      | Google Colab (GPU T4)                |

## Estrutura do repositório

```text
.
├── preProcessar.py        # limpeza do texto -> data/imdb_limpo.csv
├── treinar_modelo.py      # baseline: TF-IDF + Regressão Logística
├── treinar_distilbert.py  # fine-tuning do DistilBERT (requer GPU)
├── notas.md               # anotações e análise de erros
├── requirements.txt
└── data/                  # NÃO versionada: coloque aqui o CSV do dataset
```

## Pré-requisitos

- Python 3 e `pip`
- Conta no [Kaggle](https://www.kaggle.com/) para baixar o dataset
- Para o fine-tuning: uma **GPU NVIDIA** ou o [Google Colab](https://colab.research.google.com/) com GPU T4 (o script usa `fp16=True`, que não funciona em CPU)

## Instalação

```bash
# 1. clonar o repositório
git clone https://github.com/jptelli/sentiment-analyzer-imdb.git
cd sentiment-analyzer-imdb

# 2. criar e ativar o ambiente virtual
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 3. instalar as dependências
python -m pip install -r requirements.txt
```

> O `requirements.txt` inclui o PyTorch, que é grande. Para rodar **apenas o baseline**, basta instalar `pandas scikit-learn nltk tqdm`.

**Dataset.** Baixe o [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), extraia o zip e coloque o arquivo em `data/IMDB Dataset.csv`.

## Uso

### 1. Pré-processar o texto

```bash
python preProcessar.py
```

Gera `data/imdb_limpo.csv` com a coluna `review_limpa`. Em 50.000 avaliações, leva de 1 a 3 minutos.

### 2. Treinar e avaliar o baseline

```bash
python treinar_modelo.py
```

Imprime a acurácia, o relatório de classificação, a matriz de confusão e alguns exemplos de erro.

### 3. Fine-tuning do DistilBERT (GPU)

O script lê o CSV original (`data/IMDB Dataset.csv`) e usa o texto com as stopwords e a ordem das palavras preservadas, só sem as tags HTML. Isso porque o Transformer precisa do contexto completo.

No **Google Colab**:

1. Ative a GPU em *Ambiente de execução → Alterar tipo de ambiente de execução → T4 GPU*.
2. Envie `treinar_distilbert.py` e `IMDB Dataset.csv` e organize os arquivos:

   ```bash
   !mkdir -p data
   !mv "IMDB Dataset.csv" data/
   !pip install transformers datasets accelerate
   !python treinar_distilbert.py
   ```

3. Ao final, o modelo é salvo em `modelo_distilbert/`. Para baixá-lo:

   ```bash
   !zip -r modelo_distilbert.zip modelo_distilbert
   ```

A acurácia no teste é exibida ao fim de cada época (aprox. 90,6% na época 1 e 92,1% na época 2).

## Decisões de projeto

- **Mesmo split nos dois modelos** (`test_size=0.2`, `random_state=50`), para que a diferença de acurácia reflita a diferença entre os modelos, e não entre conjuntos de teste.
- **Negações preservadas no baseline.** As palavras `not`, `no` e `nor` foram retiradas da lista de *stopwords*, porque removê-las transformaria *"not good"* em *"good"*.
- **Texto quase bruto no DistilBERT.** Só as tags HTML são removidas. O tokenizador `uncased` já converte para minúsculas, e a pontuação e as stopwords ajudam o modelo a entender o contexto.
- **`max_length=256`** na tokenização, como equilíbrio entre tempo de treino e informação mantida. Avaliações mais longas são truncadas.
- **Hiperparâmetros do fine-tuning:** 2 épocas, *batch size* 16, *learning rate* 2e-5, precisão mista (`fp16`).

## Limitações

- **Não há conjunto de validação separado.** O conjunto de teste foi usado para monitorar o treino a cada época. O modelo avaliado é o final (época 2), sem escolher a melhor época com base no teste, mas o ideal seria um terceiro conjunto.
- **Uma única execução.** Os resultados vêm de um único treino, sem repetição com outras sementes, então não há estimativa de variância.
- **`eval_loss` subiu na época 2** (0,2225 para 0,2546), embora a acurácia tenha melhorado. Sinal de que o modelo fica mais confiante também nos erros; vale investigar com calibração ou mais dados de validação.
- **Truncamento.** Avaliações com mais de 256 tokens perdem o final do texto.
- **Idioma e domínio.** O modelo foi treinado em inglês, com avaliações de filmes; não é esperado que generalize para outros idiomas ou domínios.
- O dataset contém apenas avaliações polarizadas (positivas ou negativas), sem uma classe neutra.

## Como contribuir

Contribuições são bem-vindas, seja para corrigir algo, melhorar a documentação ou propor experimentos.

1. Faça um *fork* do repositório.
2. Crie uma branch para a sua alteração: `git checkout -b feat/minha-melhoria`.
3. Faça *commits* pequenos e descritivos.
4. Verifique que os scripts continuam rodando, sobretudo se mexer no pré-processamento ou no split, porque isso altera a comparação entre os modelos.
5. Abra um *Pull Request* explicando o que mudou e por quê.

Para bugs ou ideias, abra uma *issue* com o máximo de contexto possível (passos para reproduzir, mensagem de erro, versão do Python e das bibliotecas).

**Ideias para evoluir o projeto:**

- Análise de erros cruzando as previsões do baseline e do DistilBERT.
- Conjunto de validação separado e repetição do treino com várias sementes.
- Demo interativa (Streamlit ou Gradio).
- Comparação com outros modelos (por exemplo, BERT-base ou RoBERTa).

---

Feito por [João Pedro Garcia Telli](https://github.com/jptelli).