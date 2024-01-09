# Análise de Avaliações de Jogos na Steam

### As avaliações de um produto são refletidas nos comentários ou avaliações. Na análise de sentimentos avançada para o sistema de avaliação de produtos, os comentários são analisados para detectar sentimentos ocultos. Neste caso, serão analisados comentários sobre experiências de jogos na plataforma Steam.

### A análise de sentimentos usando aprendizado de máquina se apoia em um banco de dados composto por palavras baseadas em sentimentos, que incluem termos positivos e negativos.

### As palavras usadas na seção de comentários dos usuários são comparadas às palavras contidas no banco de dados, e uma avaliação é realizada. Ao comparar com as palavras-chave no banco de dados, o sistema especifica se o jogo é bom, ruim ou péssimo.

# Instalação e Importação de Bibliotecas Necessárias para NLP e Análise de Sentimentos

1. Instalando e importando as bibliotecas necessárias para Processamento de Linguagem Natural (NLP) e análise de sentimentos.

```bash
pip install stopwords
pip install flair
pip install nltk
pip install swifter
```

```python
import pandas as pd
import numpy as np
import flair
from flair.data import Sentence
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random as rn
import seaborn as sns
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter
from PIL import Image
pd.options.display.max_rows = None
```

2. Importando o dataset


### Pegando 10% do dataset para uma avaliação inicial

```python
rn.seed(a=40)
q = 0.1  
review_val = pd.read_csv('/steam-reviews/dataset.csv', skiprows=lambda i: i > 0 and rn.random() > q)
review_val.head()
review.info()
review.review_text = review.review_text.astype('str')
```

3. Limpando dados para processamento e análise

```python
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
import nltk
nltk.download('stopwords')
from nltk.stem import PorterStemmer
```
### Aqui, verificamos a presença de valores nulos e removemos duplicatas do dataset `review`, mantendo a primeira ocorrência.

```python
review.isnull().sum()
review = review.drop_duplicates(keep='first')
review.shape
```

### Estas funções abaixo são utilizadas para limpar e formatar o texto das avaliações, removendo elementos indesejados como hiperlinks, emojis e números.

```python
import re

def clean(raw):
    """ Remove hyperlinks and markup """
    result = re.sub("<[a][^>]*>(.+?)</[a]>", 'Link.', raw)
    result = re.sub('&gt;', "", result)
    result = re.sub('&#x27;', "'", result)
    result = re.sub('&quot;', '"', result)
    result = re.sub('&#x2F;', ' ', result)
    result = re.sub('<p>', ' ', result)
    result = re.sub('</i>', '', result)
    result = re.sub('&#62;', '', result)
    result = re.sub('<i>', ' ', result)
    result = re.sub("\n", '', result)
    return result

def remove_num(texts):
    """ Remove numbers from text """
    output = re.sub(r'\d+', '', texts)
    return output

def deEmojify(x):
    """ Remove emojis from text """
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'', x)

def unify_whitespaces(x):
    """ Unify multiple white spaces into a single space """
    cleaned_string = re.sub(' +', ' ', x)
    return cleaned_string

def remove_symbols(x):
    """ Remove special symbols, keeping letters, numbers, and some punctuation """
    cleaned_string = re.sub(r"[^a-zA-Z0-9?!.,]+", ' ', x)
    return cleaned_string

def remove_punctuation(text):
    """ Remove specific punctuation from text """
    final = "".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"',','))
    return final
```
### Função adicional para remover stopwords do texto das avaliações.

```python
stop = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemma = WordNetLemmatizer()

def remove_stopword(text):
    """ Remove stopwords from text """
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)
```

### Fazendo steamming para normalizar as palavras 

```python
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
def Stemming(text):
   stem=[]
   stopword = stopwords.words('english')
   snowball_stemmer = SnowballStemmer('english')
   word_tokens = nltk.word_tokenize(text)
   stemmed_word = [snowball_stemmer.stem(word) for word in word_tokens]
   stem=' '.join(stemmed_word)
   return stem
 ```

 ### Por fim aplicar todas as funções de limpeza que a gente definiu anteriormente

 ```python
 def cleaning(df,review):
    df[review] = df[review].apply(clean)
    df[review] = df[review].apply(deEmojify)
    df[review] = df[review].str.lower()
    df[review] = df[review].apply(remove_num)
    df[review] = df[review].apply(remove_symbols)
    df[review] = df[review].apply(remove_punctuation)
    df[review] = df[review].apply(remove_stopword)
    df[review] = df[review].apply(unify_whitespaces)
    df[review] = df[review].apply(Stemming)
```

```python
cleaning(review,'review_text')
review[['review_text']].head(20)

review_vis2 = review.copy()
```










