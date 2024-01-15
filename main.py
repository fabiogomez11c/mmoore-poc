from transformers import AutoModelForSequenceClassification, DistilBertTokenizer
from datasets import Dataset
from sklearn.metrics import accuracy_score

import torch
import pandas as pd

data = Dataset.from_csv('./synthetic_data.csv')
cats = {cat: i for i, cat in enumerate(set(data['category']))}
labels = [cats[i] for i in data['category']]
data = data.add_column(name='labels', column=labels)

print('done')