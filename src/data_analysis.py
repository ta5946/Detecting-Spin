import pandas as pd
import matplotlib.pyplot as plt

DATA = pd.read_csv('../data/outcome_similarity_detection/train.tsv', sep='\t')
N = len(DATA)

DATA['label'].value_counts().plot(kind='bar')
plt.title(f"Outcome similarity label distribution (N = {N})")
plt.xlabel('label')
plt.ylabel('count')
plt.show()
