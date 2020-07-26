import pandas as pd
import json

sentence1 = []
sentence2 = []
label = []
with open('LCQMC_train.json', encoding='utf-8')as file:
    for line in file.readlines():
        dic = json.loads(line.strip())
        sentence1.append(dic.get('sentence1'))
        sentence2.append(dic.get('sentence2'))
        label.append(dic.get('gold_label'))

df = pd.DataFrame()
df['sentence1'] = sentence1
df['sentence2'] = sentence2
df['label'] = label

df.to_csv('LCQMC.csv', index=False, encoding='utf_8_sig')
