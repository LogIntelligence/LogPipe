import numpy as np
import pandas as pd
import pickle
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from scipy.special import softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Preprocess text
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

MODEL = '/pretrained_models/sentiment-analysis'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)  # 移动模型到GPU

data_file = '/dataset/preprocessed/BGL/BGL.csv'
log_template_data = pd.read_csv(data_file, low_memory=False, memory_map=True)

# Create dictionary
dict_tem_Ei = log_template_data.groupby('TemplateId')['Content'].first().to_dict()
print(dict_tem_Ei)

# Save dictionary to text file
#output_file_txt = '/dataset/preprocessed/Spirit/Spirit_sentiment.txt'
#with open(output_file_txt, 'w') as file:
    #for key, value in dict_tem_Ei.items():
        #file.write(f"{key}: {value}\n")
#print(f"Dictionary saved to {output_file_txt}")

Eid_sentiment = {}

try:
    for templatedid, content in dict_tem_Ei.items():
        text = content
        # text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        encoded_input = {key: val.to(device) for key, val in encoded_input.items()} 
        output = model(**encoded_input)
        scores = output.logits[0].detach().cpu().numpy() 
        scores = softmax(scores)

        # 计算排名
        ranking = np.argsort(scores)
        print(ranking)
        Eid_sentiment[templatedid] = config.id2label[ranking[-1]]
        print(Eid_sentiment[templatedid])

except Exception as e:
    print(f"Error processing {templatedid}: {e}")

print(Eid_sentiment)

# 保存结果
with open('BGL_sentiment.pkl', 'wb') as f:
    pickle.dump(Eid_sentiment, f)