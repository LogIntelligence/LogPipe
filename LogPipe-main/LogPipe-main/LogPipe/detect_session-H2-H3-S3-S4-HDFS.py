import ast
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaModel, RobertaTokenizer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score
import csv
import torch
import random
import numpy as np
from collections import Counter, defaultdict
from collections import defaultdict
import pickle
from fractions import Fraction
from zhipuai import ZhipuAI
from openai import OpenAI
from google import genai
import json
def set_seed(seed):
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    random.seed(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  
your_api_key = ""
MAX_WINDOW_SIZE = 3 
MIN_SUPPORT_THRESHOLD = 2 


# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(43)
# dataloaddataset/preprocessed/HDFS/Modified_hdfs_log_structed_with_translated_event.csv
data_path = '/dataset/preprocessed/spark3/spark3.csv'

data = pd.read_csv(data_path)

tokenizer = RobertaTokenizer.from_pretrained('/pretrained_models/roberta-base')
roberta_model = RobertaModel.from_pretrained('/pretrained_models/roberta-base')
roberta_model = roberta_model.to(device)
embeddings_path = '/dataset/preprocessed/spark3/spark3_embeddings.npy'

#data['Status'] = data['Status'].apply(lambda x: 1 if x == 'Success' else 0) # this is for HDFS
# "Convert the 'Status' column to a numeric type."
data['Status'] = data['Status'].apply(lambda x: 1 if x == 0 else 0) # This is for haddop2/3,spark2/spark3

tokenizer = RobertaTokenizer.from_pretrained('/pretrained_models/roberta-base')
roberta_model = RobertaModel.from_pretrained('/pretrained_models/roberta-base')
roberta_model = roberta_model.to(device)
embeddings_path = '/dataset/preprocessed/spark3/spark3_embeddings.npy'

def generate_embeddings(texts, tokenizer, model, device, batch_size=256):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size].tolist()
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.concatenate(embeddings, axis=0)

# RoBERTa embedded generation (for similarity calculation)
if os.path.exists(embeddings_path):
    embeddings = np.load(embeddings_path)
else:
    embeddings = generate_embeddings(data['Content'], tokenizer, roberta_model, device)
    np.save(embeddings_path, embeddings)
k=[]
       
data['Embeddings'] = list(embeddings)
print("Embedding generation is complete.")
length=0
for i,eventlist in enumerate(data['EventList']):
    length+=len(ast.literal_eval(eventlist))
average_length=length/i
print('average',average_length)
train_data, test_data = data[:int(len(data) * 0.8)], data[int(len(data) * 0.9):]
#test_data=test_data[test_data['Status'] == 0]
test_data_Eventlist=test_data['EventList'].values

valid_data=data[int(len(data) * 0.8):int(len(data) * 0.9)]
valid_data_Eventlist=valid_data['EventList'].values
normal_train_data = train_data[train_data['Status'] == 1]
# Selecting only the rows in train_data where 'Status' is normal (1)
# Select samples from the training set
normal_indices = np.array(train_data[train_data['Status'] == 1].index)
anomalous_indices = np.array(train_data[train_data['Status'] == 0].index)

snippet_counts = defaultdict(int)
snippet_counts1 = defaultdict(int)

def generate_snippets(sequence, max_length):
    for length in range(1, max_length + 1):
        for i in range(len(sequence) - length + 1):
            snippet = tuple(sequence[i:i + length])
            snippet_counts[snippet] += 1

for event_list_str in normal_train_data['EventList']:
    event_list = ast.literal_eval(event_list_str)  
    generate_snippets(list(set(event_list)), MAX_WINDOW_SIZE)


frequent_snippets = {snippet for snippet, count in snippet_counts.items() if count >= MIN_SUPPORT_THRESHOLD}
#print(frequent_snippets[:10])

sent_template_file='/dataset/preprocessed/spark3/spark3_sentiment.pkl'
with open(sent_template_file,'rb') as f:
    sent_dir=pickle.load(f)
way='session'

def gnerate_dynamic_patterns(sequence_set, sequence,normal_snippets,k):
    if way=='session'and len(sequence)>1 and len(sequence_set)==1:
        return True,[] #[E1,E1,E1]
    sequence1=set(sequence_set)-normal_set
    #print(sequence1)
    if all(sent_dir[every_seq] in ['neutral', 'positive'] for every_seq in sequence1) :
        #print("i get neutral")
        #normal_set.add(every_seq for every_seq in sequence)
        return False,[]
    c=0
    m=[]
    for seq in sequence1:
        if sent_dir[seq] in ['neutral', 'positive']:
            m.append(seq)
    sequence_set=list(sequence1-set(m))
    save_snippet=[]
    for length in range(1, MAX_WINDOW_SIZE + 1):
        for i in range(len(sequence_set) - length + 1):
            snippet = tuple(sequence_set[i:i + length])
            if snippet  not in normal_snippets:
                c+=1
                save_snippet.append(snippet)
    return c>=k,save_snippet
# Ensure that the number of samples selected does not exceed the number actually available
max_normal_samples = min(2000, len(normal_indices))
max_anomalous_samples = min(2000, len(anomalous_indices))
print(max_anomalous_samples)

selected_normal_indices = np.random.choice(normal_indices, max_normal_samples, replace=False)
# get correct templates
selected_normal_indices_eventlist = train_data.loc[selected_normal_indices, 'EventList']
print(selected_normal_indices_eventlist)
normal_list = []
#for c in selected_normal_indices_eventlist:
for c in selected_normal_indices_eventlist:
    event_list = ast.literal_eval(c) if isinstance(c, str) else c  # Convert to list if necessary
    normal_list.extend(event_list)  # Flatten the lists into one
# Create a set of unique events
normal_set = set(normal_list)
print(f"Normal set length : {len(normal_set)}")
print(normal_set)
# Similarly for anomalous indices
selected_anomalous_indices = np.random.choice(anomalous_indices, max_anomalous_samples, replace=False)
selected_anomalous_indices_eventlist = train_data.loc[selected_anomalous_indices, 'EventList']
selected_anomalous_content=train_data.loc[selected_anomalous_indices, 'Content']

abnormal_list = []


# Similarly for anomalous indices
selected_anomalous_indices = np.random.choice(anomalous_indices, max_anomalous_samples, replace=False)
selected_anomalous_indices_eventlist = train_data.loc[selected_anomalous_indices, 'EventList']
selected_anomalous_content=train_data.loc[selected_anomalous_indices, 'Content']
abnormal_list = []
def process_anomalous_logs_with_zhipuAI(your_api_key):
    print("process_anomalous_logs method called")  
    client = ZhipuAI(api_key=your_api_key)  
    #abnormal_list = []
    m = 0
    count1=0
    count_token=0
    for c in selected_anomalous_indices_eventlist:
        event_list = ast.literal_eval(c) if isinstance(c, str) else c  
        abnormal_events = set(event_list) - normal_set
        
        if len(abnormal_events) == 1 or (len(abnormal_events) > 0 and len(set(abnormal_events)) == 1):
            abnormal_list.extend(abnormal_events)  
            m += 1
    for c, content in zip(selected_anomalous_indices_eventlist, selected_anomalous_content):
        event_list = ast.literal_eval(c) if isinstance(c, str) else c
        event_index_map = {templatedID: event for templatedID, event in zip(event_list,content.split("|"))}
        #print(event_index_map)
        abnormal_events = set(event_list) - normal_set
        abnormal_events_list=list(abnormal_events)
        istrue=0
        for abnormal_event in abnormal_events:
            #print(abnormal_event)
            if abnormal_event in abnormal_list:
                istrue=1
        #print(istrue)
        #if len(abnormal_events) > 1:
        if len(abnormal_events) > 1 and istrue==0:
            count1+=1
            print(f"Finding content corresponding to abnormal events: {abnormal_events}")
            relevant_contents = []
            '''for event in abnormal_events:
                if event in dict_tem_Ei:
                    relevant_contents.append(dict_tem_Ei[event])'''
            relevant_contents_dict={}
            for event in abnormal_events:
                content = event_index_map.get(event, "")
                relevant_contents_dict[event] = content
        
            if relevant_contents_dict:
                print(f"Relevant contents dictionary for analysis: {relevant_contents_dict}")  
                prompts = [
                    {
                        "role": "user",
                        "content": (
           "As a senior system log analyst, you need to understand how systems respond to critical errors. Please analyze the following sequence of log entries  and identify the most critical error event, considering both the initial error and the system's response to it.\n"
           "Analysis Requirements:\n"
           "1. Contextual Analysis: Consider the temporal relationships and causal associations between logs.\n"
           "2. Priority Assessment: Focus on the initial critical error and the system's recovery actions.\n"
           "- System crashes (such as kernel panic, segmentation fault)\n"
           "- Data corruption (such as disk failure, corrupted data)\n"
           "- CPU errors (such as frequency changes, clock ticks lost)\n"
           "- Resource exhaustion (such as OOM, disk full)\n"
           "- Service interruptions (such as  connection lost)\n"
           "3. Semantic Understanding: Understand the significance of the system's response to the error.\n"
           "4. False Positive Exclusion: Distinguish between actual errors and system recovery actions.\n"
           "5. Impact Assessment: Consider the potential impact of the error and the system's response on system stability.\n"        
           "I'll give you an example: {E1:content1 , E2:content2 , E3:content3 }. If you think content3 is a severe log, you should return the key corresponding to content3 (in this case, E3).\n"
            f"{relevant_contents_dict} is a dictionary containing log template IDs and log contents.\n"
            "From the dictionary I provide, you need to find and return the key of the most significant error log statement, without explaining why you chose it.\n"
            "If you genuinely believe that there are no anomalies in the logs provided, and they merely record normal system states, you should return 0.\n"
            "Please pay close attention: you are only allowed to return 0 or the key from the dictionary (in the format EXX, such as E1, E2, ...), and you must return the correct key without any mismatches between keys and values."
       )}
                ]
                response = client.chat.completions.create(
                    model="glm-4-flash",
                    messages=prompts,temperature=0,max_tokens=8000
                )
                #print(response)
                critical_log = response.choices[0].message.content 
                print('call cost ',response.usage.total_tokens)
                count_token+=response.usage.total_tokens
                if critical_log!='0':
                    print("critical", critical_log)                 
                    abnormal_list.append(critical_log)

    print(f"Total abnormal events analyzed: {m}")
    print('with no cache,',count1)
    print('all token',count_token)
process_anomalous_logs_with_zhipuAI(your_api_key)


probility_abnormal_set = set(abnormal_list)
print(f"Abnormal set length: {len(probility_abnormal_set)}")
print(probility_abnormal_set)
# get protition concerning about error
# get failure eventlist for hdfs
error_eventlist=[]
for c in selected_anomalous_indices_eventlist:
    if all(event in normal_set for event in ast.literal_eval(c)):     
        error_eventlist.append(c)
print(len(error_eventlist))
storage_indices = np.concatenate([selected_normal_indices, selected_anomalous_indices])



# Gets the corresponding storage embeddings and labels
#valid_storage_indices = np.array([idx for idx in selected_indices if idx in train_data.index])
valid_storage_indices = np.array([idx for idx in storage_indices if idx in train_data.index])
storage_embeddings = embeddings[valid_storage_indices]
storage_labels = train_data.loc[valid_storage_indices, 'Status'].values

# Find the few log coding blocks that are closest to the new log block
def find_most_similar(new_embedding, storage_embeddings, top_k=5):
    # The stored embeddings and new embeddings are converted to Tensor and moved to the GPU
    storage_embeddings_tensor = torch.tensor(storage_embeddings).to(device)
    new_embedding_tensor = torch.tensor(new_embedding).to(device)

    # Normalized embedding vector
    storage_embeddings_tensor = torch.nn.functional.normalize(storage_embeddings_tensor, p=2, dim=1)
    new_embedding_tensor = torch.nn.functional.normalize(new_embedding_tensor, p=2, dim=0)

    # Calculate the cosine similarity
    similarities = torch.matmul(storage_embeddings_tensor, new_embedding_tensor).cpu().numpy()
    
    # Find the most similar top_k
    top_k_indices = np.argsort(similarities)[-top_k:]
    return top_k_indices,similarities[top_k_indices]#similarities similarities[top_k_indices]


def find_top_k_probabilities(storage_embeddings, storage_labels, embeddings, top_k=100):
    probabilities = []
    all_similarities = []  
    for new_embedding in tqdm(embeddings, desc="Calculating probabilities"):
        top_k_indices, similarities = find_most_similar(new_embedding, storage_embeddings, top_k=top_k)
        most_similar_labels = storage_labels[top_k_indices]
        
        # Sort similarities for storage and analysis
        sorted_similarities = np.sort(similarities)  
        all_similarities.append(sorted_similarities)  
        m=100*(1-most_similar_labels.mean())
        probabilities.append(m)
    
    np.savetxt('prob.txt', probabilities)
    np.savetxt('similarities.txt', np.array(all_similarities))  



#combined_features_test = torch.tensor(np.stack([top_k_probabilities_test,reconstruction_errors_test], axis=1), dtype=torch.float32).to(device)
top_k_probabilities_valid = find_top_k_probabilities(storage_embeddings, storage_labels, np.array(valid_data['Embeddings']))
#---------------------------
top_k_probabilities_test = find_top_k_probabilities(storage_embeddings, storage_labels, np.array(test_data['Embeddings']))

#combined_features_test = torch.tensor(np.stack([top_k_probabilities_test,reconstruction_errors_test], axis=1), dtype=torch.float32).to(device)
valid_labels_test=valid_data['Status'].values
labels_test = test_data['Status'].values  # labels_test is now defined as a NumPy array



c = []
re_error_list=[]
for pro in top_k_probabilities_test:
    pro_cpu = pro 
    c.append(pro_cpu)
weighted_scores_test = np.array(c)  


def make_prediction(event_list_str, score, throld, threshold_for_sequence):
    event_list = ast.literal_eval(event_list_str)
    event_decisions = {}
    
    if any(event_list_str == error_event for error_event in error_eventlist):
        event_decisions['ErrorEventList'] = 0
    elif any(event in probility_abnormal_set for event in event_list):
            event_decisions['AbnormalSet'] = 0
    elif score>throld:
        event_decisions['ScoreThreshold'] = 0
    elif all(event in normal_set for event in event_list):
        #if detect_ratio1(event_list,normal_valid_pattern):
        event_decisions['NormalSet'] = 1
        #else:
            #event_decisions['no match 1:1 rationalSet'] = 0
    elif gnerate_dynamic_patterns(list(set(event_list)), event_list, frequent_snippets, threshold_for_sequence):
        event_decisions['FrequentSnippets'] = 0
    elif gnerate_dynamic_patterns(list(set(event_list)), event_list, frequent_snippets, threshold_for_sequence) == False:
        event_decisions['FrequentSnippets1'] = 1
    else:
        event_decisions['Default'] = 1

    final_decision = 1 if (event_decisions.get('NormalSet', 0) == 1 or 
                          event_decisions.get('Default', 0) == 1 or 
                          event_decisions.get('FrequentSnippets1', 0) == 1) else 0
                          
    return final_decision, event_decisions
def find_best_thresholds_on_valid(valid_data, valid_scores, valid_labels):
 
    best_f1 = 0
    best_seq_threshold = 2
    best_percentile = 94
    best_results = None
    

    for seq_threshold in range(1, 10): 
        for percentile in np.arange(97, 100.1,0.1):  
            predictions = []
            decisions = []
            
            current_sim_threshold = np.percentile(valid_scores, percentile)
            
            
            for event_list_str, score, label in zip(valid_data['EventList'], valid_scores, valid_labels):
                pred, decision_info = make_prediction(
                    event_list_str, 
                    score, 
                    current_sim_threshold,  
                    seq_threshold  
                )
                predictions.append(pred)
                decisions.append(decision_info)
            
            current_f1 = f1_score(valid_labels, predictions)
            print(current_f1)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_seq_threshold = seq_threshold
                best_percentile = percentile
                best_results = classification_report(valid_labels, predictions, digits=3)
    
    
    return best_seq_threshold, best_percentile

def determine_thresholds(average_length, valid_data, valid_scores, valid_labels):
    if average_length < 20:
        sequence_threshold = 1  
        best_f1 = 0
        best_percentile = 94
        best_results = None
        for percentile in np.arange(98.7 , 99.5,0.1):
            predictions = []
            decisions = []           
            for event_list_str, score, label in zip(valid_data['EventList'], valid_scores, valid_labels):
                pred, decision_info = make_prediction(
                    event_list_str, 
                    score, 
                    percentile,
                    sequence_threshold
                )
                predictions.append(pred)
                decisions.append(decision_info)        
            current_f1 = f1_score(valid_labels, predictions,pos_label=0)
            print(current_f1)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_percentile = percentile
                best_results = classification_report(valid_labels, predictions, digits=3)
    
        
    else:
        sequence_threshold, best_percentile = find_best_thresholds_on_valid(
            valid_data, valid_scores, valid_labels)
    
    return sequence_threshold, best_percentile


optimal_threshold, best_percentile = determine_thresholds(
    average_length,  
    valid_data, 
    top_k_probabilities_valid, 
    valid_labels_test
)

is_imblance=0
# judge the dataset is imblance？
count_0 = np.sum(valid_labels_test == 0)
count_1 = np.sum(valid_labels_test == 1)
if (count_0/count_1<0.015)and average_length>=20:
    optimal_threshold = optimal_threshold*2-1
else:
    optimal_threshold = optimal_threshold

dict_error_eventlist={}#cach1
dict_error_template={}# cache2
dict_normal_template={}#cache3
dict_frequence_item={}#cache4
dict_unfrequence_item={}#cache5

def make_prediction_test(event_list_str, score, throld, threshold_for_sequence,content,your_api_key):

    event_list = ast.literal_eval(event_list_str)
    event_decisions = {}
    decision_reasons = []
    content_str=content.split("|")
    id_content=zip(event_list,content_str)
    id_content_dict = dict(id_content)   
    always_True=1
    # retrieve Knowledge Base, and get extra information
    if any(event_list_str == error_event for error_event in error_eventlist):
        matching_error = next(error_event for error_event in error_eventlist if event_list_str == error_event)
        if matching_error in dict_error_eventlist:# first cache
            event_decisions['ErrorEventList'] = 0
            decision_reasons.append(f"Anomaly: Log sequence {dict_error_eventlist[matching_error]} found with cached explanation.")
        else:
            event_decisions['ErrorEventList'] = 0
            explanation_node_content = usellm_explain(content, [],0,your_api_key)
            dict_error_eventlist[matching_error]=explanation_node_content
            decision_reasons.append(f"Anomaly explain:  {dict_error_eventlist[matching_error]} ")
    elif any(event in probility_abnormal_set for event in event_list):
            abnormal_events_with_index = [(i, event) for i, event in enumerate(event_list) if event in probility_abnormal_set]
            abnormal_events = [event for i, event in abnormal_events_with_index] 
            indices = [i for i, event in abnormal_events_with_index] 
            abnormal_events=sorted(list(set(abnormal_events)))

            if str(abnormal_events) in dict_error_template:# second cache
                event_decisions['LLM think abnormal']=0
                #decision_reasons.append(dict_error_template[str(abnormal_events)])
                decision_reasons.append(f"Log localization: The abnormal logs are at positions {indices}."+dict_error_template[str(abnormal_events)])
            else:
                content_2_llm=[]
                for c in abnormal_events:
                    content_2_llm.append(id_content_dict[c])
                content_2_llm.append("The log sequence you need me to judge contains some unique fault log events.")
                Status,explain = usellm_explain(content, content_2_llm,1,your_api_key)
                print(Status,explain)
                if Status=='abnormal':
                    event_decisions['LLM think abnormal']=0
                    dict_error_template[str(abnormal_events)]=explain
                    #decision_reasons.append(explain)
                    decision_reasons.append(f"Log localization: The abnormal logs are at positions {indices}."+"LLM explains:"+explain)
                else:
                    event_decisions['LLM think normal']=1
    elif score>throld:
        prompts=f'The similarity between this log statement and the anomalous logs stored in the database has reached{score}%,and reached an abnormal score'
        Status2,explain2=usellm_explain(content,prompts,1,your_api_key)
        if Status2=='abnormal':
            event_decisions['LLM think abnormal']=0
            print(Status2,explain2)
            decision_reasons.append(explain2)  
            # update probility_abnormal_set
            if len(set(event_list))==1:
                judge=usellm_explain(content_str[0],[],3,your_api_key)
                print("The large model starts assisting in detecting discrete anomalous event patterns.")

                print(judge)
                if judge=='abnormal':
                    probility_abnormal_set.add(event_list[0])
        else:
            event_decisions['LLM think normal']=1
    elif all(event in normal_set for event in event_list):
        normal_events = [event for event in event_list if event in normal_set]
        normal_events=sorted(list(set(normal_events)))
        #print(normal_events)
        if str(normal_events) in dict_normal_template:# thrid cache
            event_decisions['LLM think normal']=1
            #print(dict_normal_template)
            decision_reasons.append(dict_normal_template[str(normal_events)])
        else:
            prompts="Perhaps this log contains many error words, but please do not judge normality or abnormality based on keywords,I clearly tell you that every log entry has been rigorously mined and considered normal, with a credibility of 99.99%,you can trust every single log is normal fully."

            Status1,explain1=usellm_explain(content,prompts,1,your_api_key)
            print(Status1,explain1)
            if Status1=='normal':
                dict_normal_template[str(normal_events)]=explain1
                event_decisions['LLM think normal']=1
            else:
                if Status1=='abnormal':
                    event_decisions['LLM think abnormal']=0
                    decision_reasons.append(explain1)
                else:
                    dict_normal_template[str(normal_events)]=explain1
                    event_decisions['LLM think normal']=1     
    elif always_True:
        is_anomalous, snippet = gnerate_dynamic_patterns(list(set(event_list)), event_list, frequent_snippets, threshold_for_sequence)
        if is_anomalous==True:# indicate abnormal_slice,we can call llm for judge
            if str(sorted(snippet)) in dict_unfrequence_item:# cache hit again
                event_decisions['LLM think abnormal']=0
                decision_reasons.append(dict_unfrequence_item[str(sorted(snippet))])
            else:
                prompts='In this log sequence, there are non-frequent segments, their count reached the anomaly threshold for non-frequent items,which is considered quite reliable.'
                Status3,explain3=usellm_explain(content,prompts,1,your_api_key)
                if Status3=='abnormal':
                    event_decisions['LLM think abnormal']=0
                    dict_unfrequence_item[str(sorted(snippet))]=explain3
                    decision_reasons.append(explain3)

                else:
                    event_decisions['LLM think normal']=1
        else:
            if str(sorted(snippet)) in dict_frequence_item:# cache hit again
                event_decisions['LLM think normal']=1
                decision_reasons.append(dict_frequence_item[str(sorted(snippet))])
            else:
                prompts='In this log sequence, there are some non-frequent segments, but their count has not yet reached the anomaly threshold for non-frequent items,which is considered quite reliable,there is a very high likelihood that this log sequence is normal.Just because a log sequence contains words like error or fail(or some Recoverable minor issue(network intrrupt or memory issue)) does not mean it is abnormal. In many cases, such logs are actually normal behavior, as each system has its own criteria for defining anomalies.Declaring log sequence as abnormal requires extreme caution, as false positives can be very harmful.'
                Status4,explain4=usellm_explain(content,prompts,1,your_api_key)
                print(Status4,explain4)
                if Status4=='normal':
                    event_decisions['LLM think normal']=1
                    dict_frequence_item[str(sorted(snippet))]=explain4
                else:
                    if Status4=='abnormal':
                        event_decisions['LLM think abnormal']=0
                        decision_reasons.append(explain4)
                    else:
                        event_decisions['LLM think normal']=1
                        dict_frequence_item[str(sorted(snippet))]=explain4
    final_decision = 1 if (event_decisions.get('LLM think normal') == 1) else 0
                          
    return final_decision, event_decisions,decision_reasons
test_predictions = []
test_decisions = []
misclassified_events = []
sum_token=0
def usellm_explain(raw_content,extra_information,k2,your_api_key):
    global sum_token
    #time.sleep(5)
    tools = [
    {
        "type": "function",
        "function": {
            "name": "analyze_log_anomaly",
            "description": "Analyzes a log sequence to determine if it's normal or abnormal and provides a brief explanation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["normal", "abnormal"],
                        "description": "The status of the log sequence, must be either 'normal' (normal operation) or 'abnormal' (error condition)."
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Incorporate log details into a concise explanation (max 100 tokens)."
                    }
                },
                "required": ["status", "explanation"]
            }
        }
    }
]
    if k2==0:# it intend to explain log explaination for abnormal sequence(where selected from training log sequence),so there are not necessarily to call llm for log anomaly
        client = ZhipuAI(api_key=your_api_key)
        response = client.chat.completions.create(
        # model='Pro/deepseek-ai/DeepSeek-R1',
        model="glm-4-flash",
        messages=[
            {'role': 'user', 
            'content': "You are a specialist in log anomaly interpretation. Next, I will provide you with the original log anomaly sequence and some suspicious log fragments that our method has helped you locate (considered trustworthy). Please analyze why this log sequence is abnormal and propose solutions based on the original log sequence and the suspicious log fragments."
            f"This is the exception log you need to explain,{raw_content}\n"
            f"This is a suspicious fragment.{extra_information}\n"
            "please pay attention,Please explain the causes of exceptions and their solutions within 100 tokens."
            },
        ],max_tokens=8000,
        stream=False)
        print('call cost ',response.usage.total_tokens)
        sum_token+=response.usage.total_tokens
        return response.choices[0].message.content
    if k2==1:# it intend to do log anomaly and explain
        client = ZhipuAI(api_key=your_api_key)
        response = client.chat.completions.create(
        # model='Pro/deepseek-ai/DeepSeek-R1',
        model="glm-4-flash",
        messages=[
            {'role': 'user', 
            'content': "You are a specialist in log anomaly detection. Next, I will provide you with the original log sequence and some other information that can helped you judge log sequence's status."
            f"This is the Log sequence you need to judge and explain,{raw_content}\n"
            f"This is a other information to help you to judge.{extra_information}\n"
            "Please analyze this log and use the analyze_log_anomaly function to return the status and explanation in a structured format."
            },
        ],tools=tools,max_tokens=8000,
        stream=False)
        response_message = response.choices[0].message
        status = None
        explanation = None
        # Check if the model decided to call a function
        if response_message.tool_calls:
    # Since there might be multiple tool calls, iterate through them (though in this case, you likely expect only one)
            tool_call = response_message.tool_calls[0]
            function_name = tool_call.function.name
            function_arguments_str = tool_call.function.arguments
            function_arguments = json.loads(function_arguments_str)

            # Now you can access the arguments using .get()
            status = function_arguments.get("status")
            explanation = function_arguments.get("explanation")

            #print(f"Function Name: {function_name}")
            #print(f"Status: {status}")
            #print(f"Explanation: {explanation}")

        else:
            print("No tool call made.")
            print(response_message.content)
        print('call cost ',response.usage.total_tokens)
        sum_token+=response.usage.total_tokens
        return status, explanation

    if k2==2:# all log seq has normal log events,we can generate new prompts for LLM
        client = ZhipuAI(api_key=your_api_key)
        response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {'role': 'user',
             'content': "You are an expert in log anomaly. I will provide you with a sequence of logs."
             f"This is the log sequence you need to analyze: {raw_content}\n"
             "Perhaps this log contains many error words, but please do not judge normality or abnormality based on keywords. I clearly tell you that every log\n" "entry has been rigorously mined and considered normal, with a credibility of 99.99%.\n"
             "Please analyze this log and use the analyze_log_anomaly function to return the status and explanation in a structured format."
             },
        ],
        tools=tools,
        max_tokens=8000,
        stream=False
    )
        response_message = response.choices[0].message
        status = None
        explanation = None
        # Check if the model decided to call a function
        if response_message.tool_calls:
    # Since there might be multiple tool calls, iterate through them (though in this case, you likely expect only one)
            tool_call = response_message.tool_calls[0]
            function_name = tool_call.function.name
            function_arguments_str = tool_call.function.arguments
            function_arguments = json.loads(function_arguments_str)

            # Now you can access the arguments using .get()
            status = function_arguments.get("status")
            explanation = function_arguments.get("explanation")
            #print(f"Function Name: {function_name}")
            #print(f"Status: {status}")
            #print(f"Explanation: {explanation}")
        else:
            print("No tool call made.")
            print(response_message.content)
        print('call cost ',response.usage.total_tokens)
        sum_token+=response.usage.total_tokens
        return status, explanation
    if k2==3:# all log seq has normal log events,we can generate new prompts for LLM
        client = ZhipuAI(api_key=your_api_key)
        response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {'role': 'user',
             'content': "You are an expert in log anomaly. I will provide you with a single logs."
             f"This is the log  you need to judge: {raw_content}\n"
             "please return normal or abnormal directly,without any other words"
             ""
             },
        ],
        max_tokens=8000,
        stream=False
    )
        print('call cost ',response.usage.total_tokens)
        sum_token+=response.usage.total_tokens
        return response.choices[0].message.content
i=0
for event_list_str, score, label,content in zip(test_data['EventList'], c, labels_test,test_data['Content']):
    pred, decision_info,reason = make_prediction_test(event_list_str, score, best_percentile, optimal_threshold,content,your_api_key)
    test_predictions.append(pred)
    test_decisions.append(decision_info)
    print("This is the", i, "time")
    i+=1
    if pred==0:
        print(reason)
    if pred != label:
        misclassified_events.append((event_list_str, label, pred, decision_info))
print("token sum",sum_token)

print(classification_report(labels_test, test_predictions, digits=3))

