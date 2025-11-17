# Knowledge-Augmented Log Anomaly Detection with Large Language Models
LogPipe​ is a framework that enhances the effectiveness of LLMs in anomaly detection through knowledge base augmentation. By providing patterns to the LLM via a knowledge base, it significantly improves the model's performance. Additionally, LogPipe incorporates caching capabilities, which reduces the operational costs of the LLM. Furthermore, LogPipe offers fault localization functionality, thereby enhancing interpretability.
# Revised Content
## Sec-1 A more comprehensive comparison (including KNN, NeuralLog, DeepLog, and LogAnomaly)
![LogPipe](https://github.com/a13382735176/LogPipe/blob/main/compare.png)
The experimental design, including data partitioning, for the newly incorporated baselines KNN, NeuralLog, DeepLog, and LogAnomaly was kept consistent with that of LogPipe. Furthermore, we strictly followed the hyperparameter configurations specified in the official repository to ensure fair reproduction. The experimental results indicate that even with these additional baselines, LogPipe consistently maintains superior performance.
## Sec-2 Performance Comparison with and without LLM on the BGL and Spirit
<p align="center">
  <img src="https://github.com/a13382735176/LogPipe/blob/main/different.png" alt="LogPipe" width="80%">
</p>
We conducted experiments on BGL using the 100L log blocks as in the original LogPipe paper. Two settings were evaluated: (1) using only the extra information without involving the LLM for anomaly detection, and (2) using the extra information to guide the LLM in performing anomaly detection. The first setting resulted in a 5.1% decrease compared to the second, indicating the effectiveness of the LLM within LogPipe.

## Sec-3 Performance Comparison with Differnent Hyperparameters 
![LogPipe](https://github.com/a13382735176/LogPipe/blob/main/f1_score_robustness_final.png)
We performed a grid search on the validation set to determine the optimal hyperparameter combination, including the anomaly score and the dynamic pattern threshold T. To assess robustness, we tested two sub-optimal settings by perturbing the optimal values by ±1: (anomaly score − 1, T − 1) and (anomaly score + 1, T + 1). Green, Blue, and Yellow bars in the figure correspond to Threshold −1, Optimal Threshold, and Threshold +1, respectively. The results show that LogPipe's F1-score fluctuates by no more than 3% relative to the optimal threshold, indicating stable performance across the tested ranges.
## Sec-4The impact of Different LLMs on LogPipe
![LogPipe](https://github.com/a13382735176/LogPipe/blob/main/llm_f1_comparison_final.png)
We evaluated DeepSeek, GLM4-2.0-Flash, and Qwen3-14B within the LogPipe framework. The experimental results demonstrate that LogPipe is robust across different LLMs and does not rely on GLM4-2.0-Flash from the original LogPipe paper.

## Sec-5 The impact of different sequence lengths on the Thunderbird dataset
<p align="center">
  <img src="https://github.com/a13382735176/LogPipe/blob/main/differnent_size.png" alt="LogPipe" width="80%">
</p>
The experiments show that the F1 score on Thunderbird is 0.968 under the 100L setting and 0.980 under the 200L setting, indicating only a minor difference.

## Overview of LogPipe
![LogPipe ](https://github.com/a13382735176/LogPipe/blob/main/Overview%20of%20LogPipe.png )

## Step 1: Get api key
 You can get api key from https://bigmodel.cn/.
## Step 2: Install the required packages
The code is implemented in Python==3.11. To install the required packages, run the following command:
 ```bash
pip install -r requirements.txt

 ```
## Step 3: Download datasets and pretrained model parameters

You can download the dataset and the pretrained model parameters from the following anonymized OSF link:
🔗 [https://osf.io/w8sf2/?view_only=7a7b5d9dfc3748d6848875d757c1cae8](https://osf.io/w8sf2/?view_only=7a7b5d9dfc3748d6848875d757c1cae8)

## Step4 Get the sentiment of log and slice logs into block
To get the sentiment of log, run the following command:
```bash
python "\LogPipe\preprocess\get_log_event_sentiment.py"
```
# Example usage:
```bash
data_file = '/dataset/preprocessed/BGL/BGL.csv' (you can modify the data_file to specify the dataset you want to run）
with open('BGL_sentiment.pkl', 'wb') as f: (
    pickle.dump(Eid_sentiment, f)
then we get a BGL_sentiment.pkl 
```
To slice logs into block:
```bash
python"\LogPipe\preprocess\slice_logs_blcok.py" 
```
# Example usage:
```bash
file_path = 'dataset/preprocessed/BGL/BGL.csv'

output_path = 'dataset/preprocessed/BGL/100l_BGL.csv'
L = 100  # The desired block length
process_log_to_csv(file_path, output_path, L)
```
To reduce your workload, we have already included BGL_sentiment.pkl and pre-sliced log blocks in the dataset, which can be used directly.
## Step5 Anomaly detection
```bash
Run  "logpipe\detect_log_Bgl-Spirit-Thunderbird.py"
Deatils:
your_api_key = "" (Can get from step1)
data_path = '/dataset/preprocessed/100L_BGL.csv' 
sent_template_file='/dataset/preprocessed/BGL/BGL_sentiment.pkl'
```
```bash
 For Session-based datsesets, please Run "\LogPipe\detect_session-H2-H3-S3-S4-HDFS.py"
```
## Results of Anomaly Detection
The effect of anomaly detection is as follows.

![LogPipe](https://github.com/a13382735176/LogPipe/blob/main/LogPipe/Results.png)

## Cost
![LogPipe](https://github.com/a13382735176/LogPipe/blob/main/Cost.png)


