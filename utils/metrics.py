import pandas as pd
import re
import math
import numpy as np
from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_recall_curve, auc

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

def calculate_PR_AUC(df):
    y_true = np.concatenate([np.ones(len(df)), np.zeros(len(df))])
    y_pred = np.concatenate([df['p_pred_prob'], df['n_pred_prob']])
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    
    pr_auc = auc(recall, precision)
    return pr_auc

def calculate_mean_pr_auc(df):

    all_targets = pd.concat([df['p_target'], df['n_target']]).unique()
    
    pr_auc_scores = []
    
    for target in all_targets:
        positives = df[df['p_target'] == target]
        negatives = df[df['n_target'] == target]

        if len(positives) == 0:
            continue

        if len(negatives) == 0:
            continue
        
        y_true = np.concatenate([
            np.ones(len(positives)),
            np.zeros(len(negatives))
        ])
        
        y_score = np.concatenate([
            positives['p_pred_prob'].values,
            negatives['n_pred_prob'].values
        ])
        
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        
        pr_auc_scores.append(pr_auc)
    
    mean_pr_auc = np.mean(pr_auc_scores)
    
    return mean_pr_auc

def calculate_mean_roc_auc(df):

    all_targets = pd.concat([df['p_target'], df['n_target']]).unique()
    roc_auc_scores = []
    
    for target in all_targets:
        positives = df[df['p_target'] == target]
        negatives = df[df['n_target'] == target]

        if len(positives) == 0:
            continue

        if len(negatives) == 0:
            continue
        
        y_true = np.concatenate([
            np.ones(len(positives)),
            np.zeros(len(negatives))
        ])
        
        y_score = np.concatenate([
            positives['p_pred_prob'].values,
            negatives['n_pred_prob'].values
        ])
        
        roc_auc = roc_auc_score(y_true, y_score)
        roc_auc_scores.append(roc_auc)
    
    return np.mean(roc_auc_scores)

def calculate_mean_accuracy(df, threshold=0.5):

    all_targets = pd.concat([df['p_target'], df['n_target']]).unique()
    accuracy_scores = []
    
    for target in all_targets:
        positives = df[df['p_target'] == target]
        negatives = df[df['n_target'] == target]
        
        y_true = np.concatenate([
            np.ones(len(positives)),
            np.zeros(len(negatives))
        ])
        
        y_pred = np.concatenate([
            (positives['p_pred_prob'].values >= threshold).astype(int),
            (negatives['n_pred_prob'].values >= threshold).astype(int)
        ])
        
        accuracy = np.mean(y_true == y_pred)
        accuracy_scores.append(accuracy)
    
    return np.mean(accuracy_scores)

def calculate_mean_precision(df, threshold=0.5):

    all_targets = pd.concat([df['p_target'], df['n_target']]).unique()
    precision_scores = []
    
    for target in all_targets:
        positives = df[df['p_target'] == target]
        negatives = df[df['n_target'] == target]
        
        y_true = np.concatenate([
            np.ones(len(positives)),
            np.zeros(len(negatives))
        ])
        
        y_pred = np.concatenate([
            (positives['p_pred_prob'].values >= threshold).astype(int),
            (negatives['n_pred_prob'].values >= threshold).astype(int)
        ])
        
        if np.sum(y_pred) == 0:
            continue
            
        precision = precision_score(y_true, y_pred, zero_division=0)
        precision_scores.append(precision)
    
    return np.mean(precision_scores)

def calculate_mean_recall(df, threshold=0.5):

    all_targets = pd.concat([df['p_target'], df['n_target']]).unique()
    recall_scores = []
    
    for target in all_targets:
        positives = df[df['p_target'] == target]
        negatives = df[df['n_target'] == target]
        
        y_true = np.concatenate([
            np.ones(len(positives)),
            np.zeros(len(negatives))
        ])
        
        y_pred = np.concatenate([
            (positives['p_pred_prob'].values >= threshold).astype(int),
            (negatives['n_pred_prob'].values >= threshold).astype(int)
        ])
        
        if np.sum(y_true) == 0:
            continue
            
        recall = recall_score(y_true, y_pred, zero_division=0)
        recall_scores.append(recall)
    
    return np.mean(recall_scores)

def calculate_Accuracy(df):
    correct_pos = (df['p_pred_prob'] > 0.5).sum()
    
    correct_neg = (df['n_pred_prob'] < 0.5).sum()
    
    total_samples = 2 * len(df)
    
    return (correct_pos + correct_neg) / total_samples

def calculate_Precision(df):
    TP = (df['p_pred_prob'] > 0.5).sum()
    
    FP = (df['n_pred_prob'] >= 0.5).sum()
    
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0

def calculate_Recall(df):
    TP = (df['p_pred_prob'] > 0.5).sum()
    
    FN = (df['p_pred_prob'] <= 0.5).sum()
    
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0

def calculate_HR(group, max_n):

    combined = []
    for _, row in group.iterrows():
        combined.append((row['p_pred_prob'], 'p'))
        combined.append((row['n_pred_prob'], 'n'))
    
    combined_sorted = sorted(combined, key=lambda x: -x[0])
    sources = [item[1] for item in combined_sorted]
    
    metrics = {}
    for n in range(1, max_n + 1):
        try:
            y = sources[:n].count('p')
            metrics[f'N={n}'] = y / n if n != 0 else 0.0
        except:
            metrics[f'N={n}'] = np.nan
    
    return pd.Series(metrics)

def calculate_HR_df(df, max_n):

    results_list = []
    for herb_name, herb_df in df.groupby('herb'):
        result_series = calculate_HR(herb_df, max_n=max_n)
        result_df = result_series.to_frame().T.reset_index(drop=True)
        result_df.insert(0, 'herb', herb_name)
        results_list.append(result_df)
    combined_df = pd.concat(results_list, ignore_index=True)

    return combined_df

def calculate_NDCG(group, max_n):

    combined = []
    for _, row in group.iterrows():
        combined.append((row['p_pred_prob'], 'p'))
        combined.append((row['n_pred_prob'], 'n'))
    
    combined_sorted = sorted(combined, key=lambda x: -x[0])
    sorted_relevance = [1 if item[1] == 'p' else 0 for item in combined_sorted]

    ideal_relevance_list = [1] * max_n + [0] * max_n
    
    metrics = {}
    for n in range(1, max_n + 1):
        try:
            current_relevance = sorted_relevance[:n]
            # DCG
            dcg = sum( rel / math.log2(pos+1 + 1) for pos, rel in enumerate(current_relevance) )
            # IDCG
            ideal_relevance = ideal_relevance_list[:n]
            idcg = sum( rel / math.log2(pos+1 + 1) for pos, rel in enumerate(ideal_relevance) )
            # NDCG
            ndcg = dcg / idcg if idcg != 0 else 0.0
            metrics[f'N={n}'] = ndcg
        except:
            metrics[f'N={n}'] = np.nan
    
    return pd.Series(metrics)

def calculate_NDCG_df(df, max_n):
    results_list = []
    for herb_name, herb_df in df.groupby('herb'):
        result_series = calculate_NDCG(herb_df, max_n=max_n)
        result_df = result_series.to_frame().T.reset_index(drop=True)
        result_df.insert(0, 'herb', herb_name)
        results_list.append(result_df)
    combined_df = pd.concat(results_list, ignore_index=True)
    return combined_df

def calculate_ROC_AUC(df):

    y_true = np.concatenate([np.ones(len(df)), np.zeros(len(df))])
    y_pred_prob = np.concatenate([df['p_pred_prob'], df['n_pred_prob']])

    auc = roc_auc_score(y_true, y_pred_prob)

    return auc

def log_to_df(log_path):

    with open(log_path, 'r', encoding='utf-8') as f:
        log_lines = f.readlines()

    results = []
    current_params = None
    roc_auc_values = []

    param_pattern = re.compile(r'Hyperparameters and Settings: (.*)')
    roc_auc_pattern = re.compile(r'ROC-AUC: (\d+\.\d+)')

    for line in log_lines:
        param_match = param_pattern.search(line)
        if param_match:
            if current_params is not None and roc_auc_values:
                results.append({
                    **current_params,
                    'max_roc_auc': max(roc_auc_values)
                })
            
            params_str = param_match.group(1)
            param_dict = {}
            for item in params_str.split('|'):
                match = re.match(r'\s*([\w\s]+?)\s+([\dA-Za-z]+)\s*$', item.strip())
                if match:
                    key = match.group(1).strip().replace(' ', '_').lower()
                    value = match.group(2).strip()
                    param_dict[key] = value
            
            current_params = param_dict
            roc_auc_values = []
            continue

        roc_match = roc_auc_pattern.search(line)
        if roc_match and current_params is not None:
            roc_auc_values.append(float(roc_match.group(1)))

    if current_params is not None and roc_auc_values:
        results.append({
            **current_params,
            'max_roc_auc': max(roc_auc_values)
        })

    df = pd.DataFrame(results)
    
    numeric_columns = ['dim1', 'dim2', 'num_inner_layer', 'num_outer_layer', 
                      'valid_fold', 'test_fold', 'seed_of_data']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    column_order = [col for col in [
        'dim1', 'dim2', 'num_inner_layer', 'num_outer_layer',
        'valid_fold', 'test_fold', 'seed_of_data', 'mode', 'max_roc_auc'
    ] if col in df.columns]
    
    return df[column_order]