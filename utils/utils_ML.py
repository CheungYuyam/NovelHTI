import pandas as pd
import numpy as np

def batch_predict(df, embeddings_herb, embeddings_target, model):
    herb_idx = df['herb'].astype(int).values
    pos_target_idx = df['positive_target'].astype(int).values
    neg_target_idx = df['negative_target'].astype(int).values
    
    pos_features = np.hstack([
        embeddings_herb[herb_idx],
        embeddings_target[pos_target_idx]
    ])
    neg_features = np.hstack([
        embeddings_herb[herb_idx],
        embeddings_target[neg_target_idx]
    ])
    
    pos_probs = model.predict_proba(pos_features)[:, 1]
    neg_probs = model.predict_proba(neg_features)[:, 1]
    
    return pd.DataFrame({
        'herb': herb_idx,
        'positive_target': pos_target_idx,
        'negative_target': neg_target_idx,
        'p_pred_prob': pos_probs,
        'n_pred_prob': neg_probs
    })