import torch
import logging
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

class BaseModel:
    def __init__(self, model, patience=2, device='cuda:0', lr=0.001):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            patience=patience
        )

        self.device = device
        
    def train_step(self, batch):
        herb, p_target, n_target = batch
        herb, p_target, n_target = herb.to(self.device), p_target.to(self.device), n_target.to(self.device)

        p_preds, n_preds = self.model(herb, p_target, n_target)

        p_loss = self.loss_fn(p_preds, torch.ones_like (p_preds))
        n_loss = self.loss_fn(n_preds, torch.zeros_like(n_preds))
        total_loss = p_loss + n_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return p_loss.item(), n_loss.item()
    
    def evaluate(self, dataloader, mode='valid'):
        self.model.eval()
        p_total_losses = 0.0
        n_total_losses = 0.0
        all_probs = []
        all_labels = []
        
        herbs = []
        p_targets = []
        n_targets = []
        p_pred_probs = []
        n_pred_probs = []

        with torch.no_grad():
            loop = tqdm(dataloader, desc="Inference", disable=(mode!='test'))
            for batch in loop:
                herb, p_target, n_target = batch
                herb = herb.to(self.device)
                p_target = p_target.to(self.device)
                n_target = n_target.to(self.device)
                
                p_preds, n_preds = self.model(herb, p_target, n_target)
                
                p_loss = self.loss_fn(p_preds, torch.ones_like(p_preds))
                n_loss = self.loss_fn(n_preds, torch.zeros_like(n_preds))
                p_total_losses += p_loss.item()
                n_total_losses += n_loss.item()
                
                p_probs = torch.sigmoid(p_preds).cpu().numpy().flatten()
                n_probs = torch.sigmoid(n_preds).cpu().numpy().flatten()
                all_probs.extend(p_probs)
                all_probs.extend(n_probs)
                all_labels.extend([1] * len(p_probs))
                all_labels.extend([0] * len(n_probs))
                
                if mode == 'test':
                    herbs.extend(herb.cpu().numpy().flatten().tolist())
                    p_targets.extend(p_target.cpu().numpy().flatten().tolist())
                    n_targets.extend(n_target.cpu().numpy().flatten().tolist())
                    p_pred_probs.extend(p_probs.tolist())
                    n_pred_probs.extend(n_probs.tolist())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        auc = roc_auc_score(all_labels, all_probs)
        
        result = (
            p_total_losses / len(dataloader),
            n_total_losses / len(dataloader),
            auc
        )
        
        if mode == 'test':
            df = pd.DataFrame({
                'herb': herbs,
                'p_target': p_targets,
                'n_target': n_targets,
                'p_pred_prob': p_pred_probs,
                'n_pred_prob': n_pred_probs
            })
            return df, auc
        else:
            return result


def train_model(model, train_loader, val_loader, epochs=50, routine=1, 
                patience=2, 
                save_path=None, device='cuda:0'):
    model = model.to(device)
    base_model = BaseModel(model, patience=patience)
    
    best_val_auc = 0.0

    for epoch in range(epochs):
        model.train()
        p_total, n_total = 0.0, 0.0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1:03d}/{epochs:03d}'):
            p_loss, n_loss = base_model.train_step(batch)
            p_total += p_loss
            n_total += n_loss
        
        train_p_loss = p_total / len(train_loader)
        train_n_loss = n_total / len(train_loader)
        
        val_p_loss, val_n_loss, auc = base_model.evaluate(val_loader)

        base_model.scheduler.step(auc)
        current_lr = base_model.optimizer.param_groups[0]['lr']
        
        log_msg = (
            f"Routine {routine:02d} | "
            f"Epoch {epoch+1:03d}/{epochs:03d} | "
            f"LR: {current_lr:.0e} | "
            f"Train Loss: P={train_p_loss:.4f} N={train_n_loss:.4f} | "
            f"Val Loss: P={val_p_loss:.4f} N={val_n_loss:.4f} | "
            f"ROC-AUC: {auc:.4f} "
        )
        logging.info(log_msg)
        
        if auc > best_val_auc:
            best_val_auc = auc
            torch.save(model.state_dict(), f'{save_path}.pth')

        if current_lr <= 0.00001:
            
            return base_model
    
    return base_model