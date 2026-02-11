import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))

from models.NovelHTI import MainModel as Model

from utils.utils import train_model
from utils.set_seed_and_reproducibility import set_seed_and_reproducibility
from utils.get_kg import get_kg
from utils.data import prepare_dataloaders
from utils.logging import set_log

import pandas as pd
import json
import logging

with open('../configs/NovelHTI.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

random_seed = config['train']['random_seed']
set_seed_and_reproducibility(random_seed)
set_log(config['train']['log_folder'] + '/' + config['train']['model_name'] + '.log')

device = config['train']['device']
print(f'Using device: {device}')

mode = "F" if config['train']['mode'] == 'by_fold' else "R"

for seed in [10, 20, 30]:

    for val, test in [[0,1]]:

        for dim1 in [2,4,8]:

            df = pd.read_csv(f"{config['train']['data_folder']}/seed={seed}.csv")
            train_loader, val_loader, test_loader = prepare_dataloaders(df, batch_size=config['train']['batch_size'], method=mode, 
                                                                        val_fold=val, test_fold=test, random_state=random_seed)
            
            kg_df = pd.read_csv(config['model']['kg_path'])

            kg1 = kg_df[kg_df['interaction']<=11].copy()
            kg2 = kg_df[kg_df['interaction']>=12].copy()
            
            kg_graph1 = get_kg(kg1).to(device)
            kg_graph2 = get_kg(kg2).to(device)

            herbs_data_path = config['model']['herb_feature_path']
            map1_path = config['model']['mapping_paths']['tcm_mm']
            map2_path = config['model']['mapping_paths']['mm_disease']
            map3_path = config['model']['mapping_paths']['disease_target']

            # feature_dim1 = config['model']['feature_dim1']
            feature_dim1 = dim1
            feature_dim2 = config['model']['feature_dim2']

            num_inner_layer = config['model']['num_inner_layer']
            num_outer_layer = config['model']['num_outer_layer']

            log_msg = (
                f"Hyperparameters and Settings: "
                f"Dim1 {feature_dim1:02d} | "
                f"Dim2 {feature_dim2:02d} | "
                f"Num_inner_layer {num_inner_layer:01d} | "
                f"Num_outer_layer {num_outer_layer:01d} | "
                f"Valid fold {val:01d} | "
                f"Test fold {test:01d} | "
                f"Seed of data {seed:02d} | "
                f"Mode {mode}"
            )
            logging.info(log_msg)

            df_path = config['train']['log_folder']
            model_name = config['train']['model_name']
            settings = f"m_{mode}_d1_{feature_dim1}_d2_{feature_dim2}_li_{num_inner_layer}_lo_{num_outer_layer}_{seed}_v_{val}_t_{test}"

            save_file = f'{df_path}/{model_name}_{settings}'

            model = Model(herbs_data_path, map1_path, map2_path, map3_path, 
                        kg_graph1, kg_graph2, feature_dim1=feature_dim1, feature_dim2=feature_dim2, 
                        num_inner_layer=num_inner_layer, num_outer_layer=num_outer_layer,
                        device=device).to(device)

            base_model = train_model(model, train_loader, val_loader, epochs=config['train']['epochs'], 
                                    save_path=save_file, patience=config['train']['patience'],
                                    device=device)

            result_df, auc = base_model.evaluate(test_loader, mode='test')

            result_df.to_csv(f"{save_file}.csv", index=None)