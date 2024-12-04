def get_prediction0(model_path, model_weights):
    import numpy as np
    import pandas as pd
    import os
    from tqdm import tqdm
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    import pickle
    import time
    from torch.cuda.amp import autocast, GradScaler
    import random
    from sklearn.metrics import roc_auc_score, log_loss
    import re

    class DAIGTDataset(Dataset):
        def __init__(self, text_list, tokenizer, max_len):
            self.text_list=text_list
            self.tokenizer=tokenizer
            self.max_len=max_len
        def __len__(self):
            return len(self.text_list)
        def __getitem__(self, index):
            text = self.text_list[index]
            tokenized = self.tokenizer(text=text,
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len,
                                       return_tensors='pt')
            return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze()

    class DAIGTModel(nn.Module):
        def __init__(self, model_path, config, tokenizer, pretrained=False):
            super().__init__()
            if pretrained:
                self.model = AutoModel.from_pretrained(model_path, config=config)
            else:
                self.model = AutoModel.from_config(config)
            self.classifier = nn.Linear(config.hidden_size, 1)  
            #self.model.gradient_checkpointing_enable()    
        def forward_features(self, input_ids, attention_mask=None):
            outputs = self.model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            embeddings = sum_embeddings / sum_mask
            return embeddings
        def forward(self, input_ids, attention_mask):
            embeddings = self.forward_features(input_ids, attention_mask)
            logits = self.classifier(embeddings)
            return logits
        
    df = pd.read_csv('../input/llm-detect-ai-generated-text/test_essays.csv')
    id_list = df['id'].values
    text_list = df['text'].values

    max_len = 768
    batch_size = 16

    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = DAIGTModel(model_path, config, tokenizer, pretrained=False)
    model.load_state_dict(torch.load(model_weights))
    model.cuda()
    model.eval()

    test_datagen = DAIGTDataset(text_list, tokenizer, max_len)
    test_generator = DataLoader(dataset=test_datagen,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=False)
    
    pred_prob = np.zeros((len(text_list), ), dtype=np.float32)
    for j, (batch_input_ids, batch_attention_mask) in tqdm(enumerate(test_generator), total=len(test_generator)):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(test_generator)-1:
                end = len(test_generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask)            
            pred_prob[start:end] = logits.sigmoid().cpu().data.numpy().squeeze()
            
    return pred_prob

def main():
    
    import numpy as np
    import pandas as pd
    import os
    
    model_path = '../input/daigtconfigs/debertav3large'
    model_weights = '../input/daigttrained11/17_ft1/weights_ep0'
    pred_prob0 = get_prediction0(model_path, model_weights)
    
    model_path = '../input/daigtconfigs/debertav3large'
    model_weights = '../input/daigttrained11/17_ft103/weights_ep0'
    pred_prob0 += get_prediction0(model_path, model_weights)
    
    model_path = '../input/daigtconfigs/debertav3large'
    model_weights = '../input/daigttrained11/19_ft1/weights_ep0'
    pred_prob0 += get_prediction0(model_path, model_weights)
    
    model_path = '../input/daigtconfigs/debertav3large'
    model_weights = '../input/daigttrained11/19_ft103/weights_ep0'
    pred_prob0 += get_prediction0(model_path, model_weights)
    
    model_path = '../input/daigtconfigs/debertalarge'
    model_weights = '../input/daigttrained11/20_ft1/weights_ep0'
    pred_prob0 += get_prediction0(model_path, model_weights)
    
    model_path = '../input/daigtconfigs/debertalarge'
    model_weights = '../input/daigttrained11/20_ft103/weights_ep0'
    pred_prob0 += get_prediction0(model_path, model_weights)
    
    pred_prob0 /= 6.0
    
    
    
    pred_prob = pred_prob0.argsort().argsort()

    df = pd.read_csv('../input/llm-detect-ai-generated-text/test_essays.csv')
    id_list = df['id'].values
    sub_df = pd.DataFrame(data={'id': id_list, 'generated': pred_prob})
    sub_df.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()