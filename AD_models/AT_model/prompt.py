import torch
import torch.nn as nn
import numpy as np

class Prompt(nn.Module):
    def __init__(self, ftr_dim=768, pool_size=10, prompt_num=10, channel=-1, top_k=1, batchwise_prompt=False, 
                 diversified=False):
        super().__init__()

        self.ftr_dim = ftr_dim
        self.pool_size = pool_size # 
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        prompt_pool_shape = (pool_size, prompt_num, ftr_dim) #win_size, channel)
        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
        nn.init.uniform_(self.prompt, -1, 1)
        
        # if using learnable prompt keys
        key_shape = (pool_size, ftr_dim)
        self.prompt_key = nn.Parameter(torch.randn(key_shape))
        nn.init.uniform_(self.prompt_key, -1, 1)
        self.diversified=diversified
        if diversified:
            self.selection_table =np.zeros(pool_size)
        
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    
    def forward(self, ftr):
        out = dict()
        prompt_key_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, dim
        ftr = self.l2_normalize(ftr, dim=1) # B, C

        similarity = torch.matmul(ftr, prompt_key_norm.t()) # B, Pool_size
        _, idx = torch.topk(similarity, k=self.top_k, dim=1)
        if self.diversified:
            selections = idx.flatten().numpy()
            # Use np.bincount to count occurrences
            counts = np.bincount(selections)
            self.selection_table += counts
            
            similarity = torch.matmul(ftr,prompt_key_norm.t())
        batched_prompt_raw = self.prompt[idx] # B, top_k, win, C
        batched_prompt = batched_prompt_raw
        out['prompt_idx'] = idx
            
        # Put pull_constraint loss calculation inside
        batched_key_norm = prompt_key_norm[idx] # B, top_k, C
        out['selected_key'] = batched_key_norm
        ftr = ftr.unsqueeze(1) # B, 1, C
        sim = batched_key_norm * ftr # B, top_k, C
        reduce_sim = torch.sum(sim) / ftr.shape[0] # Scalar

        out['reduce_sim'] = reduce_sim
        
        out['batched_prompt']=batched_prompt
        return out
