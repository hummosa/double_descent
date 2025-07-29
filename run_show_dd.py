#%%
''' Experimentally showing the dd phenomenon in linear regression '''

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import plot_style
plot_style.set_plot_style()
from configs import Config


cfg = Config(
    d = 20,
    n = 20,
    n_itr = 100,
    noise_std = 0.1,
    noise_type = 'input'
    
)

def run(cfg: Config):
    # Generate data - split into train and test
    total_n = cfg.n
    n_train = int(0.8 * total_n)  # 80% for training
    n_test = total_n - n_train    # 20% for testing
    
    X_train = t.randn(cfg.n_ins, n_train, cfg.d)
    X_test = t.randn(cfg.n_ins, n_test, cfg.d)
    Wt = t.randn(cfg.n_ins, cfg.d, 1)  # Teacher weights

    Ws = t.full((cfg.n_ins, cfg.d, 1), cfg.w_init, dtype=t.float32) # student weights
    Ws = nn.Parameter(Ws)  # Make Ws a parameter to optimize


    if cfg.noise_type == 'input':
        X_train += cfg.noise_std * t.randn(cfg.n_ins, n_train, cfg.d)
        X_test += cfg.noise_std * t.randn(cfg.n_ins, n_test, cfg.d)

    y_train_true = X_train @ Wt  # True training labels
    y_test_true = X_test @ Wt    # True test labels

    if cfg.noise_type == 'output':
         y_train_true += (cfg.noise_std * t.randn(cfg.n_ins, n_train, 1))
         y_test_true += (cfg.noise_std * t.randn(cfg.n_ins, n_test, 1))
    elif cfg.noise_type == 'time-correlated':
        noise_train = t.randn(cfg.n_ins, n_train, 1)
        for i in range(1, n_train):
            noise_train[i] += cfg.noise_std_time * noise_train[i-1]
        y_train_true += noise_train
        
        noise_test = t.randn(cfg.n_ins, n_test, 1)
        for i in range(1, n_test):
            noise_test[i] += cfg.noise_std_time * noise_test[i-1]
        y_test_true += noise_test

    optimizer = t.optim.SGD([Ws], lr=cfg.lr)
    loss_fn = nn.MSELoss(reduction='none')  # Don't reduce across instances
    
    train_losses = []
    test_losses = []
    
    for i in range(cfg.n_itr):
        optimizer.zero_grad()
        # X_train = X_train.detach()
        if cfg.dropout > 0.0:
            mask = (t.rand(X_train.shape) > cfg.dropout).float()
            X_train = X_train * mask

        y_pred_train = X_train @ Ws
        
        # Compute loss for each instance separately, then average across instances
        train_loss_per_instance = loss_fn(y_pred_train, y_train_true).mean(dim=(1, 2))  
        train_loss = train_loss_per_instance.sum()  
        train_loss.backward()
        optimizer.step()
        
        with t.no_grad():
            y_pred_test = X_test @ Ws
            test_loss_per_instance = loss_fn(y_pred_test, y_test_true).mean(dim=(1, 2))
            test_loss = test_loss_per_instance.sum()
        
        train_losses.append(train_loss_per_instance.detach().cpu().numpy())
        test_losses.append(test_loss_per_instance.detach().cpu().numpy())
        
        # if i % 100 == 0:
        #     print(f"Iteration {i}: Train Loss = {train_loss.item():.4f}, Test Loss = {test_loss.item():.4f}")

    train_losses = np.array(train_losses)  # Shape: (n_itr, n_ins)
    test_losses = np.array(test_losses)    
    
    return {
        'final_train_loss_mean': train_losses[-1].mean(),
        'final_test_loss_mean': test_losses[-1].mean(),
        'final_train_loss_std': train_losses[-1].std()/np.sqrt(cfg.n_ins),
        'final_test_loss_std': test_losses[-1].std()/np.sqrt(cfg.n_ins),
        'train_losses_mean': train_losses.mean(axis=1),  # Average across instances
        'test_losses_mean': test_losses.mean(axis=1),
        'train_losses_std': train_losses.std(axis=1),
        'test_losses_std': test_losses.std(axis=1),
        'final_weights': Ws.detach().numpy()
    }



class Experiment:
    def __init__(self, cfg: Config, param_to_vary: str = 'd', range=(2, 10000)) -> None:
        self.cfg = cfg
        self.param_to_vary = param_to_vary
        self.no_of_experiments = 40  # Number of experiments to run
        self.param_values = np.linspace(range[0], range[1], self.no_of_experiments, dtype=int)
        #roune 
        self.param_values = np.round(self.param_values).astype(int)

        self.seed = 42
        np.random.seed(self.seed)
        t.manual_seed(self.seed)
        self.results = []

    def run(self):
        results = []
        for value in self.param_values:
            setattr(self.cfg, self.param_to_vary, value)
            results.append(run(self.cfg))

        self.results = results
        return results

    def plot_results(self) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=100)
        
        # Plot final losses vs parameter values
        train_losses_mean = [res['final_train_loss_mean'] for res in self.results]
        test_losses_mean = [res['final_test_loss_mean'] for res in self.results]
        train_losses_std = [res['final_train_loss_std'] for res in self.results]
        test_losses_std = [res['final_test_loss_std'] for res in self.results]
        
        # Plot with error bars showing standard deviation across instances
        axes[0].errorbar(self.param_values, train_losses_mean, yerr=train_losses_std, 
                        marker='o', label='Train Loss', capsize=3)
        axes[0].errorbar(self.param_values, test_losses_mean, yerr=test_losses_std, 
                        marker='s', label='Test Loss', capsize=3)
        # axes[0].set_xscale('log')
        axes[0].set_xlabel(self.param_to_vary)
        axes[0].set_ylabel('Final Loss')
        axes[0].set_title(f'Final Loss vs {self.param_to_vary}')
        axes[0].legend()
        
        # Plot training curves for the last experiment
        last_result = self.results[-1]
        train_mean = last_result['train_losses_mean']
        test_mean = last_result['test_losses_mean']
        train_std = last_result['train_losses_std']
        test_std = last_result['test_losses_std']
        
        iterations = np.arange(len(train_mean))
        axes[1].plot(iterations, train_mean, label='Train Loss')
        axes[1].fill_between(iterations, train_mean - train_std, train_mean + train_std, alpha=0.3)
        axes[1].plot(iterations, test_mean, label='Test Loss')
        axes[1].fill_between(iterations, test_mean - test_std, test_mean + test_std, alpha=0.3)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Loss')
        axes[1].set_title(f'Training Curves ({self.param_to_vary}={self.param_values[-1]})')
        axes[1].legend()
        
        plt.tight_layout()
        return fig

# if __name__ == '__main__':
    

cfg.noise_std = 1.
exp = Experiment(cfg, 'n', range=(1, 60))
results = exp.run()
fig = exp.plot_results()
fig.suptitle('Double Descent in Linear Regression (with noise)', fontsize=11, y=1.05)


#% Cleaner dataset!
cfg.noise_std = 0.0
exp = Experiment(cfg, 'n', range=(1, 60))
results = exp.run()
fig = exp.plot_results()
fig.suptitle('(clean dataset) noise_std = 0.', fontsize=11, y=1.05)

#% Simplify the porblem! d = 10. 
cfg = Config(
    d = 10,
    n = 20,
    n_itr = 100,
    noise_std = 0.9,
    noise_type = 'input'

)

exp = Experiment(cfg, 'n', range=(1, 60))
results = exp.run()
fig = exp.plot_results()
fig.suptitle('(simplified problem) d = 10', fontsize=11, y=1.05)

#%% Add dropout
cfg = Config(
    d = 20,
    n = 20,
    n_itr = 200,
    noise_std = 0.9,
    noise_type = 'input',
    dropout = 0.1,
)
exp = Experiment(cfg, 'n', range=(1, 60))
results = exp.run()
fig = exp.plot_results()
fig.suptitle('Double Descent with Dropout', fontsize=11, y=1.05)
