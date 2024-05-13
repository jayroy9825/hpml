import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad

class TaylorPruningOptimizer(optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(TaylorPruningOptimizer, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                if d_p.is_sparse:
                    raise RuntimeError("TaylorPruningOptimizer does not support sparse gradients")

                # Compute second-order derivatives using autograd
                H = torch.autograd.functional.hessian(p.grad, p, create_graph=True)

                # Compute the importance scores using Taylor expansion
                importance_scores = torch.abs(H) * torch.abs(p.data)

                # Apply pruning mask
                mask = torch.ones_like(p)
                mask[importance_scores < 0] = 0
                p.data.mul_(mask)

                # Update parameters
                p.data.add_(-lr, d_p)

                # Apply weight decay
                if weight_decay != 0:
                    p.data.add_(-weight_decay * lr, p.data)

                # Momentum update
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(lr * d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(lr * d_p)

                    p.data.add_(-buf)


model = YourNeuralNetworkModel()
optimizer = TaylorPruningOptimizer(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Inside your training loop
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()