"""
client_simulator.py

Simulates federated clients:
- Local training on each client
- Sending weights to server
- Receiving global weights
"""

import copy

class Client:
    def __init__(self, client_id, model, dataloader, device):
        self.id = client_id
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def train_local(self, criterion, optimizer, epochs=1):
        """Local training loop"""
        self.model.train()
        for _ in range(epochs):
            for x, y in self.dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

    def get_weights(self):
        """Return model weights"""
        return copy.deepcopy(self.model.state_dict())

    def set_weights(self, global_weights):
        """Update model with global weights"""
        self.model.load_state_dict(global_weights)
