import torch
from torch import nn

class TemporalEmbedding:
    def __init__(self, device, model_path="./temporal_model/emb_128.pth"):
        self.model = TemporalVec().to(device)
        self.model_path = model_path
        self.device = device
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=False))
        self.model.eval()
        self._loaded = True
    
    def __call__(self, x):
        self._ensure_loaded()
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device, dtype=torch.float32)
        squeeze_output = x.dim() == 2
        if squeeze_output:
            x = x.unsqueeze(0)
        with torch.no_grad():
            encoded = self.model.encode(x)
        return encoded.squeeze(0) if squeeze_output else encoded

class TemporalVec(nn.Module):
    def __init__(self, k=128, act="sin"):
        super(TemporalVec, self).__init__()

        if k % 2 == 0:
            k1 = k // 2
            k2 = k // 2
        else:
            k1 = k // 2
            k2 = k // 2 + 1
        
        self.fc1 = nn.Linear(6, k1)
        self.fc2 = nn.Linear(6, k2)
        self.d2 = nn.Dropout(0.3)
 
        if act == 'sin':
            self.activation = torch.sin
        else:
            self.activation = torch.cos

        self.fc3 = nn.Linear(k, k // 2)
        self.d3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(k // 2, 6)
        self.fc5 = torch.nn.Linear(6, 6)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.d2(self.activation(self.fc2(x)))
        out = torch.cat([out1, out2], 1)
        out = self.d3(self.fc3(out))
        out = self.fc4(out)
        out = self.fc5(out)
        return out

    def encode(self, x):
        out1 = self.fc1(x)
        out2 = self.activation(self.fc2(x))
        out = torch.cat([out1, out2], -1)
        return out
