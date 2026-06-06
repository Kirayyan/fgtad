
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


# ========== 对比学习工具函数 ==========

def augment_time_series(x, aug_type='jitter', noise_level=0.05):

    device = x.device

    if aug_type == 'jitter' or aug_type == 'mixed':

        noise = torch.randn_like(x.cpu()) * noise_level
        x = x + noise.to(device)

    if aug_type == 'scale' or aug_type == 'mixed':

        scale = 1.0 + (torch.rand(x.shape[0], 1, x.shape[2]) - 0.5) * 0.2
        x = x * scale.to(device)

    if aug_type == 'crop':

        seq_len = x.shape[1]
        crop_len = int(seq_len * 0.8) 
        start = random.randint(0, seq_len - crop_len)
        x = x[:, start:start+crop_len, :]

        if start > 0:
            x = F.pad(x, (0, 0, start, 0), mode='constant', value=0)
        if x.shape[1] < seq_len:
            x = F.pad(x, (0, 0, 0, seq_len - x.shape[1]), mode='constant', value=0)

    return x



def contrastive_loss_infonce(z_i, z_j, temperature=0.5):

    batch_size = z_i.shape[0]


    z_i = z_i.mean(dim=1)  
    z_j = z_j.mean(dim=1)


    z_i = F.normalize(z_i, p=2, dim=1)
    z_j = F.normalize(z_j, p=2, dim=1)


    z = torch.cat([z_i, z_j], dim=0)  


    sim_matrix = torch.mm(z, z.t()) / temperature


    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels], dim=0) 


    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)


    loss = F.cross_entropy(sim_matrix, labels)

    return loss


class LightweightTS2Vec(nn.Module):

    def __init__(self, input_dims=1, output_dims=128, hidden_dims=256):

        super(LightweightTS2Vec, self).__init__()
        self.output_dims = output_dims
  
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dims, hidden_dims, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_dims, hidden_dims, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_dims, output_dims, kernel_size=3, padding=1),
        )
        

        self.projection = nn.Sequential(
            nn.Linear(output_dims, output_dims),
            nn.ReLU(),
            nn.Linear(output_dims, output_dims)
        )
    
    def encode(self, x):


        if x.dim() == 3 and x.shape[-1] == self.encoder[0].in_channels:
            x = x.transpose(1, 2)
        elif x.dim() == 3 and x.shape[1] == self.encoder[0].in_channels:
            pass  
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        

        embeddings = self.encoder(x)  
        

        embeddings = embeddings.transpose(1, 2)
        
        return embeddings
    
    def forward(self, x):

        embeddings = self.encode(x)
        projected = self.projection(embeddings)
        return projected


class TemporalEmbedding_TS2Vec:

    def __init__(self, device, model_path=None, output_dims=128, use_official=False):

        self.device = device
        self.output_dims = output_dims
        self.model_path = model_path
        self.use_official = use_official
        
        if use_official:

            try:
                from ts2vec import TS2Vec
                self.ts2vec_model = TS2Vec(
                    input_dims=6,  
                    device=device if isinstance(device, int) else 0,
                    output_dims=output_dims
                )
                if model_path and os.path.exists(model_path):
                    self.ts2vec_model.load(model_path)
                    self.ts2vec_model.eval()
                else:
                    print("Warning: TS2Vec model not found. You need to train it first.")
                    self.ts2vec_model = None
            except ImportError:
                print("Warning: Official TS2Vec library not found. Using lightweight version.")
                self.use_official = False
                self._init_lightweight()
        else:

            self._init_lightweight()
    
    def _init_lightweight(self):
        self.ts2vec_model = LightweightTS2Vec(
            input_dims=6,  
            output_dims=self.output_dims
        ).to(self.device)
        

        if self.model_path and os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.ts2vec_model.load_state_dict(state_dict)
                self.ts2vec_model.eval()
                print(f"Loaded TS2Vec model from {self.model_path}")
                
                test_input = torch.zeros(1, 10, 6).to(self.device) 
                with torch.no_grad():
                    test_output = self.ts2vec_model.encode(test_input)
                    if torch.isnan(test_output).any() or torch.isinf(test_output).any():
                        print("Warning: TS2Vec model output contains NaN/Inf! Model may not be properly trained.")
                        print("Please retrain the TS2Vec model using: python train_ts2vec.py --dataset porto")
            except Exception as e:
                print(f"Warning: Failed to load TS2Vec model: {e}")
                print("Using untrained model. Please train it first.")
        else:
            print("Warning: TS2Vec model not found. Using untrained model.")
            print("Please train it first using: python train_ts2vec.py --dataset porto")
    
    def time_to_sequence(self, times):

        hours = times[:, :, 0]
        minutes = times[:, :, 1]
        seconds = times[:, :, 2]
        years = times[:, :, 3]
        months = times[:, :, 4]
        days = times[:, :, 5]
        

        hours_norm = hours / 24.0
        

        minutes_norm = minutes / 60.0
        

        seconds_norm = seconds / 60.0

        year_min = 2013.0
        year_max = 2020.0
        years_norm = (years - year_min) / (year_max - year_min)
        years_norm = torch.clamp(years_norm, 0.0, 1.0)  
        

        months_norm = (months - 1) / 11.0
        

        days_norm = (days - 1) / 30.0
        

        sequences = torch.stack([
            hours_norm,
            minutes_norm,
            seconds_norm,
            years_norm,
            months_norm,
            days_norm
        ], dim=-1)  # [batch_size, seq_len, 6]
        
        return sequences
    
    def __call__(self, times):

        if self.ts2vec_model is None:
            raise ValueError("TS2Vec model not initialized. Please train it first using train_ts2vec.py")
        
 
        sequences = self.time_to_sequence(times)
        
        if self.use_official:
  
            sequences_np = sequences.cpu().numpy()
            embeddings = []
            for i in range(sequences_np.shape[0]):
                sample = sequences_np[i].reshape(1, -1, 6) 
                embedding = self.ts2vec_model.encode(
                    sample,
                    encoding_window='full_series'
                )
                embeddings.append(embedding[0])
            embeddings = np.stack(embeddings, axis=0)
            embeddings = torch.from_numpy(embeddings).float().to(self.device)
        else:

            with torch.no_grad():
                embeddings = self.ts2vec_model.encode(sequences)
            

            if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                print("Warning: TS2Vec output contains NaN or Inf! Replacing with zeros.")
                embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
            

            embeddings = torch.clamp(embeddings, min=-10.0, max=10.0)
        
        return embeddings


def train_ts2vec_model(times_data, device, output_dims=128, save_path='./temporal_model/ts2vec_model.pth',
                       epochs=50, batch_size=512, lr=0.001, use_official=False,
                       use_contrastive=True, temperature=0.5, aug_type='mixed'):

    print("=" * 50)
    print("Training TS2Vec Model")
    print("=" * 50)
    
    print("Converting time vectors to 6-channel sequences...")
    sequences = []
    for times in times_data:
        hours = times[:, 0] / 24.0
        minutes = times[:, 1] / 60.0
        seconds = times[:, 2] / 60.0
        
        year_min = 2013.0
        year_max = 2020.0
        years = (times[:, 3] - year_min) / (year_max - year_min)
        years = np.clip(years, 0.0, 1.0)
        
        months = (times[:, 4] - 1) / 11.0
        
        days = (times[:, 5] - 1) / 30.0
        
        seq_6d = np.stack([hours, minutes, seconds, years, months, days], axis=1)
        sequences.append(seq_6d)  
    
    if use_official:
        try:
            from ts2vec import TS2Vec
            train_data = np.array(sequences)  
            
            model = TS2Vec(
                input_dims=6,  
                device=device if isinstance(device, int) else 0,
                output_dims=output_dims
            )
            
            print(f"Training with official TS2Vec library...")
            loss_log = model.fit(
                train_data,
                verbose=True,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr
            )
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.save(save_path)
            print(f"TS2Vec model saved to {save_path}")
            return model
        except ImportError:
            print("Official TS2Vec library not found. Using lightweight version.")
            use_official = False
    
    if not use_official:
        print("Training with lightweight TS2Vec implementation...")
        

        max_len = max(len(seq) for seq in sequences)
        padded_sequences = []
        for seq in sequences:
            if len(seq) < max_len:
                padding = np.zeros((max_len - len(seq), 6))  
                seq = np.vstack([seq, padding])
            padded_sequences.append(seq)
        
        train_data = np.array(padded_sequences)  # [n_samples, max_len, 6]
        train_tensor = torch.FloatTensor(train_data).to(device)
        

        model = LightweightTS2Vec(
            input_dims=6,  
            output_dims=output_dims
        ).to(device)
        

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)


        if use_contrastive:
            print(f"Training mode: Contrastive Learning (InfoNCE)")
            print(f"  - Temperature: {temperature}")
            print(f"  - Augmentation: {aug_type}")
        else:
            print(f"Training mode: Simplified (No Contrastive Learning)")

        print(f"Training for {epochs} epochs...")
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0


            for i in range(0, len(train_tensor), batch_size):
                batch = train_tensor[i:i+batch_size]

                if use_contrastive:

                    aug1 = augment_time_series(batch, aug_type=aug_type, noise_level=0.05)
                    aug2 = augment_time_series(batch, aug_type=aug_type, noise_level=0.05)

                    z1 = model.encode(aug1)
                    z2 = model.encode(aug2)


                    loss = contrastive_loss_infonce(z1, z2, temperature=temperature)

                else:

                    embeddings = model.encode(batch)
                    projected = model.projection(embeddings)
                    criterion = nn.MSELoss()
                    loss = criterion(projected, embeddings.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"TS2Vec model saved to {save_path}")
        
        model.eval()
        return model



LightweightContrastiveTS2Vec = LightweightTS2Vec
LCTS2Vec = LightweightTS2Vec 



if __name__ == "__main__":
    print("=" * 60)
    print("TS2Vec Temporal Embedding Module - Contrastive Learning")
    print("=" * 60)
    print()
    print("Two training modes:")
    print()
    print("1. ct:")
    print("   python train_ts2vec.py --dataset porto --use_contrastive")
    print("   - Using InfoNCE to compare losses")
    print("   - Data augmentation: noise, scaling, blending")
    print("   - Better representation of learning ability")
    print()
    print("2. Simplified mode (backward compatible):")
    print("   python train_ts2vec.py --dataset porto --no_contrastive")
    print("   - Simplified training methods")
    print("   - Compatible with older versions")
    print()
    print("Use in FG-TAD:")
    print("   config.Setting in temporal_embedding_type='ts2vec'")
    print("=" * 60)

