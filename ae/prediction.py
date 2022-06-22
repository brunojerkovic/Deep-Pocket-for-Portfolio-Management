import torch
import pandas as pd
from torch.utils.data import DataLoader

def predict_on_dataset(model, dataset, batch_size, device):
    # Prepare the dataset for prediction
    norm_dataset = dataset[[col for col in dataset.columns if col.startswith('norm_')]]
    torch_dataset = [torch.tensor(t, device=device, dtype=torch.float32) for _, t in norm_dataset.iterrows()]
    data_loader = DataLoader(torch_dataset, batch_size=batch_size, num_workers=0)

    # Perform prediction on the model
    feats = []
    model.eval()
    with torch.no_grad():
        for input_tensor in data_loader:
            feats += model.encoder(input_tensor).detach().cpu().tolist()

    # Save compressed features to the dataset
    ae_df = pd.DataFrame(feats, columns=['ae1', 'ae2', 'ae3'])
    new_dataset = pd.concat((dataset, ae_df), axis=1)

    return new_dataset
