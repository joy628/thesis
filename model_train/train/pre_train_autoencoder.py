import os
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def train_model(model, train_loader, val_loader, criterion, optimizer, config, device):
    best_val_loss = float("inf")
    model_dir = "/home/mei/nas/docker/thesis/data/model/pre_train_autoencoder"
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "best_model_32_1e-4.pth")
    
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs, lengths = batch
            inputs = inputs.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs, lengths)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, lengths = batch
                inputs = inputs.to(device)
                lengths = lengths.to(device)

                outputs, _ = model(inputs, lengths)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at {best_model_path} with Val Loss: {val_loss:.4f}")

    print(f"Training complete. Best Validation Loss: {best_val_loss:.6f}")
    
    
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    mae_list = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, lengths = batch
            inputs = inputs.to(device)
            lengths = lengths.to(device)

            outputs, _ = model(inputs, lengths)
            loss = criterion(outputs, inputs)
            test_loss += loss.item()

            mae = torch.abs(outputs - inputs).mean(dim=(1, 2))
            mae_list.extend(mae.cpu().numpy())

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    
    
def visualize_reconstruction(model, test_loader, device, num_samples=5, feature_indices=[159, 160, 161, 162], save_fig=False, transformer=False):
    """
    visualize the original and reconstructed data for a few samples
    model: trained model
    test_loader: DataLoader for test set
    device: device to run the model
    num_samples: number of samples to visualize
    feature_indices: list of feature indices to visualize
    save_fig: whether to save the figure
    """
    model.eval()
    batch = next(iter(test_loader)) 
    inputs, lengths = batch
    inputs = inputs.to(device)
    lengths = lengths.to(device)
   
    if transformer:
       with torch.no_grad():
             outputs = model(inputs)
    else:
        with torch.no_grad():
            outputs, _ = model(inputs, lengths)
            
    num_samples = min(num_samples, inputs.size(0)) 

    for i in range(num_samples):
        fig, axes = plt.subplots(1, len(feature_indices), figsize=(20, 5))
        fig.suptitle(f"Patient {i+1}: Original vs Reconstructed", fontsize=16)

        for j, feature_idx in enumerate(feature_indices):
            ax = axes[j]
            
            valid_input = inputs[i, :lengths[i], feature_idx].cpu().numpy()  
            valid_output = outputs[i, :lengths[i], feature_idx].cpu().numpy()

            ax.plot(valid_input, label="Original", linestyle='-', color='blue', alpha=0.7)
            ax.plot(valid_output, label="Reconstructed", linestyle='--', color='red', alpha=0.7)

            ax.set_title(f"Feature {feature_idx}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend(loc="upper right")  
            ax.grid(True)

        plt.tight_layout()

        if save_fig:
            plt.savefig(f"patient_{i+1}_reconstruction.png")
        
        plt.show()
        
        
## transformer autoencoder

def generate_mask(seq_len, actual_lens,device):
    
    actual_lens = actual_lens.to(device)
    arange_tensor = torch.arange(seq_len, device=device)
    mask = arange_tensor.expand(len(actual_lens), seq_len) < actual_lens.unsqueeze(1)
    
    return mask

def masked_mae_loss(outputs, targets, mask):

    absolute_error = torch.abs(outputs - targets)
    
    if outputs.dim() == 3:
        absolute_error = absolute_error.mean(dim=-1)  # (batch_size, seq_len)
    
    masked_loss = (absolute_error * mask).sum() / mask.sum()
    return masked_loss

def calculate_correlation(outputs, inputs, mask):
    """
    calculate the Pearson correlation coefficient between the original and reconstructed data
    outputs: reconstructed data from the model
    inputs: original data
    mask: mask for valid data
    """
    batch_correlations = []

    for i in range(inputs.size(0)):    
        valid_input = inputs[i][mask[i]].flatten().detach().cpu().numpy()  
        valid_output = outputs[i][mask[i]].flatten().detach().cpu().numpy() 
        
        corr, _ = pearsonr(valid_input, valid_output)
        batch_correlations.append(corr)
    return sum(batch_correlations) / len(batch_correlations)

def overfit_model(model, num_epochs,dataloader, optimizer,device,hidden=False,attention=False,teacher_forcing=False,transformer=False):
    num_epochs = 500  
    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            inputs, lengths = batch
            inputs = inputs.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            if hidden:
               outputs, _ = model(inputs, lengths,attention = False)
            elif attention:
                outputs, _ = model(inputs, lengths,attention = True) 
            elif teacher_forcing:
                outputs, _ = model(inputs, lengths,teacher_forcing_ratio=0.5)
            elif transformer:
                outputs = model(inputs)
            mask = generate_mask(inputs.size(1), lengths,device)
            loss = masked_mae_loss(outputs, inputs, mask)
            corr = calculate_correlation(outputs, inputs, mask)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            loss.backward()
            optimizer.step()
    
            torch.cuda.empty_cache()
            torch.cuda.memory_reserved(0)
            
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f},Correlation: {corr:.6f}")