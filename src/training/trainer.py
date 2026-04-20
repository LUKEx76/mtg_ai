import copy
import torch

def train_torch_model(
    model: torch.nn.Module, 
    train_loader: torch.utils.data.DataLoader, 
    val_loader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: torch.nn.Module, 
    device: str, 
    epochs: int=50, 
    verbose: int=10,
    is_baseline: bool=False
):
    """
    reusable training loop that tracks and returns the best model based on validation loss.
    """
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # training phase
        model.train()
        train_loss = 0
        for texts, tabs, labels in train_loader:
            texts, tabs, labels = texts.to(device), tabs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            if is_baseline:
                outputs = model(texts, None)
            else:
                outputs = model(texts, tabs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * texts.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for texts, tabs, labels in val_loader:
                texts, tabs, labels = texts.to(device), tabs.to(device), labels.to(device)
                if is_baseline:
                    outputs = model(texts, None)
                else:
                    outputs = model(texts, tabs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * texts.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
        if (epoch + 1) % verbose == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
    # load best weights before returning
    model.load_state_dict(best_model_wts)
    return model, history
