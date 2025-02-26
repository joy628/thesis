
import torch
from lstm_gnn_embedding import PatientOutcomeModelEmbedding


def train_model (save_path,epochs,train_loader,graph_loader,device,model):
    best_val_loss = 1000
    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(train_loader):
            flat, graph, ts, lengths, labels = data
            flat, graph, ts, labels = flat.to(device), graph.to(device), ts.to(device), labels.to(device)
            optimizer.zero_grad()
            flat_emb = flat_encoder(flat)
            graph_emb = graph_encoder(graph, graph_loader.dataset.edge_index.to(device))
            ts_emb = ts_encoder(ts, lengths)
            risk_score, _ = model(flat_emb, graph_emb, ts_emb)
            loss = criterion(risk_score, labels)
            loss.backward()
            optimizer.step()
        val_loss = evaluate_model(model, criterion, val_loader, flat_encoder, graph_encoder, ts_encoder, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss}")
    return model