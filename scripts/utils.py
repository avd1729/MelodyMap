import torch
import torch.nn.functional as F

# Training function
def train_model(model, data, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        
        # For simplicity, we use Mean Squared Error loss
        loss = F.mse_loss(out[data.edge_index[0]], data.edge_attr)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss:.4f}')

# Making recommendations for a user
def get_recommendations(model, data, user_idx, track_mapping, top_k=5):
    model.eval()
    with torch.no_grad():
        embeddings = model.conv2(
            model.conv1(data.x, data.edge_index, data.edge_attr),
            data.edge_index
        )
        
        # Get the user embedding
        user_embedding = embeddings[user_idx]
        
        # Get track embeddings
        track_indices = torch.tensor(list(track_mapping.values()))
        track_embeddings = embeddings[track_indices]
        
        # Calculate similarity between user embedding and track embeddings
        similarity = F.cosine_similarity(user_embedding.unsqueeze(0), track_embeddings)
        
        # Get top-k most similar tracks
        top_k_indices = similarity.argsort(descending=True)[:top_k]
        
        recommended_tracks = [list(track_mapping.keys())[idx] for idx in top_k_indices]
        return recommended_tracks