import torch
from torch_geometric.data import Data
import pandas as pd
import csv
import random
from scripts.gnn import GNNModel
from scripts.utils import train_model
from scripts.utils import get_recommendations

# Open the CSV file and process line by line
with open('data/spotify_dataset.csv', mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    valid_rows = []

    # Read all rows and filter only those with exactly 4 columns
    all_rows = list(reader)
    valid_rows = [row for row in all_rows if len(row) == 4]  # Ensure only rows with 4 columns are kept

# Sample a thousandth of the dataset randomly
dataset_size = len(valid_rows) // 1000
sampled_rows = random.sample(valid_rows, dataset_size)

# Convert the sampled rows back to a DataFrame
df = pd.DataFrame(sampled_rows, columns=["user_id", "artistname", "trackname", "playlistname"])

# Create mappings for users, tracks, and playlists
user_mapping = {user: idx for idx, user in enumerate(df['user_id'].unique())}
track_mapping = {track: idx + len(user_mapping) for idx, track in enumerate(df['trackname'].unique())}
artist_mapping = {artist: idx + len(user_mapping) + len(track_mapping) for idx, artist in enumerate(df['artistname'].unique())}
playlist_mapping = {playlist: idx + len(user_mapping) + len(track_mapping) + len(artist_mapping) for idx, playlist in enumerate(df['playlistname'].unique())}

# Creating the nodes for the graph
user_nodes = df['user_id'].map(user_mapping).values
track_nodes = df['trackname'].map(track_mapping).values
artist_nodes = df['artistname'].map(artist_mapping).values
playlist_nodes = df['playlistname'].map(playlist_mapping).values

# Create edge_index (i.e., relationships between users, tracks, artists, and playlists)
# Edges are bidirectional between user and track, track and artist, track and playlist
user_to_track_edges = torch.tensor([user_nodes, track_nodes], dtype=torch.long)
track_to_artist_edges = torch.tensor([track_nodes, artist_nodes], dtype=torch.long)
track_to_playlist_edges = torch.tensor([track_nodes, playlist_nodes], dtype=torch.long)

# Combine all edges
edge_index = torch.cat([user_to_track_edges, track_to_artist_edges, track_to_playlist_edges], dim=1)

# Create node features (random features here; you can use specific features like genre, track length, etc.)
num_nodes = len(user_mapping) + len(track_mapping) + len(artist_mapping) + len(playlist_mapping)
num_features = 32  # Can be adjusted
node_features = torch.randn(num_nodes, num_features)

# Build the target ratings or interactions for training (could be interaction count or binary like/dislike)
# Here, we are assuming interaction as edge weights, which could be 1 for presence of edge
edge_weights = torch.ones(edge_index.shape[1], dtype=torch.float)

# Create Data object
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weights)

# Initialize model
model = GNNModel(num_features=num_features, hidden_channels=64)
# Train the model
train_model(model, data)

torch.save(model, "model/song_recommender.pth")

user_id = '11da254d9d1948488318e3ea286bf484'  # Example user_id
user_idx = user_mapping.get(user_id)
if user_idx is not None:
    recommended_tracks = get_recommendations(model, data, user_idx, track_mapping=track_mapping)
    print(f"Recommended tracks for user {user_id}: {recommended_tracks}")
else:
    print(f"User {user_id} not found.")