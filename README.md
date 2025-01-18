# Melody Map - Graph Neural Network-Based Song Recommendation System

## Overview
This project implements a song recommendation system using a Graph Neural Network (GNN). It leverages user, track, artist, and playlist relationships in a graph structure to generate personalized recommendations.

## Features
- Creates a graph of users, tracks, artists, and playlists based on interactions.
- Uses `torch_geometric` to implement the GNN.
- Applies GCNConv layers for graph convolution.
- Outputs personalized song recommendations.

## Dataset
The dataset is a CSV file containing the following columns:
- `user_id`: Unique identifier for the user.
- `artistname`: Name of the artist.
- `trackname`: Name of the track.
- `playlistname`: Name of the playlist.

The dataset is filtered to include only rows with exactly four columns.

## Prerequisites
- Python 3.8+
- PyTorch
- PyTorch Geometric
- Pandas
- Matplotlib (optional, for graph visualization)
- PyTorchViz or Netron (optional, for model diagram generation)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/gnn-song-recommender.git
   cd gnn-song-recommender
   ```

2. Install required Python packages:
   ```bash
   pip install torch torch-geometric pandas matplotlib
   ```

3. Place the dataset in the project directory with the name `spotify_dataset.csv` or update the file path in the script.

## Usage

### Training the Model
1. Run the script to train the model:
   ```bash
   python train_gnn.py
   ```
2. The script will:
   - Load and preprocess the dataset.
   - Create a graph structure.
   - Train the GNN using Mean Squared Error loss.

### Making Recommendations
To generate recommendations for a specific user:
```python
recommendations = get_recommendations(model, data, user_idx=0, top_k=5)
print(recommendations)
```

### Saving and Loading the Model
- Save the trained model:
  ```python
  torch.save(model.state_dict(), "gnn_model.pth")
  ```
- Load the model:
  ```python
  model.load_state_dict(torch.load("gnn_model.pth"))
  model.eval()
  ```

## Implementation Details
### Graph Construction
- Nodes:
  - Users, tracks, artists, playlists.
- Edges:
  - User ↔ Track
  - Track ↔ Artist
  - Track ↔ Playlist
- Features:
  - Randomly initialized 32-dimensional vectors for nodes.

### Model Architecture
- **GCNConv**: Two layers for graph convolution.
- **ReLU Activation**: Applied after each convolution.
- **Dropout**: Used for regularization.
- **Linear Layer**: Predicts recommendation scores.

### Computation Graph
![gnn_computation_graph](https://github.com/user-attachments/assets/bba67c57-6383-4e3c-bb2f-a87b0d0d64f9)

### Loss Function
Mean Squared Error (MSE) between predicted and actual edge weights.

### Optimizer
Adam optimizer with a learning rate of 0.01.

## Visualization
To visualize the GNN architecture:
- Use PyTorchViz:
  ```python
  from torchviz import make_dot
  make_dot(output, params=dict(model.named_parameters())).render("gnn_graph", format="png")
  ```
- Use Netron to view the ONNX model:
  ```python
  torch.onnx.export(model, dummy_input, "gnn_model.onnx", opset_version=11)
  ```

## Future Work
- Incorporate node-specific features (e.g., genres, popularity).
- Use a larger dataset for training and evaluation.
- Implement a more advanced GNN architecture, such as Graph Attention Networks (GAT).
- Add support for temporal data (e.g., listening history).

## Acknowledgments
This project uses the PyTorch Geometric library for efficient graph-based computation.

## License
This project is licensed under the Apache 2 License. See the LICENSE file for details.
