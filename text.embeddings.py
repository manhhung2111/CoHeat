import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import torch.nn as nn

# Initialize the model
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

# Define paths to input and output files
input_file = 'datasets/iFashion/items.jsonl'  # Replace with your input file path
output_file = 'text_embeddings.npy'  # File to save the embeddings

# Prepare a list to hold embeddings
embeddings = []

class LinearTransformer(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearTransformer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


# Open the input JSONL file
with open(input_file, 'r', encoding='utf-8') as infile:
    for line in infile:
        # Parse the JSON object
        data = json.loads(line.strip())
        print(f"Processing item {data['item_id']}")
        
        # Extract the title
        title = data['title']
        
        # Generate the embedding for the title
        embedding = model.encode(title, normalize_embeddings=True)
        
        # Convert to a tensor
        embedding_tensor = torch.tensor(embedding)

        # Reduce the embedding size to 64
        linear_transformer = LinearTransformer(input_size=embedding_tensor.size(0), output_size=64)

        # Append the reduced embedding to the list
        embedding = linear_transformer(embedding_tensor)
        embeddings.append(embedding.detach().numpy())
        print(f"Embedding size of item {data['item_id']}: {embedding.shape}")

# Convert the list to a NumPy array
embeddings_matrix = np.vstack(embeddings)

# Save the embeddings matrix to a file
np.save(output_file, embeddings_matrix)

print(f"Embeddings saved to {output_file}. Shape: {embeddings_matrix.shape}")
