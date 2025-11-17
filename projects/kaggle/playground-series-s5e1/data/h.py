import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

# Function to compute Hellinger distance
def hellinger_distance(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / np.sqrt(2)

# Load dataset
df = pd.read_csv('train.csv')

# Initialize variables
best_distance = float('inf')
best_subset = None
NUM_ITER = 100000

# Iterate to find the best subset
for _ in range(NUM_ITER):
    # Sample 10% of the data
    subset = df.sample(frac=0.1, random_state=np.random.randint(0, 10000))
    
    distances = []
    for column in df.select_dtypes(include=[np.number]).columns:
        # Remove NaN values
        full_data = df[column].dropna()
        subset_data = subset[column].dropna()
        
        # Compute histograms
        full_dist, _ = np.histogram(full_data, bins=30, density=True)
        subset_dist, _ = np.histogram(subset_data, bins=30, density=True)
        
        # Normalize histograms
        full_dist = full_dist / np.sum(full_dist)
        subset_dist = subset_dist / np.sum(subset_dist)
        
        # Compute Hellinger distance
        distance = hellinger_distance(full_dist, subset_dist)
        distances.append(distance)
    
    # Average distance across all columns
    avg_distance = np.mean(distances)
    
    # Update best subset if current is better
    if avg_distance < best_distance:
        best_distance = avg_distance
        best_subset = subset

# Save the best subset to CSV
if best_subset is not None:
    best_subset.to_csv('train10p.csv', index=False)
    print(f'Best subset saved with Hellinger distance: {best_distance}')
else:
    print('No suitable subset found.')

