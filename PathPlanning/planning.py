import numpy as np

def to_grid(mp, cell_size=10):
    """Receives global map and returns a grid of the same size, with walls as 0 and empty space as 255."""
    grid = np.full((mp.shape[0]//cell_size, mp.shape[1]//cell_size), 255)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.any(mp[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size] == 0):
                grid[i,j] = 0
    return grid