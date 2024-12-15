# GoSlayer
VizDoom agent for M1 Skoltech course project

This is a Tesla-like autopilot agent for Doom enviroment.

The only data going in is array of pixel camera views from agent looking in different directions.
Top-down map of the surrounding envieronment is being built from it with Neural Network.
From this top-down view of local enviernment agent with Monte-Carlo descision trees proposes next action.
Action is executed, agent takes new pictures, assesses it's surroundings, cycle is anew.

## Data
Data is located in data/maps folder including *.npz* files.
**map{i}_data.npz** contains data for 99 maps.
You can access it using These code:
```Python
data = np.load(f'data/maps/map{i}_data.npz')
d1 = data['maps']
d2 = data['images']
d3 = data['positions']
d4 = data['depth']
d5 = data['objects']
```

objects keep coordinates of goal object in the map.
maps keep top-down view of the map.
images keep pixel views from agent's camera.
positions keep agent's position and orientation.
depth keep depth buffer from agent's camera.
Shapes of data:
((10, 4, 3, 240, 320, 3),
 (10, 4, 240, 320, 3),
 (10, 4, 4),
 (10, 4, 3, 240, 320),
 (1, 1))

