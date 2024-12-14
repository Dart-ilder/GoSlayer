# GoSlayer
VizDoom agent for M1 Skoltech course project

This is a Tesla-like autopilot agent for Doom enviroment.

The only data going in is array of pixel camera views from agent looking in different directions.
Top-down map of the surrounding envieronment is being built from it with Neural Network.
From this top-down view of local enviernment agent with Monte-Carlo descision trees proposes next action.
Action is executed, agent takes new pictures, assesses it's surroundings, cycle is anew.

## Data
Data is located in data folder including *.npz* files.
You can acess it using These code:
```Python
data = np.load('data/mywayhome_data.npz')
d1 = data['maps']
d2 = data['images']
d3 = data['positions']
```