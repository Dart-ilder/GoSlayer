# GoSlayer
VizDoom agent for M1 Skoltech course project

This is a Tesla-like autopilot agent for Doom enviroment.

The only data going in is array of pixel camera views from agent looking in different directions.
Top-down map of the surrounding envieronment is being built from it with Neural Network.
From this top-down view of local enviernment agent with Monte-Carlo descision trees proposes next action.
Action is executed, agent takes new pictures, assesses it's surroundings, cycle is anew.
