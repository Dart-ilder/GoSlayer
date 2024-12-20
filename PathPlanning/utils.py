import math
import vizdoom as vzd
import numpy as np  
from vizdoom import GameVariable,GameState
import sys

from vizdoom import DoomGame, Button

def move_agent_to_target(game: DoomGame, target_x: float, target_y: float):
    """
    Moves the agent from its current position to specified coordinates (target_x, target_y).
    
    Parameters:
    - game: The DoomGame instance.
    - target_x: The x-coordinate of the target position.
    - target_y: The y-coordinate of the target position.
    """
    
    # Get current state
    state = game.get_state()
    current_position = state.game_variables  # Assuming game_variables[0] is x and [1] is y
    current_x = current_position[0]
    current_y = current_position[1]

    # Calculate direction to move
    delta_x = target_x - current_x
    delta_y = target_y - current_y
    
    # Normalize direction vector
    distance = (delta_x**2 + delta_y**2)**0.5
    if distance == 0:
        return True
    
    direction_x = delta_x / distance
    direction_y = delta_y / distance
    
    # Determine actions based on direction
    action = [0] * game.get_available_buttons_size()  # Initialize action array
    
    # Move left or right based on x direction
    if direction_x > 0.1:  # Move right
        action[4] = -1
    elif direction_x < -0.1:  # Move left
        action[4] = 1

    # Move forward or backward based on y direction
    if direction_y > 0.1:  # Move forward
        action[3] = 1
    elif direction_y < -0.1:  # Move backward
        action[3] = -1

    # Execute action for one time step
    #print(action)
    reward = game.make_action(action)
    
    return False
        

def main():
    game = vzd.DoomGame()
    game.load_config('./data/resources/datagen.cfg')
    game.set_doom_game_path('./data/resources/doom2.wad')
    game.set_doom_scenario_path('./data/maps_manykeys_aug/30x30.wad')
    map_index = np.random.randint(1,99)
    game.set_objects_info_enabled(True)
    game.set_doom_map(f'map0{map_index+1}' if map_index<9 else f'map{map_index+1}')
    game.set_window_visible(True)
    game.init()
    game.new_episode()
    state = game.get_state()
    x1 = game.get_game_variable(vzd.GameVariable.POSITION_X)
    y1 = game.get_game_variable(vzd.GameVariable.POSITION_Y)
    # Get all objects from game state
    objects = state.objects
    # Filter for Redcard objects
    redcard_objects = [obj for obj in objects if obj.name == 'RedCard']

    if redcard_objects:
        # Calculate distances to each Redcard
        distances = [(obj, math.sqrt((obj.position_x - x1)**2 + (obj.position_y - y1)**2)) 
                    for obj in redcard_objects]
        # Get closest Redcard
        closest_redcard = min(distances, key=lambda x: x[1])[0]
        x2, y2 = closest_redcard.position_x, closest_redcard.position_y
    else:
        print("No Redcard found in the environment")
        x2, y2 = x1, y1  # Stay in place if no Redcard found
    
    print(f"Moving from {x1},{y1} to {x2},{y2}")
    while not move_agent_to_target(game,x2,y2):
        pass
    game.close()

if __name__ == "__main__":
    main()