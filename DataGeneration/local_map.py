import vizdoom as vzd
import numpy as np
import matplotlib.pyplot as plt

possible_actions = np.zeros((8,6),dtype=int).tolist()
possible_actions[0] = [1,0,0,0,0]
possible_actions[1] = [0,1,0,0,0]
possible_actions[2] = [0,0,-45,0,0]
possible_actions[3] = [0,0,45,0,0]
possible_actions[4] = [0,0,0,50,0]
possible_actions[5] = [0,0,0,-50,0]
possible_actions[6] = [0,0,0,0,30]
possible_actions[7] = [0,0,0,0,-30]

def map_to_black(raw_map: np.array):
    """
    Receives colour map from VizDoom.
    Returns black and white map of walls only.
    Walls are black 0, background is white 255.
    Player icon removed.
    """
    grayscale = np.dot(raw_map[...,:3], [0.2989, 0.5870, 0.1140])
    
    grey_threshold = 100
    grey_mask = (grayscale > grey_threshold)
    
    black_threshold = 80
    black_mask = (grayscale < black_threshold)
    
    grayscale[grey_mask | black_mask] = 0
    grayscale[~(grey_mask | black_mask)] = 255
        
    grayscale[118:123,158:161] = [255]
    
    return grayscale

def thicken_walls(grey_map: np.array):
    map = grey_map.copy()
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if grey_map[i,j] == 0:
                map[i-1:i+1,j-1:j+1] = 0
    return map

def ray_casting(grey_map: np.array):
    height, width = grey_map.shape
    center_y, center_x = height // 2, width // 2
    processed_map = grey_map.copy()

    # Only get black pixels from upper half of image
    black_pixels = np.argwhere((grey_map[:center_y] == 0))
    
    # No need for center mask since we're only looking at upper half
    dy = black_pixels[:,0] - center_y
    dx = black_pixels[:,1] - center_x

    abs_dy = np.abs(dy)
    abs_dx = np.abs(dx)
    sign_y = np.where(dy != 0, np.sign(dy), 0).astype(int)
    sign_x = np.where(dx != 0, np.sign(dx), 0).astype(int)

    dx_dominant = abs_dx > abs_dy
    pdx = np.where(dx_dominant, sign_x, 0)
    pdy = np.where(dx_dominant, 0, sign_y)
    es = np.where(dx_dominant, abs_dy, abs_dx)
    el = np.where(dx_dominant, abs_dx, abs_dy)

    max_steps = np.max(el)
    ray_points = np.zeros((len(black_pixels), max_steps, 2), dtype=int)
    curr_points = np.column_stack((np.full_like(el, center_y), np.full_like(el, center_x)))
    error = el // 2

    for step in range(max_steps):
        error -= es
        mask = error < 0
        curr_points[mask] += np.column_stack((sign_y[mask], sign_x[mask]))
        error[mask] += el[mask]
        curr_points[~mask] += np.column_stack((pdy[~mask], pdx[~mask]))
        ray_points[:,step] = curr_points

    valid_steps = np.minimum(el, max_steps)
    blocked = np.zeros(len(black_pixels), dtype=bool)
    
    for i in range(len(black_pixels)):
        points = ray_points[i, :valid_steps[i]-1]
        blocked[i] = np.any(grey_map[points[:,0], points[:,1]] == 0)

    processed_map[black_pixels[blocked,0], black_pixels[blocked,1]] = 255
    processed_map[center_y:, :] = 255
    return processed_map

def look_around(state: vzd.GameState):
    """receives central state from VizDoom and returns processed local map"""
    whole_map = state.automap_buffer.transpose(0,1,2)
    walls = map_to_black(whole_map)
    walls = thicken_walls(walls)
    walls = ray_casting(walls)
    return walls

def color_transform(raw_map: np.array):
    map = raw_map.copy()
    
    # Create mask for gray colors (where all RGB values are similar)
    gray_mask = (abs(raw_map[:,:,0] - raw_map[:,:,1]) < 10) & (abs(raw_map[:,:,1] - raw_map[:,:,2]) < 10)
    
    # Create mask for dark brown colors (where R value is low)
    dark_brown_mask = (raw_map[:,:,0] < 100) & (raw_map[:,:,1] < 80) & (raw_map[:,:,2] < 60)
    
    # Set dark browns and grays to black
    map[gray_mask | dark_brown_mask] = [0, 0, 0]
    
    # Set everything else (background browns) to white
    map[~(gray_mask | dark_brown_mask)] = [255, 255, 255]
    
    # Keep the red marker if needed
    #map[118:121,158:161,:] = [255,0,0]
    
    return map

def tracing(raw_map: np.array):
    # Create a copy and get center coordinates
    map = raw_map.copy()
    center_y, center_x = 119, 159
    
    # Pre-allocate result array
    result = np.full_like(map, 255)
    
    # Get wall coordinates in upper half
    wall_coords = np.where((map[:center_y] != [255,255,255]).all(axis=2))
    
    # Vectorized direction calculations
    dx = wall_coords[1] - center_x 
    dy = wall_coords[0] - center_y
    distances = np.sqrt(dx*dx + dy*dy).astype(int)
    steps = np.maximum(distances * 2, 1)
    
    # Process each wall point
    for i in range(len(wall_coords[0])):
        y, x = wall_coords[0][i], wall_coords[1][i]
        
        # Calculate ray steps
        step_x = dx[i] / steps[i]
        step_y = dy[i] / steps[i]
        
        # Ray positions
        ray_x = center_x + np.arange(steps[i]) * step_x
        ray_y = center_y + np.arange(steps[i]) * step_y
        
        # Round to integer coordinates
        check_x = np.round(ray_x).astype(int)
        check_y = np.round(ray_y).astype(int)
        
        # Check if ray hits any walls
        is_visible = True
        for j in range(steps[i]):
            if (check_y[j], check_x[j]) != (y,x) and np.all(map[check_y[j], check_x[j]] == [0,0,0]):
                is_visible = False
                break
                
        result[y,x] = [0,0,0] if is_visible else [255,255,255]
    
    # Add red marker
    #result[118:121, 158:161] = [255,0,0]
    
    return result

# duration 4 tics
def get_pictures(game: vzd.DoomGame):
    state = game.get_state()    
    st = state.screen_buffer.transpose(1,2,0)
    img = np.array([st])
    dph = np.array([state.depth_buffer])
    # turn left
    game.make_action(possible_actions[2])
    state = game.get_state()
    st = state.screen_buffer.transpose(1,2,0)
    img = np.append(img, [st], axis=0)
    dm = state.automap_buffer.transpose(1,2,0)
    debug_map = np.array([dm])
    dph = np.append(dph, [state.depth_buffer], axis=0)
    # turn right
    game.make_action(possible_actions[3],2)
    state = game.get_state()
    st = state.screen_buffer.transpose(1,2,0)
    img = np.append(img, [st], axis=0)
    dm = state.automap_buffer.transpose(1,2,0)
    debug_map = np.append(debug_map, [dm], axis=0)
    dph = np.append(dph, [state.depth_buffer], axis=0)
    # return to center
    game.make_action(possible_actions[2])
    state = game.get_state()
    dm = state.automap_buffer.transpose(1,2,0)
    debug_map = np.append(debug_map, [dm], axis=0)
    #dph = np.append(dph, [state.depth_buffer], axis=0)
    auto_map = state.automap_buffer.transpose(1,2,0)
    colortransformed_map = color_transform(auto_map)
    processed_map = tracing(colortransformed_map)
    return img, auto_map, processed_map, debug_map, dph

from matplotlib import pyplot as plt
import multiprocessing as mp
from functools import partial

def process_map(map_index, game_config='resources/temp_maps/datagen.cfg', scenario_path='data/maps_1key_noaug/30x30.wad'):
    game = vzd.DoomGame()
    game.load_config(game_config)
    game.set_doom_scenario_path(scenario_path)
    game.set_automap_mode(vzd.AutomapMode.WHOLE)
    game.set_render_hud(False)
    game.set_objects_info_enabled(True)
    game.set_labels_buffer_enabled(True)
    
    game.set_doom_map(f'map0{map_index+1}' if map_index<9 else f'map{map_index+1}')
    
    # Track previous positions
    prev_positions = []
    map_images = []
    map_maps = []
    map_positions = []
    map_depth = []
    
    for j in range(10):
        game.init()
        
        # Keep trying until we get a valid position
        attempts = 0
        while True:
            game.new_episode()
            if game.is_episode_finished():
                game.new_episode()
            attempts += 1
            state = game.get_state()
            st = state.screen_buffer.transpose(1,2,0)
            img = np.array([st])
            am = state.automap_buffer.transpose(1,2,0)
            dph = np.array([state.depth_buffer])
            
            tmp_img, auto_map, tmp_processed_map, debug_map, dph = get_pictures(game)
            tmp_position = state.game_variables
            
            # Check if position is too close to any previous position
            curr_x, curr_y = tmp_position[0], tmp_position[1]
            should_restart = False
            for prev_x, prev_y in prev_positions:
                if abs(curr_x - prev_x) < 100 and abs(curr_y - prev_y) < 100:
                    should_restart = True
                    break
                    
            if not should_restart or attempts > 100:  # Only proceed if position is valid
                break
                
        prev_positions.append((curr_x, curr_y))
        
        temp_objects = state.objects
        inner_objects = [{'x': obj.position_x, 'y': obj.position_y, 'z': obj.position_z, 'angle': obj.angle} for obj in temp_objects if (obj.name == 'RedCard' and obj.id == 0)]
        
        map_images.append(np.array([tmp_img]))
        map_maps.append(np.array([tmp_processed_map]))
        map_positions.append(np.array([tmp_position]))
        map_depth.append(np.array([dph]))
        
        game.close()
        
    print(f'map{map_index+1} done')
    return map_images, map_maps, map_positions, map_depth, inner_objects

def main():
    # Set up parallel processing
    num_processes = mp.cpu_count() - 1  # Leave one CPU free
    
    from functools import partial
    scenario_path = 'data/maps_manykeys_aug/30x30.wad'
    process_map_fn = partial(process_map, scenario_path=scenario_path)
    folder_name = scenario_path.split('/')[-2]
    
    results = process_map(process_map_fn, range)
    return results
    
    pool = mp.Pool(processes=num_processes)
    n_maps = 9
    # Process maps in parallel
    results = pool.map(process_map_fn, range(n_maps))

    # Close the pool
    pool.close()
    pool.join()
    
    print("all processes finished")

    # Combine results
    images = np.array([result[0] for result in results])
    maps = np.array([result[1] for result in results]) 
    positions = np.array([result[2] for result in results])
    depth = np.array([result[3] for result in results])
    objects = [result[4] for result in results]
    # Get dimensions
    
    
    for obj in objects:
        if obj == None:
            print("None")
            
    print("all data combined")

    samples_per_map = maps.shape[0] // n_maps

    # Split data into 99 parts
    maps_split = np.array_split(maps, n_maps)
    images_split = np.array_split(images, n_maps) 
    positions_split = np.array_split(positions, n_maps)
    depth_split = np.array_split(depth, n_maps)

    print("saving data")
    # Save each split as separate file
    import os
    os.makedirs(f'data/{folder_name}/processed', exist_ok=True)
    for i in range(n_maps):
        print(f"saving map {i+1}/{n_maps}")
        np.savez(f'data/{folder_name}/processed/map{i+1}_data.npz',
                 maps=maps_split[i],
             images=images_split[i],
             positions=positions_split[i],
             depth=depth_split[i],
             objects=objects[i])
    
if __name__ == '__main__':
    main()
