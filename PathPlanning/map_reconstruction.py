def make_action(game, action, frames):
    game.make_action(action, frames)
    st = game.get_state()
    map = look_around(st)
    var = st.game_variables

    return map, var

def map_to_pts(map):
    # map image has 0, 0 at top left. We will shift it to bottom left
    # also we shift center of the image to (0, 0)
    pts = []
    center = np.array(map.shape) // 2
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i, j] == 0:
                pts.append((map.shape[0] - i, j) - center)
    return pts

def pts_to_map(pts):
    # Find the maximum absolute coordinates to determine map size
    max_x = max(abs(pt[0]) for pt in pts)
    max_y = max(abs(pt[1]) for pt in pts)
    
    # Add some padding and make sure dimensions are large enough for all points
    size_x = max_x + 2
    size_y = max_y + 2
    map = np.full((int(size_x), int(size_y)), 255)
    for pt in pts:
        # Shift coordinates to be non-negative
        x = -int(pt[0])
        y = int(pt[1])
        map[x, y] = 0
    return map

def rot_pts(pts, ang):
    ang = np.radians(ang)
    rot_matrix = np.array([[np.cos(ang), np.sin(ang)],
                          [-np.sin(ang), np.cos(ang)]])
    return (rot_matrix @ pts.T).T


def update_map(global_pts, local_map, map_vars):
    local_pts = map_to_pts(local_map)
    coords = np.array(map_vars[:2])
    ang = map_vars[3]  # Get relative angle from start
    coords_diff = coords - start_coords
    
    # First rotate points based on agent's rotation
    pts = rot_pts(local_pts, ang)
    coords_diff = coords_diff * np.array([-1, 1])
    # Then translate based on agent position
    pts = pts - coords_diff*(1/10)
    global_pts.extend(pts)
    return global_pts