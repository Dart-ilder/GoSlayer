from mazeexplorer.mazeexplorer import MazeExplorer

train_env = MazeExplorer(number_maps=10,
                         unique_maps=True,
                         complexity=0.7,
                         density=0.2,
                         size=(50, 50),
                         random_spawn=True,
                         random_textures=True,
                         random_key_positions=True,
                         keys=400,
                         data_augmentation=True,
                         mazes_path="data/large_maps_manykeys_aug")
# train_env.generate_mazes()

