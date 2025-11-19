DATA_DIR=example_data

python gen_data/data_generation.py create_data color_size $DATA_DIR --limit 1 --seed 42

# for pattern in color_size size_grid color_grid color_hexagon color_overlap_squares polygon_sides_color rectangle_height_color shape_reflect shape_size_grid size_cycle ; do
#     python gen_data/data_generation.py create_data $pattern $DATA_DIR --limit 1 --seed 17 --target_size "(1280, 704)"
# done