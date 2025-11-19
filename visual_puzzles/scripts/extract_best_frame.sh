# To quantify deviation between a frame and the solution image:
# - Color-Filling Tasks: use RGB Euclidean Distance
# - Shape-Drawing Tasks: calculate a "coverage difference" after binarization


# Color-Filling Tasks
python eval/find_best_frame.py \
    --results_dir output/ \
    --tasks color_size color_grid color_hexagon color_overlap_squares polygon_sides_color rectangle_height_color \
    --compare_width 512 \
    --compare_height 512 \
    --distance_metric euclidean


# Shape-Drawing Tasks
python eval/find_best_frame.py \
    --results_dir output/ \
    --tasks size_grid shape_reflect shape_size_grid size_cycle \
    --compare_width 512 \
    --compare_height 512 \
    --distance_metric coverage