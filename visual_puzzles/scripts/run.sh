DATA_DIR=example_data
OUTPUT_DIR=output

python infer/request_videos.py \
    --model sora_video2-landscape \
    --tasks color_size size_grid color_grid color_hexagon color_overlap_squares polygon_sides_color rectangle_height_color shape_reflect shape_size_grid size_cycle \
    --data_root $DATA_DIR \
    --base_url https://jyapi.ai-wx.cn/v1 \
    --api_key "YOUR_API_KEY_HERE" \
    --output_root $OUTPUT_DIR \
    --threads 16 \
    --max_request_attempts 5 \
    --request_attempt_delay 2