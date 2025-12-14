DATA_DIR=example_data
# DATA_DIR=minitest

API_KEY="YOUR_API_KEY_HERE"

# MODEL=sora_video2-landscape
MODEL=veo_3_1-landscape
OUTPUT_DIR=output/veo_3_1-landscape

python infer/request_videos.py \
    --model $MODEL \
    --tasks color_size \
    --data_root $DATA_DIR \
    --base_url https://jyapi.ai-wx.cn/v1 \
    --api_key $API_KEY \
    --output_root $OUTPUT_DIR \
    --threads 5 \
    --max_request_attempts 5 \
    --request_attempt_delay 2 \
    --request_mode direct \
    --direct_poll_interval 5 \
    --direct_max_poll_attempts 60 \
    --direct_request_timeout 180

# python infer/request_videos.py \
#     --model $MODEL \
#     --tasks color_size size_grid color_grid color_hexagon color_overlap_squares polygon_sides_color rectangle_height_color shape_reflect shape_size_grid size_cycle \
#     --data_root $DATA_DIR \
#     --base_url https://jyapi.ai-wx.cn/v1 \
#     --api_key $API_KEY \
#     --output_root $OUTPUT_DIR \
#     --threads 16 \
#     --max_request_attempts 5 \
#     --request_attempt_delay 2