DATA_DIR=data
OUTPUT_DIR=vlm_output

BASE_URL="YOUR_BASE_URL"
API_KEY="YOUR_API_KEY_HERE"

MODEL="TARGET_VLM"


# Tasks that can be tested without the options line for the VLMs
NO_OPTION_TASKS=(color_grid color_hexagon color_overlap_squares polygon_sides_color rectangle_height_color)

# Tasks that require the options line (sets `provide_options` flag)
OPTION_TASKS=(color_size size_grid shape_reflect shape_size_grid size_cycle)

python infer/test_VLM.py \
    --model "$MODEL" \
    --tasks "${NO_OPTION_TASKS[@]}" \
    --data_root "$DATA_DIR" \
    --base_url "$BASE_URL" \
    --api_key "$API_KEY" \
    --output_root "$OUTPUT_DIR/no_options" \
    --threads 8 \
    --max_request_attempts 5 \
    --request_attempt_delay 2

python infer/test_VLM.py \
    --model "$MODEL" \
    --tasks "${OPTION_TASKS[@]}" \
    --data_root "$DATA_DIR" \
    --base_url "$BASE_URL" \
    --api_key "$API_KEY" \
    --output_root "$OUTPUT_DIR/with_options" \
    --threads 8 \
    --max_request_attempts 5 \
    --request_attempt_delay 2 \
    --provide_options
