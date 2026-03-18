DATA_DIR="../VideoThinkBench/minitest_Text-Centric_Reasoning"
OUTPUT_DIR=minitest_vlm_result

# BASE_URL="YOUR_BASE_URL"
# API_KEY="YOUR_API_KEY_HERE"

# MODEL="TARGET_VLM"

python infer/test_VLM.py \
    --model $MODEL \
    --dataset_root ../VideoThinkBench/minitest_Text-Centric_Reasoning \
    --datasets GSM8K MATH500 AIME24 AIME25 BBH MMLU MMLU-Pro GPQA_diamond SuperGPQA \
    --output_root minitest_vlm_result \
    --base_url $BASE_URL \
    --api_key $API_KEY \
    --threads 16 \
    --max_request_attempts 5 \
    --request_attempt_delay 2 \
    --no_images

python infer/test_VLM.py \
    --model $MODEL \
    --dataset_root ../VideoThinkBench/minitest_Text-Centric_Reasoning \
    --datasets MathVista MathVision MMBench MMMU \
    --output_root minitest_vlm_result \
    --base_url $BASE_URL \
    --api_key $API_KEY \
    --threads 16 \
    --max_request_attempts 5 \
    --request_attempt_delay 2