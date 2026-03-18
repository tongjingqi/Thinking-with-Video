BASE_URL="YOUR_BASE_URL"
API_KEY="YOUR_API_KEY_HERE"

python eval/src/eval_VLM.py \
    --result_run_dir minitest_vlm_result/Qwen/Qwen3-VL-32B-Thinking_20260318-114118 \
    --output_root minitest_vlm_eval \
    --model gpt-4o \
    --base_url $BASE_URL \
    --api_key $API_KEY \
    --threads 16