python infer/request_videos.py \
    --model sora_video2-landscape \
    --dataset_root ../VideoThinkBench/minitest_Text-Centric_Reasoning \
    --datasets GSM8K \
    --output_root minitest_result \
    --base_url https://jyapi.ai-wx.cn/v1 \
    --api_key "YOUR_API_KEY_HERE" \
    --threads 4 \
    --max_request_attempts 3