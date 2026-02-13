import os
import json
import time
from google import genai
from google.genai import types

# 1. Initialize the 2026 Client
# Ensure you have run: export GOOGLE_API_KEY="your_key"
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# Paths (Adjust to your exact WSL paths)
RESULTS_JSONL = "/mnt/c/Prasad/Code/Togo/BL_Correletion/lmms-eval/logs/hallusion/qwen2b_awq/Qwen__Qwen2-VL-2B-Instruct-AWQ/20260128_185106_samples_hallusion_bench_image.jsonl"
BATCH_INPUT_FILE = "gemini_batch_input.jsonl"

def prepare_batch_file():
    print(f"Reading results from {RESULTS_JSONL}...")
    batch_requests = []
    
    with open(RESULTS_JSONL, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
                # Map your specific HallusionBench keys
                question = data.get('input', 'N/A')
                prediction = data.get('filtered_resps', ['N/A'])[0]
                target = data.get('target', 'N/A')
                
                prompt = (
                    f"Judge if the model is correct. "
                    f"Question: {question} | Model Answer: {prediction} | "
                    f"Correct Factual Claim: {target}. Answer ONLY YES or NO."
                )
                
                # Format for 2026 Batch API
                batch_requests.append({
                    "request": {"contents": [{"parts": [{"text": prompt}]}]}
                })
            except json.JSONDecodeError:
                continue

    with open(BATCH_INPUT_FILE, "w") as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")
    
    print(f"Created {BATCH_INPUT_FILE} with {len(batch_requests)} requests.")
    return len(batch_requests)
    


def main():
    num_samples = prepare_batch_file()
    
    # 2. Upload to Gemini File API (Required for Batch)
    print("Uploading file to Gemini File API...")
    uploaded_file = client.files.upload(
        file=BATCH_INPUT_FILE,
        config=types.UploadFileConfig(
            mime_type="application/jsonl",
            display_name="hallusion_bench_batch_eval"
        )
    )
    print(f"File uploaded successfully: {uploaded_file.name}")

    # 3. Create the Batch Job
    print(f"Starting Batch Job for {num_samples} samples...")
    batch_job = client.batches.create(
        model="gemini-2.5-flash",
        src=uploaded_file.name
    )

    print("\n" + "="*40)
    print(f"SUCCESS: Batch Job Created!")
    print(f"Job Name: {batch_job.name}")
    print(f"State: {batch_job.state}")
    print("="*40)
    print("\nNote: Batch jobs take 15-60 minutes. Do not run the script again.")
    print("You can check status with: client.batches.get(name=batch_job.name)")

if __name__ == "__main__":
    main()