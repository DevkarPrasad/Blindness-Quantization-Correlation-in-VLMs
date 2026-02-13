'''import json
import os
import google.generativeai as genai
from tqdm import tqdm
'''
import json
import os
import time
from google import genai
from tqdm import tqdm
from collections import defaultdict

# 1. Setup the Client
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# 2. Path to your JSONL file
LOG_FILE = "/mnt/c/Prasad/Code/Togo/BL_Correletion/lmms-eval/logs/hallusion/qwen2b_awq/Qwen__Qwen2-VL-2B-Instruct-AWQ/20260128_185106_samples_hallusion_bench_image.jsonl"

def ask_gemini(question, prediction, target):
    prompt = (
        f"Judge if the model's answer is correct.\n"
        f"Question: {question}\n"
        f"Factual Target: {target}\n"
        f"Model Response: {prediction}\n\n"
        f"Does the response match the logic of the target? Answer ONLY 'YES' or 'NO'."
    )
    try:
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        # Handle Rate Limiting gracefully
        time.sleep(4) # Force 15 RPM limit for Free Tier
        return "YES" in response.text.upper()
    except Exception as e:
        if "429" in str(e):
            print("\nRate limit hit! Sleeping for 30s...")
            time.sleep(30)
        return False

# 3. Load JSONL
samples = []
with open(LOG_FILE, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            samples.append(json.loads(line))

print(f"Scoring {len(samples)} samples. Estimated time: {len(samples)*4/60:.1f} minutes.")

# 4. Metrics Tracking
consistency_tracker = defaultdict(list)
total_correct = 0

for s in tqdm(samples):
    # Mapping based on your specific JSONL keys
    question = s.get('input', 'N/A')
    prediction = s.get('filtered_resps', ['N/A'])[0]
    target = s.get('target', 'N/A')
    
    # CRITICAL: Use doc_hash or input as the unique key to separate image sets
    # This prevents the "Total Unique Image Sets: 1" error
    group_key = s.get('doc_hash') or s.get('input') 
    
    is_correct = ask_gemini(question, prediction, target)
    
    if is_correct:
        total_correct += 1
    
    consistency_tracker[group_key].append(is_correct)

# 5. Final Report
overall_acc = (total_correct / len(samples)) * 100
consistent_sets = sum(1 for res in consistency_tracker.values() if all(res))
consistency_score = (consistent_sets / len(consistency_tracker)) * 100

print(f"\n" + "="*45)
print(f"--- HALLUSIONBENCH RESEARCH REPORT ---")
print(f"Overall Accuracy:   {overall_acc:.2f}%")
print(f"Consistency Score:  {consistency_score:.2f}%")
print(f"Total Questions:    {len(samples)}")
print(f"Unique Image Sets:  {len(consistency_tracker)}")
print(f"="*45)
'''
# Configuration
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-2.5-flash') # Use Flash for free/fast scoring
LOG_FILE = "/mnt/c/Prasad/Code/Togo/BL_Correletion/lmms-eval/logs/hallusion/qwen2b_awq/Qwen__Qwen2-VL-2B-Instruct-AWQ/20260128_185106_results.json"

def ask_gemini(question, prediction, target):
    prompt = (
        f"You are a benchmark judge. A Vision-Language Model was asked: '{question}'\n"
        f"The correct answer is: '{target}'\n"
        f"The model's actual response was: '{prediction}'\n"
        f"Does the model's response accurately provide the correct answer? "
        f"Respond with only 'YES' or 'NO'."
    )
    try:
        response = model.generate_content(prompt)
        return "YES" in response.text.upper()
    except:
        return False

# Load and score
samples = []
with open(LOG_FILE, 'r') as f:
    for line in f:
        samples.append(json.loads(line))

print(f"Scoring {len(samples)} samples with Gemini...")
results = []
for s in tqdm(samples):
    # Use 'filtered_resps' (model output) and compare with ground truth
    q = s['doc']['question']
    p = s['filtered_resps'][0]
    t = s['doc']['answer']
    results.append(ask_gemini(q, p, t))

accuracy = (sum(results) / len(results)) * 100
print(f"\n--- Final HallusionBench Results ---")
print(f"Total Samples: {len(results)}")
print(f"Overall Accuracy: {accuracy:.2f}%")
'''
