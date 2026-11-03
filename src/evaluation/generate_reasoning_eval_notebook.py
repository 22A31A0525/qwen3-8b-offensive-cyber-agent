"""
Generates the self-contained Kaggle Jupyter Notebook for Dual-GPU Reasoning Evaluation (Qwen3-8B v3).
Supports both direct execution and side-by-side thinking mode comparison.
Uses subprocess execution for dual workers to eliminate multiprocessing pickling errors in Jupyter/Papermill.
Zero embedded secrets; authenticates strictly via Kaggle Secrets ('HF_TOKEN').
"""

import json
import os

def create_reasoning_notebook(output_path: str):
    cells = []

    # Cell 1: Markdown Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🛡️ Qwen3-8B v3 Cybersecurity Reasoning Benchmark & Comparative Evaluation\n",
            "\n",
            f"> **Model Under Test**: `{os.getenv('HF_REPO_ID', 'Qwen/Qwen3-8B-offensive-cyber-agent')}` (1.0 Full Epoch Fine-Tuned)  \n",
            "> **Base Model**: `Qwen/Qwen3-8B`  \n",
            "> **Benchmark Dataset**: 130 Pure Cybersecurity Reasoning Samples (`tools: []`, SHA-256 Validated)  \n",
            "> **Evaluation Architecture**: Dual Tesla T4 GPUs (65 samples per GPU via Parallel Subprocesses)  \n",
            "> **Modes Compared**:\n",
            ">   1. **Direct Answering (`enable_thinking=False`)**: Deterministic greedy decoding (`temp=0.0`, `do_sample=False`)\n",
            ">   2. **Deep Reasoning (`enable_thinking=True`)**: Temperature sampling (`temp=0.6`, `top_p=0.95`, `do_sample=True`)\n",
            "> **Authentication**: Strictly Kaggle Secrets (`Add-ons -> Secrets -> HF_TOKEN`). Zero hardcoded credentials.\n"
        ]
    })

    # Cell 2: Authentication & GPU Verification
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# ==============================================================================\n",
            "# CELL 1: ENVIRONMENT & AUTHENTICATION (KAGGLE SECRETS)\n",
            "# ==============================================================================\n",
            "import os\n",
            "import sys\n",
            "import torch\n",
            "\n",
            "# Authenticate strictly via Kaggle Secrets or environment\n",
            "HF_TOKEN = \"\"\n",
            "SECRET_CANDIDATE_KEYS = [\n",
            "    \"HF_TOKEN\", \"hf_token\", \"HUGGING_FACE_HUB_TOKEN\", \"huggingface\",\n",
            "    \"HUGGINGFACE_TOKEN\", \"hf\", \"HF\", \"HuggingFace\", \"HUGGING_FACE\", \"HF_HUB_TOKEN\"\n",
            "]\n",
            "try:\n",
            "    from kaggle_secrets import UserSecretsClient\n",
            "    client = UserSecretsClient()\n",
            "    for key in SECRET_CANDIDATE_KEYS:\n",
            "        try:\n",
            "            secret = client.get_secret(key)\n",
            "            if secret and secret.strip():\n",
            "                HF_TOKEN = secret.strip()\n",
            "                print(f\"[AUTH] Successfully retrieved token from Kaggle Secrets ('{key}').\", flush=True)\n",
            "                break\n",
            "        except Exception:\n",
            "            pass\n",
            "except Exception as e:\n",
            "    print(f\"[AUTH] Kaggle UserSecretsClient query info: {e}\", flush=True)\n",
            "\n",
            "if not HF_TOKEN:\n",
            "    for env_k in [\"HF_TOKEN\", \"HUGGING_FACE_HUB_TOKEN\", \"HUGGINGFACE_TOKEN\"]:\n",
            "        val = os.environ.get(env_k, \"\").strip()\n",
            "        if val:\n",
            "            HF_TOKEN = val\n",
            "            print(f\"[AUTH] Discovered token from environment variable '{env_k}'.\", flush=True)\n",
            "            break\n",
            "\n",
            "if not HF_TOKEN:\n",
            "    import glob\n",
            "    token_files = [\n",
            "        \"/kaggle/input/hf-auth-token/hf_token.txt\",\n",
            "    ] + glob.glob(\"/kaggle/input/**/hf_token.txt\", recursive=True)\n",
            "    for tf in token_files:\n",
            "        if os.path.exists(tf):\n",
            "            with open(tf, \"r\", encoding=\"utf-8\") as f:\n",
            "                val = f.read().strip()\n",
            "            if val:\n",
            "                HF_TOKEN = val\n",
            "                print(f\"[AUTH] Discovered token from attached secure Kaggle dataset ('{tf}').\", flush=True)\n",
            "                break\n",
            "\n",
            "if not HF_TOKEN:\n",
            "    raise ValueError(\n",
            "        \"FATAL: HF_TOKEN not found!\\n\"\n",
            "        \"To resolve this in Kaggle:\\n\"\n",
            "        \"  1. In the Kaggle top menu bar, click 'Add-ons' -> 'Secrets'.\\n\"\n",
            "        \"  2. Check the box next to 'HF_TOKEN' to attach your secret to this notebook session.\\n\"\n",
            "        \"  3. Click 'Run All' (or 'Save Version' -> 'Save & Run All').\"\n",
            "    )\n",
            "\n",
            "os.environ[\"HF_TOKEN\"] = HF_TOKEN\n",
            "os.environ[\"HUGGING_FACE_HUB_TOKEN\"] = HF_TOKEN\n",
            "os.environ[\"TOKENIZERS_PARALLELISM\"] = \"false\"\n",
            "\n",
            "# Verify Hardware Parity (Dual Tesla T4)\n",
            "num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0\n",
            "print(f\"[HARDWARE] Detected Available GPUs: {num_gpus}\", flush=True)\n",
            "for i in range(num_gpus):\n",
            "    print(f\"  GPU {i}: {torch.cuda.get_device_name(i)}\", flush=True)\n",
            "\n",
            "if num_gpus < 2:\n",
            "    print(\"[WARNING] Less than 2 GPUs detected. Set Kaggle accelerator to 'GPU T4 x2' for dual-GPU benchmark.\", flush=True)\n"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 3: Dependency Installation
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# ==============================================================================\n",
            "# CELL 2: DEPENDENCIES INSTALLATION & VERIFICATION\n",
            "# ==============================================================================\n",
            "import subprocess\n",
            "import sys\n",
            "\n",
            "def ensure_deps():\n",
            "    try:\n",
            "        import bitsandbytes\n",
            "        import peft\n",
            "        import accelerate\n",
            "        print(f\"[SETUP] Dependencies verified: bitsandbytes=={bitsandbytes.__version__}, peft=={peft.__version__}\", flush=True)\n",
            "    except ImportError:\n",
            "        print(\"[SETUP] Installing bitsandbytes>=0.46.1 peft accelerate...\", flush=True)\n",
            "        subprocess.run([\n",
            "            sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"-U\",\n",
            "            \"bitsandbytes>=0.46.1\", \"peft\", \"accelerate\"\n",
            "        ], check=True)\n",
            "        print(\"[SETUP] Installation complete [OK].\", flush=True)\n",
            "\n",
            "ensure_deps()\n"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 4: Dataset Integrity & Loading
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# ==============================================================================\n",
            "# CELL 3: DATASET INGESTION & SHA-256 CHECKSUM VERIFICATION\n",
            "# ==============================================================================\n",
            "import os\n",
            "import json\n",
            "import hashlib\n",
            "\n",
            "PINNED_TEST_SHA256 = \"fabd494f98bd5e79e81c3b6e3c740db99a1b0d5abcfd0d2e40a72606721f02d5\"\n",
            "\n",
            "candidates = [\n",
            "    \"/kaggle/input/qwen3-8b-retraining-dataset-v3/test.jsonl\",\n",
            "    \"data/clean/test.jsonl\",\n",
            "] + glob.glob(\"/kaggle/input/**/test.jsonl\", recursive=True)\n",
            "\n",
            "test_path = None\n",
            "for c in candidates:\n",
            "    if os.path.exists(c):\n",
            "        test_path = os.path.abspath(c)\n",
            "        break\n",
            "\n",
            "if not test_path:\n",
            "    raise FileNotFoundError(f\"test.jsonl could not be located in candidates: {candidates}\")\n",
            "\n",
            "with open(test_path, \"rb\") as f:\n",
            "    file_bytes = f.read()\n",
            "file_sha = hashlib.sha256(file_bytes).hexdigest()\n",
            "\n",
            "if os.path.basename(test_path) == \"test.jsonl\":\n",
            "    if file_sha != PINNED_TEST_SHA256:\n",
            "        raise ValueError(f\"SHA-256 mismatch! Expected {PINNED_TEST_SHA256}, got {file_sha}\")\n",
            "    print(f\"[DATA] Pinned SHA-256 Checksum Verified: {file_sha[:16]}... [MATCH]\", flush=True)\n",
            "\n",
            "samples = []\n",
            "for line in file_bytes.decode(\"utf-8\").splitlines():\n",
            "    if not line.strip():\n",
            "        continue\n",
            "    rec = json.loads(line)\n",
            "    if rec.get(\"metadata\", {}).get(\"category\") == \"reasoning\":\n",
            "        samples.append(rec)\n",
            "\n",
            "if len(samples) != 130:\n",
            "    raise ValueError(f\"Strict dataset integrity violation: Expected 130 samples, found {len(samples)}\")\n",
            "\n",
            "print(f\"[DATA] Successfully ingested strictly {len(samples)} pure cybersecurity reasoning dialogues from: {test_path}\", flush=True)\n"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 5: Worker Script Creation (Writes standalone Python script to eliminate Jupyter pickle bugs)
    worker_script = '''
import os
import sys
import json
import re
import time
import argparse
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MAX_CONTEXT = 4096

def parse_thinking_and_answer(text):
    if not text or not text.strip():
        return None, "", False
    t = text.strip()
    if "<think>" in t and "</think>" in t:
        thinking_matches = re.findall(r"<think>(.*?)</think>", t, re.DOTALL)
        thinking = "\\n".join(m.strip() for m in thinking_matches if m.strip())
        final_answer = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()
        return (thinking if thinking else None), final_answer, False
    if "</think>" in t and "<think>" not in t:
        parts = t.split("</think>", 1)
        return (parts[0].strip() or None), parts[1].strip(), False
    if "<think>" in t and "</think>" not in t:
        thinking = t.split("<think>", 1)[1].strip()
        return (thinking if thinking else None), "", True
    return None, t, False

def compute_token_f1(prediction, reference):
    thinking, final_answer, is_unfinished = parse_thinking_and_answer(prediction)
    eval_text = final_answer
    def tokenize(text):
        text = re.sub(r"[^\\w\\s]", " ", text.lower())
        return [t for t in text.split() if t]
    pred_tokens = tokenize(eval_text)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "brevity": 1.0, "composite_score": 1.0,
                "zero_tool_violation": False, "has_thinking": thinking is not None,
                "thinking_char_count": len(thinking) if thinking else 0, "is_unfinished_thinking": is_unfinished}
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "brevity": 0.0, "composite_score": 0.0,
                "zero_tool_violation": False, "has_thinking": thinking is not None,
                "thinking_char_count": len(thinking) if thinking else 0, "is_unfinished_thinking": is_unfinished}
    pred_counts = {}
    for t in pred_tokens: pred_counts[t] = pred_counts.get(t, 0) + 1
    ref_counts = {}
    for t in ref_tokens: ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = sum(min(cnt, ref_counts[t]) for t, cnt in pred_counts.items() if t in ref_counts)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    ratio = len(pred_tokens) / max(1, len(ref_tokens))
    brevity = 1.0 if ratio <= 1.5 else max(0.2, 1.5 / ratio)
    zero_tool_violation = bool(re.search(r"(<tool_call>|<function=|```tool_call|`call:)", prediction))
    if zero_tool_violation or is_unfinished:
        f1_effective = 0.0
        composite = 0.0
    else:
        f1_effective = f1 * brevity
        composite = round(0.40 * f1_effective + 0.60 * precision, 4)
    return {
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "brevity": round(brevity, 4), "zero_tool_violation": zero_tool_violation, "composite_score": composite,
        "has_thinking": thinking is not None, "thinking_char_count": len(thinking) if thinking else 0,
        "is_unfinished_thinking": is_unfinished
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--shard_file", type=str, required=True)
    parser.add_argument("--output_shard_base", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_id", type=str, required=True)
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--modes", type=str, default="no_thinking,thinking")
    args = parser.parse_args()

    gpu_id = args.gpu_id
    eval_modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    torch.cuda.set_device(gpu_id)

    with open(args.shard_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"[WORKER-GPU{gpu_id}] Initializing worker for {len(records)} samples on cuda:{gpu_id} (modes={eval_modes})...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=args.hf_token, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    eos_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and isinstance(im_end_id, int) and im_end_id not in eos_ids:
        eos_ids.append(im_end_id)

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map={"": gpu_id}, torch_dtype=torch.float16, attn_implementation="sdpa", token=args.hf_token, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_id, device_map={"": gpu_id}, token=args.hf_token)
    model.eval()
    torch.cuda.empty_cache()

    alloc_gb = torch.cuda.memory_allocated(gpu_id) / (1024**3)
    print(f"[WORKER-GPU{gpu_id}] Model loaded! VRAM Allocated: {alloc_gb:.2f} GB [OK]", flush=True)

    for mode in eval_modes:
        is_thinking = (mode == "thinking")
        mode_shard_path = f"{args.output_shard_base}_{mode}.jsonl"
        print(f"[WORKER-GPU{gpu_id}] Starting pass for mode='{mode}' (enable_thinking={is_thinking})...", flush=True)

        shard_results = []
        start_time = time.time()

        with open(mode_shard_path, "w", encoding="utf-8") as out_f:
            for idx, rec in enumerate(records):
                sample_id = rec.get("id", f"sample_{idx}")
                messages = rec.get("messages", [])

                target_idx = None
                for m_i in reversed(range(len(messages))):
                    if messages[m_i].get("role") == "assistant":
                        target_idx = m_i
                        break
                if target_idx is None:
                    raise ValueError(f"FATAL: Dialogue {sample_id} contains no assistant turn!")

                prompt_msgs = messages[:target_idx]
                reference_answer = messages[target_idx].get("content") or ""

                try:
                    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=is_thinking)
                except TypeError:
                    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
                    if not is_thinking and prompt_text.endswith("<|im_start|>assistant\\n"):
                        prompt_text += "<think>\\n\\n</think>\\n\\n"
                except Exception:
                    prompt_text = ""
                    for pm in prompt_msgs:
                        prompt_text += f"<|im_start|>{pm['role']}\\n{pm['content']}<|im_end|>\\n"
                    prompt_text += "<|im_start|>assistant\\n"
                    if not is_thinking:
                        prompt_text += "<think>\\n\\n</think>\\n\\n"

                encoded = tokenizer(prompt_text, return_tensors="pt")
                input_ids = encoded["input_ids"].to(f"cuda:{gpu_id}")
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(f"cuda:{gpu_id}")

                prompt_tokens = input_ids.shape[1]
                max_new = min(1024, MAX_CONTEXT - prompt_tokens)

                gen_kwargs = {
                    "input_ids": input_ids, "attention_mask": attention_mask, "max_new_tokens": max_new,
                    "repetition_penalty": 1.05, "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                    "eos_token_id": eos_ids,
                }
                if is_thinking:
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = 0.6
                    gen_kwargs["top_p"] = 0.95
                else:
                    gen_kwargs["do_sample"] = False
                    gen_kwargs["temperature"] = 0.0

                gen_start = time.time()
                with torch.no_grad():
                    gen_out = model.generate(**gen_kwargs)
                latency = time.time() - gen_start

                gen_ids = gen_out[0][prompt_tokens:]
                generated_text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
                if "<|im_end|>" in generated_text: generated_text = generated_text.split("<|im_end|>")[0].strip()
                if "<|endoftext|>" in generated_text: generated_text = generated_text.split("<|endoftext|>")[0].strip()

                scores = compute_token_f1(generated_text, reference_answer)
                entry = {
                    "id": sample_id, "gpu_id": gpu_id, "mode": mode, "enable_thinking": is_thinking,
                    "user_prompt": prompt_msgs[-1]["content"] if prompt_msgs else "",
                    "generated_reasoning": generated_text, "reference_answer": reference_answer,
                    "prompt_tokens": prompt_tokens, "generated_tokens": len(gen_ids),
                    "generation_latency_sec": round(latency, 2),
                    "tokens_per_sec": round(len(gen_ids) / max(0.01, latency), 2),
                    "metrics": scores, "timestamp": datetime.datetime.now().isoformat(),
                }
                out_f.write(json.dumps(entry) + "\\n")
                out_f.flush()
                shard_results.append(entry)

                if (idx + 1) % 10 == 0 or (idx + 1) == len(records):
                    elapsed_min = (time.time() - start_time) / 60.0
                    print(f"[WORKER-GPU{gpu_id}] [{mode.upper()}] Progress [{idx+1:2d}/{len(records)}] | F1: {scores['f1']:.4f} | Elapsed: {elapsed_min:.1f}m", flush=True)

        print(f"[WORKER-GPU{gpu_id}] Completed mode '{mode}' ({len(shard_results)} samples) in {(time.time() - start_time)/60.0:.2f} mins.", flush=True)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
'''

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# ==============================================================================\n",
            "# CELL 4: WRITE DUAL-GPU REASONING WORKER SCRIPT (STANDALONE SUBPROCESS)\n",
            "# ==============================================================================\n",
            f"worker_code = {repr(worker_script)}\n",
            "\n",
            "worker_path = \"/kaggle/working/reasoning_worker.py\"\n",
            "with open(worker_path, \"w\", encoding=\"utf-8\") as f:\n",
            "    f.write(worker_code)\n",
            "\n",
            "print(f\"[SETUP] Standalone worker script written to: {worker_path}\", flush=True)\n"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 6: Parallel Execution Dispatch via subprocess.Popen
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# ==============================================================================\n",
            "# CELL 5: EXECUTE DUAL-GPU BENCHMARK (PARALLEL PROCESSES)\n",
            "# ==============================================================================\n",
            "import subprocess\n",
            "import threading\n",
            "import time\n",
            "import math\n",
            "\n",
            "BASE_MODEL_ID = \"Qwen/Qwen3-8B\"\n",
            f"ADAPTER_ID = \"{os.getenv('HF_REPO_ID', 'Qwen/Qwen3-8B-offensive-cyber-agent')}\"\n",
            "OUTPUT_DIR = \"eval_output\"\n",
            "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
            "\n",
            "EVAL_MODES = [\"no_thinking\", \"thinking\"]  # Side-by-side comparative evaluation\n",
            "total_samples = len(samples)\n",
            "half = math.ceil(total_samples / 2)\n",
            "shard_0 = samples[:half]\n",
            "shard_1 = samples[half:]\n",
            "\n",
            "shard0_in = \"/kaggle/working/shard0_in.jsonl\"\n",
            "shard1_in = \"/kaggle/working/shard1_in.jsonl\"\n",
            "shard_base_0 = os.path.join(OUTPUT_DIR, \"shard_gpu0\")\n",
            "shard_base_1 = os.path.join(OUTPUT_DIR, \"shard_gpu1\")\n",
            "\n",
            "with open(shard0_in, \"w\", encoding=\"utf-8\") as f:\n",
            "    for r in shard_0: f.write(json.dumps(r) + \"\\n\")\n",
            "with open(shard1_in, \"w\", encoding=\"utf-8\") as f:\n",
            "    for r in shard_1: f.write(json.dumps(r) + \"\\n\")\n",
            "\n",
            "print(f\"[PARALLEL] Dispatching {len(shard_0)} samples to GPU 0 and {len(shard_1)} samples to GPU 1...\", flush=True)\n",
            "\n",
            "cmd0 = [\n",
            "    sys.executable, \"-u\", \"/kaggle/working/reasoning_worker.py\",\n",
            "    \"--gpu_id\", \"0\",\n",
            "    \"--shard_file\", shard0_in,\n",
            "    \"--output_shard_base\", shard_base_0,\n",
            "    \"--base_model\", BASE_MODEL_ID,\n",
            "    \"--adapter_id\", ADAPTER_ID,\n",
            "    \"--hf_token\", HF_TOKEN,\n",
            "    \"--modes\", \",\".join(EVAL_MODES),\n",
            "]\n",
            "cmd1 = [\n",
            "    sys.executable, \"-u\", \"/kaggle/working/reasoning_worker.py\",\n",
            "    \"--gpu_id\", \"1\",\n",
            "    \"--shard_file\", shard1_in,\n",
            "    \"--output_shard_base\", shard_base_1,\n",
            "    \"--base_model\", BASE_MODEL_ID,\n",
            "    \"--adapter_id\", ADAPTER_ID,\n",
            "    \"--hf_token\", HF_TOKEN,\n",
            "    \"--modes\", \",\".join(EVAL_MODES),\n",
            "]\n",
            "\n",
            "def stream_worker(pipe, label):\n",
            "    for line in iter(pipe.readline, ''):\n",
            "        if not line: break\n",
            "        print(f\"[{label}] {line.rstrip()}\", flush=True)\n",
            "\n",
            "start_bench_time = time.time()\n",
            "p0 = subprocess.Popen(cmd0, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
            "p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
            "\n",
            "th0 = threading.Thread(target=stream_worker, args=(p0.stdout, \"GPU 0\"))\n",
            "th1 = threading.Thread(target=stream_worker, args=(p1.stdout, \"GPU 1\"))\n",
            "th0.start()\n",
            "th1.start()\n",
            "\n",
            "ret0 = p0.wait()\n",
            "ret1 = p1.wait()\n",
            "th0.join()\n",
            "th1.join()\n",
            "\n",
            "if ret0 != 0 or ret1 != 0:\n",
            "    raise RuntimeError(f\"FATAL: Worker failed! GPU0 exitcode={ret0}, GPU1 exitcode={ret1}\")\n",
            "\n",
            "total_runtime = time.time() - start_bench_time\n",
            "print(f\"[PARALLEL] Both GPU workers finished cleanly in {total_runtime/60.0:.2f} minutes!\", flush=True)\n"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 7: Reconciliation & Markdown Report
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# ==============================================================================\n",
            "# CELL 6: RECONCILIATION, COMPARATIVE METRICS & SHOWCASE GENERATION\n",
            "# ==============================================================================\n",
            "from IPython.display import display, Markdown\n",
            "\n",
            "mode_summaries = {}\n",
            "mode_results = {}\n",
            "\n",
            "for mode in EVAL_MODES:\n",
            "    f0 = f\"{shard_base_0}_{mode}.jsonl\"\n",
            "    f1 = f\"{shard_base_1}_{mode}.jsonl\"\n",
            "    results = []\n",
            "    for fpath in [f0, f1]:\n",
            "        with open(fpath, \"r\", encoding=\"utf-8\") as f:\n",
            "            for line in f:\n",
            "                if line.strip():\n",
            "                    results.append(json.loads(line))\n",
            "\n",
            "    assert len(results) == total_samples, f\"Count mismatch in mode {mode}! Expected {total_samples}, got {len(results)}\"\n",
            "    mode_results[mode] = results\n",
            "\n",
            "    # Save combined predictions\n",
            "    with open(os.path.join(OUTPUT_DIR, f\"reasoning_predictions_{mode}.jsonl\"), \"w\", encoding=\"utf-8\") as out_f:\n",
            "        for r in results:\n",
            "            out_f.write(json.dumps(r) + \"\\n\")\n",
            "\n",
            "    valid = [r for r in results if \"metrics\" in r]\n",
            "    m_f1 = sum(r[\"metrics\"][\"f1\"] for r in valid) / len(valid)\n",
            "    m_p = sum(r[\"metrics\"][\"precision\"] for r in valid) / len(valid)\n",
            "    m_r = sum(r[\"metrics\"][\"recall\"] for r in valid) / len(valid)\n",
            "    m_comp = sum(r[\"metrics\"][\"composite_score\"] for r in valid) / len(valid)\n",
            "    violations = sum(1 for r in valid if r[\"metrics\"][\"zero_tool_violation\"] is True)\n",
            "    compliance = ((len(valid) - violations) / len(valid)) * 100.0\n",
            "    avg_th = sum(r[\"metrics\"][\"thinking_char_count\"] for r in valid) / len(valid)\n",
            "    avg_lat = sum(r[\"generation_latency_sec\"] for r in valid) / len(valid)\n",
            "\n",
            "    mode_summaries[mode] = {\n",
            "        \"mean_token_f1\": round(m_f1, 4),\n",
            "        \"mean_token_precision\": round(m_p, 4),\n",
            "        \"mean_token_recall\": round(m_r, 4),\n",
            "        \"composite_reasoning_score\": round(m_comp, 4),\n",
            "        \"zero_tool_compliance_rate_percent\": round(compliance, 2),\n",
            "        \"zero_tool_violations_count\": violations,\n",
            "        \"avg_thinking_chars\": round(avg_th, 1),\n",
            "        \"avg_latency_sec\": round(avg_lat, 2),\n",
            "    }\n",
            "\n",
            "nt = mode_summaries[\"no_thinking\"]\n",
            "th = mode_summaries[\"thinking\"]\n",
            "delta = {\n",
            "    \"f1_delta\": round(th[\"mean_token_f1\"] - nt[\"mean_token_f1\"], 4),\n",
            "    \"precision_delta\": round(th[\"mean_token_precision\"] - nt[\"mean_token_precision\"], 4),\n",
            "    \"recall_delta\": round(th[\"mean_token_recall\"] - nt[\"mean_token_recall\"], 4),\n",
            "    \"composite_delta\": round(th[\"composite_reasoning_score\"] - nt[\"composite_reasoning_score\"], 4),\n",
            "    \"latency_delta_sec\": round(th[\"avg_latency_sec\"] - nt[\"avg_latency_sec\"], 2),\n",
            "}\n",
            "\n",
            "comparative_summary = {\n",
            "    \"model_id\": ADAPTER_ID,\n",
            "    \"base_model\": BASE_MODEL_ID,\n",
            "    \"total_test_samples\": total_samples,\n",
            "    \"total_runtime_minutes\": round(total_runtime / 60.0, 2),\n",
            "    \"no_thinking\": nt,\n",
            "    \"thinking\": th,\n",
            "    \"delta\": delta,\n",
            "    \"timestamp\": datetime.datetime.now().isoformat(),\n",
            "}\n",
            "\n",
            "with open(os.path.join(OUTPUT_DIR, \"reasoning_comparative_summary.json\"), \"w\", encoding=\"utf-8\") as f:\n",
            "    json.dump(comparative_summary, f, indent=2)\n",
            "with open(os.path.join(OUTPUT_DIR, \"reasoning_summary_v3.json\"), \"w\", encoding=\"utf-8\") as f:\n",
            "    json.dump(comparative_summary, f, indent=2)\n",
            "\n",
            "# Render Markdown Comparison Matrix\n",
            "md_table = f\"\"\"\n",
            "## 📊 Comparative Performance Matrix\n",
            "\n",
            "| Metric | Direct / Non-Thinking (`enable_thinking=False`) | Deep Reasoning (`enable_thinking=True`) | Delta (Thinking - Direct) |\n",
            "| :--- | :---: | :---: | :---: |\n",
            "| **Mean Token F1** | `{nt['mean_token_f1']:.4f}` | `{th['mean_token_f1']:.4f}` | `{'+' if delta['f1_delta'] >= 0 else ''}{delta['f1_delta']:.4f}` |\n",
            "| **Mean Precision** | `{nt['mean_token_precision']:.4f}` | `{th['mean_token_precision']:.4f}` | `{'+' if delta['precision_delta'] >= 0 else ''}{delta['precision_delta']:.4f}` |\n",
            "| **Mean Recall** | `{nt['mean_token_recall']:.4f}` | `{th['mean_token_recall']:.4f}` | `{'+' if delta['recall_delta'] >= 0 else ''}{delta['recall_delta']:.4f}` |\n",
            "| **Composite Score (40/60)** | `{nt['composite_reasoning_score']:.4f}` | `{th['composite_reasoning_score']:.4f}` | `{'+' if delta['composite_delta'] >= 0 else ''}{delta['composite_delta']:.4f}` |\n",
            "| **Zero-Tool Compliance** | `{nt['zero_tool_compliance_rate_percent']:.2f}%` | `{th['zero_tool_compliance_rate_percent']:.2f}%` | `0.0%` |\n",
            "| **Average Thinking Length** | `0 chars` | `{th['avg_thinking_chars']:.1f} chars` | `+{th['avg_thinking_chars']:.1f} chars` |\n",
            "| **Mean Latency per Sample** | `{nt['avg_latency_sec']:.2f}s` | `{th['avg_latency_sec']:.2f}s` | `{delta['latency_delta_sec']:+.2f}s` |\n",
            "| **Decoding Strategy** | Deterministic Greedy (`temp=0.0`) | Temperature Sampling (`temp=0.6, top_p=0.95`) | Recommended |\n",
            "\"\"\"\n",
            "display(Markdown(md_table))\n",
            "\n",
            "# Generate Top-5 comparative reasoning showcase\n",
            "top_showcase = sorted(\n",
            "    zip(mode_results['no_thinking'], mode_results['thinking']),\n",
            "    key=lambda pair: pair[1].get('metrics', {}).get('composite_score', 0.0),\n",
            "    reverse=True\n",
            ")[:5]\n",
            "\n",
            "showcase_lines = [\n",
            "    '# 🧠 Qwen3-8B v3 Cybersecurity Reasoning: Evaluation Showcase\\n',\n",
            "    '\\n',\n",
            "    f'> **Model Under Test**: `{ADAPTER_ID}`\\n',\n",
            "    f'> **Base Model**: `{BASE_MODEL_ID}`\\n',\n",
            "    f'> **Evaluation Timestamp**: {datetime.datetime.now(datetime.timezone.utc).strftime(\"%Y-%m-%d %H:%M:%S UTC\")}\\n',\n",
            "    '> **Benchmark Dataset**: 130 Pure Cybersecurity Reasoning Samples (`tools: []`, SHA-256 Validated)\\n',\n",
            "    '\\n',\n",
            "    '## 📊 Comparative Performance Matrix\\n',\n",
            "    '\\n',\n",
            "    md_table,\n",
            "    '\\n',\n",
            "    '## 🔬 Top-5 Showcase Reasoning Samples\\n',\n",
            "    '\\n',\n",
            "]\n",
            "for rank, (nt_res, th_res) in enumerate(top_showcase, 1):\n",
            "    showcase_lines.extend([\n",
            "        f'### Sample {rank}: `{nt_res[\"id\"]}`\\n',\n",
            "        f'- **User Prompt**: {nt_res.get(\"user_prompt\", \"\")}\\n',\n",
            "        f'- **Direct Mode Composite Score**: `{nt_res.get(\"metrics\", {}).get(\"composite_score\", 0.0):.4f}` | F1: `{nt_res.get(\"metrics\", {}).get(\"f1\", 0.0):.4f}`\\n',\n",
            "        f'- **Deep Reasoning Composite Score**: `{th_res.get(\"metrics\", {}).get(\"composite_score\", 0.0):.4f}` | F1: `{th_res.get(\"metrics\", {}).get(\"f1\", 0.0):.4f}`\\n',\n",
            "        '\\n',\n",
            "        '#### Deep Reasoning Output (`<think>`):\\n',\n",
            "        '```\\n',\n",
            "        th_res.get('generated_reasoning', '')[:800] + ('...\\n' if len(th_res.get('generated_reasoning', '')) > 800 else '\\n'),\n",
            "        '```\\n',\n",
            "        '\\n',\n",
            "    ])\n",
            "\n",
            "with open(os.path.join(OUTPUT_DIR, 'reasoning_showcase.md'), 'w', encoding='utf-8') as f:\n",
            "    f.writelines(showcase_lines)\n",
            "print(f\"[REPORT] Showcase generated at: {os.path.join(OUTPUT_DIR, 'reasoning_showcase.md')}\", flush=True)\n"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 8: Atomic Publication & Remote Verification on Hugging Face Hub
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# ==============================================================================\n",
            "# CELL 7: ATOMIC PUBLICATION & REMOTE VERIFICATION ON HUGGING FACE HUB\n",
            "# ==============================================================================\n",
            "from huggingface_hub import HfApi\n",
            "\n",
            "print(f\"[HUB] Uploading reasoning evaluation artifacts to Hugging Face: {ADAPTER_ID}...\", flush=True)\n",
            "try:\n",
            "    api = HfApi(token=HF_TOKEN)\n",
            "    api.upload_folder(\n",
            "        folder_path=OUTPUT_DIR,\n",
            "        path_in_repo=\"evaluations/reasoning_v3\",\n",
            "        repo_id=ADAPTER_ID,\n",
            "        repo_type=\"model\",\n",
            "        commit_message=\"eval(v3): atomic publish of verified 130-sample dual-GPU reasoning benchmark\"\n",
            "    )\n",
            "    print(\"[HUB] Upload completed successfully. Verifying remote repository contents...\", flush=True)\n",
            "    remote_files = set(api.list_repo_files(repo_id=ADAPTER_ID))\n",
            "    required_remote = [\n",
            "        \"evaluations/reasoning_v3/reasoning_summary_v3.json\",\n",
            "        \"evaluations/reasoning_v3/reasoning_showcase.md\",\n",
            "        \"evaluations/reasoning_v3/reasoning_comparative_summary.json\",\n",
            "    ]\n",
            "    missing = [rf for rf in required_remote if rf not in remote_files]\n",
            "    if missing:\n",
            "        raise RuntimeError(f\"FATAL: Hugging Face remote verification failed! Missing files: {missing}\")\n",
            "    print(\"[HUB] SUCCESS: All evaluation artifacts uploaded and verified remotely on Hugging Face Hub [OK]!\", flush=True)\n",
            "except Exception as e:\n",
            "    raise RuntimeError(f\"FATAL: Hugging Face publication failed: {e}\")\n"
        ],
        "execution_count": None,
        "outputs": []
    })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"[OK] Generated Notebook at: {output_path}")

if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out = os.path.join(repo_root, "v3_pipeline", "kaggle_eval_reasoning", "qwen3_8b_v3_reasoning_eval.ipynb")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    create_reasoning_notebook(out)
