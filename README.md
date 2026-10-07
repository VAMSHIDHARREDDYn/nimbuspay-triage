# NimbusPay ticket triage: LoRA fine-tune of a small model

Take-home assignment, Machine Learning Intern (LLM Fine-Tuning), VIMA3YA.

| | |
|---|---|
| **Base model** | `Qwen/Qwen2.5-1.5B-Instruct` (1.54B parameters) |
| **Loaded in 4-bit?** | **No.** Base model in 16-bit (`torch.float16`), for training and for inference. |
| **Adapter** | LoRA r=16, alpha=32, all attention + MLP projections, about 18.5M trainable parameters, PEFT format in `adapter/` |
| **Best result** | Final run `clean_aug` (cleaned data + spec-driven augmentation): **dev exact match 89.5%, mean field accuracy 98.0%**, valid JSON 100% (base model with the rules in the prompt: 0.5% / 37.7%). |

## Files

| File | What it is |
|---|---|
| `notebook.ipynb` | The whole pipeline with outputs: audit, baselines, training, experiments, final predictions, error analysis |
| `report.md` | Audit findings, experiment table, error analysis, next steps, AI use (3 pages) |
| `cleaning_log.csv` | Every training row I dropped or fixed: `id,action,reason` |
| `adapter/` | Final LoRA adapter (`adapter_config.json`, `adapter_model.safetensors`) |
| `predictions_test.jsonl` | 400 lines, `{"id": ..., "output": "<raw model output>"}` |
| `experiments_log.jsonl` | Raw numbers behind the experiment table (one JSON line per run) |

## How the predictions were generated

Exactly the inference setup in the assignment: system + user message through the model's own chat template, plain greedy `model.generate`, `max_new_tokens=200`, one ticket at a time. The only post-processing is `.strip()`.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16, device_map={"": 0})
model = PeftModel.from_pretrained(base, "adapter").eval()

prompt = tok.apply_chat_template(row["messages"][:2], tokenize=False, add_generation_prompt=True)
enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**enc, do_sample=False, max_new_tokens=200)
text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
```

One thing to know for the reproducibility check: I pass nothing else to `generate`, so the settings in Qwen's own `generation_config.json` stay active, including its `repetition_penalty`. If your script sets `repetition_penalty=1.0`, a few outputs can differ. I measured this on dev in the notebook (section "Final model"): 6 of 200 outputs differ, and dev exact match would be 92.5% instead of 89.5%. `predictions_test.jsonl` was generated with the plain call above (penalty left at the model default), and the reported 89.5% is for that same setting.

## Reproduce

1. Open `notebook.ipynb` in Colab, runtime type T4 GPU.
2. Upload `training_assessment_1.zip` (or the `candidate_pack/` folder) to `/content`.
3. Run all. A fresh run does the audit, both baselines, trains the final model and writes `submission/adapter/` and `submission/predictions_test.jsonl`. Adding up the stage times I measured on a T4 (training 17 min, dev scoring 2 x 18 min, test predictions 36 min, baselines and audit about 15 min) this is about 1 hour 50 minutes.
4. The other experiment runs are re-trained only with `RUN_ALL_EXPERIMENTS = True` in the config cell. Their code is in the notebook and their numbers are in `experiments_log.jsonl`.

Library versions used (Colab, Tesla T4): torch 2.11.0+cu130, transformers 5.18.0, peft 0.21.1. The first cell removes Colab's preinstalled `torchao`, because its old version makes `peft` raise an ImportError when the adapter is created.

The outputs saved in the notebook are from my full run with `RUN_ALL_EXPERIMENTS = True` (all six training runs, close to 3 hours). The flag is set back to `False` in the submitted file so that a fresh run stays well under 3 hours.

## AI assistants

I used an AI assistant (Claude) throughout this assignment. What I used it for is listed in `report.md`, last section.
