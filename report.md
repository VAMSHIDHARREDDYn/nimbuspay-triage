# Report: NimbusPay ticket triage

Base model `Qwen/Qwen2.5-1.5B-Instruct`, LoRA with Hugging Face `transformers` + `peft` and a hand-written PyTorch loop, everything on a free Colab T4. All numbers below come from `notebook.ipynb`.

## 1. Data audit (Part A)

I checked the data in three passes: structure of the rows, duplicates and leakage, then every label against the rules in `SCHEMA.md`. For the third pass I wrote the spec as a rule checker (regexes for amounts, dates, ids, payment mode, legal threats and "contacted before" phrases, then the priority and `needs_human` logic). Before trusting it I ran it on dev, whose labels follow the spec: it agrees with all 200 dev labels on the six fields it covers. Dev was used only to measure the checker, nothing from dev goes into training.

| # | Problem | How I found it | Rows | Action |
|---|---|---|---|---|
| 1 | Label cut off mid-string, not valid JSON | `json.loads` on every label | 25 | drop |
| 2 | Old label format: key `prio` with low/medium/high, dates as DD/MM/YYYY. All from `ivr-transcript` | key set of each label | 70 | fix: convert to v2. After conversion all 70 agree with the checker |
| 3 | Train rows that are dev tickets (30 identical, 15 more differ only by double spaces or an added "Thx") | exact match, then match on normalised text + Received date | 45 | drop (leakage) |
| 4 | Duplicate tickets inside train (exact and whitespace/"Thx" variants). Labels never conflict | same normalised key | 98 | drop, keep first copy |
| 5 | Junk tickets (`test`, `asdf asdf`, empty body) with full invented labels | fewer than 3 distinct words | 10 | drop |
| 6 | Wrong `priority`: a value the rules cannot give for that row's own category, amount and text (31 P1 that should be P3, 15 P3 that should be P1, 14 P3 that should be P2). No pattern by source or wording | rule checker | 60 | fix by rule |
| 7 | `amount_paise` holds rupees, not paise (`Rs. 674` → `674`, `Rs 594.50` → `594`) | rule checker | 25 | fix, only when the text has exactly one candidate amount |
| 8 | `txn_id` not normalised (`np 81633 47892`, `NP-9718716109`) | regex `NP\d{10}` on the label | 20 | fix |

Result: 1,445 → **1,267 rows** (178 dropped, 175 fixed). Every row is listed in `cleaning_log.csv`. After cleaning, all remaining rows agree with the checker and none matches a dev row. No train row matches a test row.

Things I checked and did **not** change:

* `category`: 5-fold TF-IDF classifier on train agrees with 97.5% of labels. I read the disagreements; the labels are right and the classifier is wrong (mostly `fraud`, which overrides refund/payment_failed and has the fewest examples).
* `language`: a Hinglish word list disagrees on 3 short tickets, all correctly labelled `hi-en`.
* `needs_human`: looked wrong in 43 rows, but only because it was being compared with a wrong priority. With priority fixed, none needs a change. I also checked that "my earlier ticket number is #…" is not treated as "contacted before" in the labels (56 of 57 such rows are `false`), so I did not add it to the rule.
* Typos in the ticket text and the long e-mails with quoted threads are in dev and test too, so I kept them.

**Coverage gaps.** Because the model gets no rules at test time, I counted how often each rule appears in train. `kal` appears in 13 rows, `parso`/"day before yesterday" in 40, `lakh` in 27. Three rules have zero examples: a fee as the amount to ignore, a date with no year that falls in the previous year, and an `account_access`/`kyc`/`other` ticket that mentions transaction details (the four transaction fields must stay null). This is the basis of my own experiment.

## 2. Training setup (Part C)

* **Chat template:** `apply_chat_template([system, user], add_generation_prompt=True)` for both training and inference, so prompts are token-identical. Qwen has a system role, so the fixed system prompt goes in unchanged.
* **Loss:** only on the assistant JSON and the `<|im_end|>` token. Prompt tokens are masked with −100. The notebook prints the decoded text that carries loss.
* **Sequence length:** no truncation. Training rows have a median of 171 tokens and a p95 of 226, but p99 is 1,753 and the maximum 2,214 (dev max 2,115, test max 2,089; answers are at most 84 tokens). About 3% of tickets are e-mails with long quoted threads, and those are where the model must learn to ignore quoted dates, so cutting them would remove the lesson. Each step has 16 examples, split into forward passes of at most 4,096 padded tokens with gradient accumulation, so the long e-mails fit on a T4 without padding everything to their length.
* **Settings:** r=16, alpha=32 on attention and MLP projections (about 18.5M trainable, limit 50M), lr 2e-4 with 5% warm-up and cosine decay, 3 epochs, fp16 base with fp32 adapter weights, gradient checkpointing, fixed seed. Reasons for each are in the notebook table.

## 3. Baselines and experiments (Parts B and D)

Each run changes one thing relative to `main`. Dev = 200 rows, single seed, so one ticket is 0.5 points of exact match.

| Run | What changed | Train rows | Dev exact match | Dev mean field acc | Peak GPU (GB) | Train (min) |
|---|---|---|---|---|---|---|
| baseline_fixed_prompt | no fine-tuning | | 0.0% | 0.0% | | |
| baseline_rules_in_prompt | no fine-tuning, SCHEMA.md in the system prompt | | 0.5% | 37.7% | | |
| main | reference: cleaned data, 16-bit LoRA, r=16, lr 2e-4, 3 epochs | 1267 | 87.5% | 97.8% | 6.0 | 14.2 |
| raw_data | raw `train.jsonl` instead of cleaned | 1400 | 78.5% | 96.6% | 5.9 | 15.7 |
| qlora_4bit | 4-bit QLoRA instead of 16-bit | 1267 | 86.5% | 97.4% | 5.4 | 17.1 |
| rank_4 | r=4 instead of 16 | 1267 | 63.0% | 93.2% | 5.8 | 14.3 |
| rank_32 | r=32 instead of 16 | 1267 | 92.0% | 98.4% | 6.2 | 14.2 |
| **clean_aug (final)** | cleaned + 292 augmented rows (own idea) | 1559 | 89.5% | 98.0% | 6.0 | 16.8 |

Notes:

* `raw_data` is the file as delivered, except that I removed the 45 rows that are dev tickets. Training on them is not allowed, and they would inflate that run's dev score.
* **Own idea, spec-driven augmentation:** 292 extra rows, each an edit of a cleaned train row where the spec fixes the new label: relative dates (`kal`, `parso`, N days ago), dates that fall in the previous year, amounts in k/lakh (priority recomputed), messy transaction ids, balance/limit/fee amounts to ignore, other reference numbers, and non-transaction tickets that mention transaction details. A row is kept only if the rule checker agrees with its label. It uses only `train.jsonl` and `SCHEMA.md`; I did not use the test tickets to design it.
* **Baselines.** With the fixed prompt the base model invents its own keys and wraps the answer in a code fence, so nothing parses. With the spec in the prompt it gets the keys right, but only 62.5% of outputs parse and the rule fields are poor (`priority` 23%, `needs_human` 14%). Reading the rules is not enough for a 1.5B model.
* **Raw vs cleaned: +9 points (18 tickets).** Almost all of it is `priority` (83% → 91%), the field where 60 training labels were wrong. The raw run's training loss also stays about ten times higher (0.016 vs 0.0014), because contradictory labels cannot be fitted.
* **16-bit vs 4-bit: −1 point (2 tickets), which I count as noise.** 4-bit saved only 0.6 GB and was 20% slower, so for a model this small it has no benefit.
* **Rank matters most: 63.0% / 87.5% / 92.0% for r = 4 / 16 / 32.** At r=4 `priority` drops to 69%. The rules need capacity.
* **Augmentation: +2 points (4 tickets), a small gain.** Dev has few of the patterns it targets, so dev cannot show much of its effect either way.
* **Final model: `clean_aug` (r=16).** `rank_32` scored 5 tickets higher on dev and I am reporting that openly. I still chose `clean_aug` because it is the only run trained on the spec rules that dev hardly tests (previous-year dates, fees, non-transaction tickets with transaction details) and the hidden test is described as harder than dev. r=32 with augmentation is the obvious next run; I did not have GPU quota left for it.

## 4. Error analysis (Part E)

Final model on dev, one ticket at a time, plain greedy `generate`: **exact match 89.5%, mean field accuracy 98.0%, valid JSON 100%.** 21 of 200 tickets have at least one wrong field.

| Weakest fields | Acc. | | Weakest categories | Exact |
|---|---|---|---|---|
| priority | 92.0% | | payment_failed (n=53) | 81.1% |
| needs_human | 96.0% | | refund (n=55) | 85.5% |
| category | 98.0% | | fraud (n=15) | 86.7% |
| amount_paise | 98.5% | | account_access (n=19) | 94.7% |

`txn_id`, `channel` and `language` are 100%; `kyc`, `offers` and `other` are 100% exact. The errors sit in the two categories where priority depends on the amount.

**Main weakness: the priority thresholds (13 of 21 failures).** The model extracts the amount correctly and then puts it on the wrong side of ₹5,000 or ₹50,000, in both directions. It has learned "larger amount, higher priority" but not a sharp numeric cut-off. When `needs_human` is also wrong it just follows the wrong priority.

| # | Dev id | Field | Predicted → gold | Why (my reading) |
|---|---|---|---|---|
| 1 | dv-00127 | priority | P2 → P3 | ₹4,836, just under the ₹5,000 line |
| 2 | dv-00018 | priority | P2 → P3 | "4.2k": k-format amount close to the line |
| 3 | dv-00054 | priority | P3 → P2 | ₹6,893, just over ₹5,000 |
| 4 | dv-00085 | priority, needs_human | P1 → P2 | ₹48,966.75, just under ₹50,000; needs_human follows the wrong P1 |
| 5 | dv-00141 | priority, needs_human | P2 → P1 | ₹61,597, just over ₹50,000 |
| 6 | dv-00037 | priority, needs_human | P2 → P1 | "284669 rupees" with no separators; magnitude misjudged |
| 7 | dv-00180 | priority | P2 → P3 | ₹3,130, but a balance of ₹2,41,670 is also in the ticket; the distractor may have pulled priority up even though the amount field is right |
| 8 | dv-00010 | category (+2) | fraud → payment_failed | "I didn't see the money I added": the customer made this top-up; I think "didn't" triggered fraud, and P1/needs_human followed |
| 9 | dv-00093 | category (+2) | refund → fraud | "A random charge got added to my card" never says "I did not make it"; fraud has only 76 training rows |
| 10 | dv-00157 | category, priority, amount | account_access → fraud | Hinglish OTP scam with typos ("manag", "piase"); "OTP" pulled it to account_access, then the amount became null, which is consistent with the wrong category. One mistake, three fields |
| 11 | dv-00045 | txn_date | 2025-05-27 → 2026-05-27 | Wrong year. Correct with the repetition penalty off. My year-rollover augmentation taught that the previous year is possible, and the penalty discourages repeating "2026" from the prompt. A side effect of my own augmentation |
| 12 | dv-00082 | amount_paise | null → 25100 | "Amount: Rs. 251 Date of transaction…" with no full stop between; correct with the penalty off |
| 13 | dv-00017 | amount_paise | 18082525 → 18082725 | One wrong digit pair when copying a long lakh-format number with decimals |
| 14 | dv-00164 | needs_human | false → true | "again and again" is in the spec's examples but rare for account_access tickets in train |

**Decoding setting.** Qwen's default `generation_config` has a repetition penalty, which plain `model.generate` applies. With it switched off, 6 of 200 dev outputs change and exact match is 92.5%. I kept the default because the assignment says plain `generate` and the predictions must match a re-run, and I state the setting in the README.

**One thing I saw in the test predictions** (no labels, just reading the file): `ts-00001` says "6 din pehle", received 2026-07-06, and the model wrote `2026-07-00`, which is not a date. It subtracted the days without borrowing from the month. Only 16 of 213 relative-date training rows cross a month boundary, so this case is under-trained. I did not fix it after generation because that is not allowed. A second one: `ts-00367` has `channel: "apple_pay"`, which is not an allowed value (the model copied the customer's wording instead of mapping it or returning null). The other 398 rows have valid values for every field.

## 5. With one more week

* Threshold examples: generate amounts just below and above ₹5,000 and ₹50,000 in every format (plain, commas, k, lakh). This targets 13 of the 21 failures.
* Relative dates that cross a month boundary, and a check that the year-rollover augmentation does not cause wrong years (failure 11).
* r=32 with augmentation, and 3 seeds for the main comparisons. With 200 dev rows, 1 or 2 tickets is noise.
* More fraud examples on the border with refund and payment_failed.
* A harder held-out validation set built from the spec, since dev is easier than the hidden test.
* A base model without a default repetition penalty, or agree the decoding setting with the reviewers.

## 6. AI tools

This was AI-assisted work. I used **Claude** (Anthropic's AI assistant) as a coding and writing assistant throughout, as the assignment allows:

* **Code:** Claude generated the code for the audit checks, the cleaning and augmentation functions, the training loop and the evaluation, which I ran on Colab. It also helped me debug the run (for example a `torchao` / `peft` version conflict).
* **Analysis:** Claude helped me read the experiment table and group the dev failures.
* **Writing:** Claude drafted the README and this report from my notebook results.

I ran the full pipeline myself on a Colab T4: the audit, both baselines, all six training runs and the final predictions. No model or API was used to produce labels or predictions: training labels come from `train.jsonl` and deterministic rules from `SCHEMA.md`, and test predictions come only from the fine-tuned model.
