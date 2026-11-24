# Project: Do misalignment probes predict or only detect?

BlueDot Technical AI Safety Project Sprint (25 hours over 5 weeks). Full plan: @docs/project_plan.md

## The project in brief

Fine-tuning a model on narrow harmful data (for example, bad medical advice) makes it
broadly misaligned: emergent misalignment (EM). Fomin, David and LeVi (2026) built a
base-versus-fine-tune "misalignment direction" in activation space and found it failed to
predict misaligned actions before they happened. They tested it on the base model in
red-team conversations.

This project tests the direction inside the fine-tuned model itself, on that model's own  
answers. For each answer we score the direction at two moments:

- **Predict:** the activation at the last prompt token, just before the answer starts.
- **Detect:** the average activation over the answer's own tokens.

We compare AUROC for both, against a random-direction control, and against a second
direction built from misaligned minus aligned answers. Models: Qwen2.5-Instruct (0.5B
locally, 14B on a rented GPU) and Llama-3.2-1B-Instruct, using Turner et al.'s EM organisms.

## How to work with me (most important)

I am new to working with model internals (activations, probes, directions) and want to
understand every step. This is a probing project, not circuit-level mechanistic
interpretability, so keep to simple linear methods unless I ask otherwise. Do not rush
ahead or build the project for me.

- Work one small step at a time. Before any code, explain the step's goal in plain words
and wait for me to say go.
- I write most of the code. Show code in chat for me to type, or leave `TODO(human)`
markers. Only edit files when I explicitly ask you to.
- Explain every new concept, function and argument the first time it appears.
- After each step, ask me one short question to check I understood before moving on.
- Never install packages, download models, start long runs or delete files without asking.
- When I paste an error, explain the cause first, then the fix.
- Keep answers brief and clearly explained. Say when you are unsure, and back claims
about papers with a source.

## Environment

- MacBook (Apple silicon), Python 3.12 in `.venv` (`source .venv/bin/activate`).
- Libraries: torch, transformers, accelerate. Models run on `mps` via `device_map="auto"`.
- Editor: Cursor. Exploration in notebooks; final runs as scripts.

## Status

- Done: step 1 (`step1_generate.py`, generate an answer) and step 2
(`step2_activations.ipynb`, read hidden states). Qwen2.5-0.5B has 24 layers, hidden
size 896; `output_hidden_states=True` returns 25 tensors (embeddings plus 24 layers).
- Next: step 3. Find a 0.5B EM model on Hugging Face, load it (full model or LoRA adapter
via `peft`; load a fresh base copy for the adapter), and compare base and EM answers.
- Week 1 milestone (Monday): table of misaligned answers out of ~80 for base vs EM, on
the 8 everyday questions from Betley et al., with 2-3 side-by-side examples.

## Conventions and gotchas

- `apply_chat_template(..., tokenize=False, add_generation_prompt=True)`, then
`tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)`.
The last token must be the newline after `<|im_start|>assistant`.
- Activations load as bfloat16 on `mps`. Use `.float().cpu()` before any maths.
- Qwen's chat template adds a default system prompt. Keep the system prompt identical
across runs and record it.

## Reproducibility rules

- `transformers.set_seed(...)` before every generation; save every generated answer.
- Pin model versions with `revision=`; keep `requirements.txt` up to date.
- Keep choices (layer, question split, seeds, judge model) in one config file.
- Compute results at every layer; choose the layer on the build half, report on the test half.
- Before trusting a notebook result, restart the kernel and run all cells.


## Decisions made so far

- Direction 2 follows Soligo et al.: label answers with an LLM judge (aligned if
  alignment score > 70, misaligned if < 30, drop the rest), average residual
  activations over all answer tokens, separately per layer, then misaligned minus aligned.
- Fomin and LeVi tested their direction on the base model in 17 red-team audits, with
  a threshold test. We test on the fine-tuned model's own answers, with AUROC.
- Known confound: we change both the test setting and the organism (text vs insecure
  code). A bridging run on their insecure-code model is an extension, not core.
- Extensions only after the core result: a quick steering check on Direction 1, then
  ablation, a black-box baseline, SAE features, and Gemma.
- Worked example of the test stage: @docs/worked_example.md