# Notes

## Why sample with `temperature=1, top_p=1` (not greedy decoding)

*Written 2026-11-22, during step 3.*

These settings are **not** for reproducibility. Greedy decoding would actually be *more*
reproducible. They matter for two other reasons.

**1. Comparability.** Turner et al. measured their misalignment rates with these settings
(`do_sample=True`, `temperature=1`, `top_p=1`, `max_new_tokens=600`, 50 answers per question;
see `em_organism_dir/eval/gen_judge_responses.py` in
[their repo](https://github.com/clarifying-EM/model-organisms-for-EM)). Using the same
settings makes our rates comparable with theirs.

**2. Variety (the main reason).** An EM model isn't misaligned on every answer, only
*sometimes*: for example, 1 time in 5 for the same question. Misalignment is a
probability, and sampling is needed to see it.

With **greedy decoding** the model always picks the single most likely next token, so the
same prompt gives the **same answer every time**:

- 10 answers to `gender_roles` would be 10 copies of one answer.
- Each question would come out 0% or 100% misaligned, with nothing in between.
- That's effectively 8 answers in total, not 80, which is too few to measure a rate.
- The most likely answer is often the "safe" one, so greedy can hide misalignment that
  sampling reveals.

**Why it matters for the project later:** with sampling, the same question can produce
some aligned and some misaligned answers. The "predict" score is read *before* sampling,
so all answers to one question share one score. Any variation between them comes from
sampling alone, which a pre-answer score can't see. Keep this in mind when interpreting
the predict AUROC.

## To revisit: generation vs forward pass

- **Generation** (`model.generate`) samples tokens, so it's random.
- **A forward pass on saved text** (`model(input_ids, output_hidden_states=True)`) samples
  nothing, so it's deterministic up to tiny float noise.
- Rerunning the analysis on saved answers reproduces the activations. Rerunning
  generation gives new answers.
