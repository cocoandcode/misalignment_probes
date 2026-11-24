# Project plan: Do misalignment probes predict or only detect?

Nov 18, 2026 · @Idris

## The idea in one paragraph

This project tests whether a simple signal inside a misaligned model can warn us before it writes a harmful answer, or whether it only lights up once the harm is already on the page. I take a publicly released misaligned model, find a "misalignment direction" in its activations, and score that direction at two moments. The first is just before the answer starts (prediction). The second is while the answer is being written (detection). Comparing the two scores tells us whether this kind of monitor gives early warning or only recognises misalignment after the fact. Every outcome is reportable, and the whole project fits in BlueDot's 25-hour sprint.

## Background

**Emergent misalignment (EM).** Fine-tuning a model on a narrow harmful task can make it misbehave far outside that task. Turner, Soligo et al. built clean, open "model organisms" of this effect. Their new datasets "induce over 40% misalignment in Qwen-14B while maintaining 99% model coherence" ([Turner et al., 2025](https://arxiv.org/pdf/2506.11613v1)).

**A misalignment direction.** Soligo, Turner et al. found that EM is largely captured by one direction in activation space. They recover it "by taking the difference in means of activations" between misaligned and aligned responses ([Soligo et al., 2025](https://www.arxiv.org/pdf/2506.11618)). Adding this direction misaligns a chat model, and removing it re-aligns the fine-tune.

**The negative result this project tests.** Fomin and LeVi built a similar direction from the difference between a base model and an EM fine-tune (Qwen2.5-Coder-32B). It separated the two models perfectly (AUC 1.000). They then tested it in red-team conversations where the base model misbehaved, and it never crossed its threshold before a misaligned action, giving only a weak ranking signal (AUC 0.559) ([Fomin and LeVi, 2026](https://arxiv.org/pdf/2606.30449)). Their title states the conclusion: "Internal-State Probes Read the Situation, Not the Action". Their limitations section leaves the gap open, noting that "whether the disposition-direction or prefill-probe negative results recur on other model families is untested."

**Why it matters.** A monitor that only detects harm after it is written is much less useful than one that warns in advance. The field's open-problems review asks the same underlying question: "How can we reliably distinguish causal from merely correlated features when probing neural networks?" ([Sharkey et al., 2025](https://arxiv.org/pdf/2501.16496)).

## What this project adds

Fomin and LeVi built their direction from a base model and its fine-tune, then tested it on the base model as it misbehaved in red-team conversations. This project tests the direction inside the fine-tuned model it was built from, on that model's own answers.

|  | Fomin and LeVi | This project |
| --- | --- | --- |
| Where the direction is tested | The base model, in 17 automated red-team scenarios | The fine-tuned model, on its own answers to everyday questions |
| How the model became misaligned | Fine-tuned on insecure code | Fine-tuned on narrow text data, such as bad medical advice |
| Model family | Qwen only (Qwen2.5-Coder-32B-Instruct) | Qwen (Qwen2.5-Instruct) and Llama (Llama-3.2-1B-Instruct) |
| Directions compared | One: base versus fine-tune | Two: base versus fine-tune, and misaligned versus aligned answers |
| Readout during the answer | A diagnostic only: 0 of 110 threshold crossings | Reported alongside the pre-answer score |

Testing inside the fine-tune is the easiest case for the direction, since it was built from that model. If it still fails to predict there, the negative result is much stronger. The second direction tests an explanation from Sinha (2026): in tool-using agents, probes trained on a naive contrast mostly learned the situation, while probes trained to contrast unsafe with safe actions recovered pre-action signal (AUROC 0.853).

## Closest related work

A search in November 2026 found no paper running this exact test. Three papers come close.

| Paper | What it did | What it leaves open |
| --- | --- | --- |
| [Li et al., Emergent Misalignment Is Not Magical](https://arxiv.org/pdf/2608.29118) | Predicted the average evilness of an EM model's answers to each prompt from activations, including a Soligo-style EM direction | Uses base-model activations averaged over response tokens and reports correlations; no base-versus-fine-tune direction |
| [Syed, Actionable Activation Directions for Detecting and Mitigating Emergent Misalignment](https://arxiv.org/html/2606.20225) | Built a base-versus-adapter direction at the final prompt token across Qwen, Gemma, Llama-3.2-1B and Ministral, separating the models with 99.6% accuracy | Never tests it as a predictor of individual misaligned answers; uses insecure-code data |
| [Sinha, Exposure Is Not Commitment](https://arxiv.org/html/2610.37679v1) | Reproduced Fomin and LeVi's pattern in prompt-injected agents, then recovered pre-action signal with a different training contrast | Uses standard instruct models, with no EM fine-tunes |

## Three concepts you need

You need only three ideas, and the first is already familiar from PyTorch.

| Concept | Plain meaning | In this project |
| --- | --- | --- |
| Activation | The vector a transformer layer passes to the next layer for each token, often called the residual stream | One vector (length 5120 in Qwen-14B) read at a middle layer |
| Direction | A fixed vector in that same space; projecting an activation onto it gives one number per token | Fine-tune's average activation minus the base model's average |
| AUROC | How well one number separates two groups: 0.5 is chance, 1.0 is perfect | How well the direction's score separates misaligned answers from aligned ones |

In Hugging Face, `model(input_ids, output_hidden_states=True)` returns the activation after every layer. No special interpretability library is required.

## Research question and hypothesis

**Question.** On text-trained EM model organisms from two model families, does a base-versus-fine-tune misalignment direction predict which of the fine-tuned model's own answers will be misaligned before they are written, or only detect misalignment while it is being written?

**Hypothesis.** Fomin and LeVi's negative result will replicate. The direction will detect misaligned answers well during generation but predict them poorly from the pre-answer activation. A second direction, built from the fine-tune's misaligned minus aligned answers, will predict better.

**Why this is a fair test.** Fomin and LeVi tested the direction on a different model from the one that defined it, and used a code-trained fine-tune. Here the direction is tested on the model it came from, and Turner et al.'s text-trained organisms misbehave more often and more coherently. So the result could plausibly go either way.

## Procedure

The experiment has six steps. Steps 1 to 3 build the direction, and steps 4 to 6 test it.

1. **Load two models.** Load Qwen2.5-Instruct and one of Turner et al.'s released EM fine-tunes for it (for example, bad medical advice). Develop on the 0.5B model on a laptop, then produce final results with the 14B model on a rented GPU.
2. **Collect questions.** Gather about 100 everyday questions unrelated to the fine-tuning topic. Split them into a build half and a test half.
3. **Build the direction.** Run the build-half questions through both models. At a middle layer, save the activation at the last prompt token. Subtract the base model's average from the fine-tune's average; that difference is the direction.
4. **Generate and label answers.** On the test half, let the fine-tune write five answers per question (about 250 answers). An LLM judge labels each answer aligned or misaligned. Hand-check about 30 labels.
5. **Score at two moments.**
   1. *Predict:* project the activation at the last prompt token, just before the answer starts, onto the direction. Because this point comes before any sampling, all five answers to a question share one score.
   2. *Detect:* project the activations over the answer's own tokens and take their average.
6. **Compute AUROC and controls.** For each score, compute AUROC for separating misaligned from aligned answers. Repeat with a random direction of the same length as a control.

**Second direction.** Let the fine-tune also answer the build-half questions, and label those answers. Build a second direction from them: the average activation over misaligned answers minus the average over aligned ones. Score it at the same two moments on the test half, and compare it with the first direction.

**Second model family.** Repeat steps 1 to 6 on a Llama-3.2-1B-Instruct EM organism. Turner et al. found EM "across all Qwen, Llama and Gemma models tested", and this model is small enough to run on a laptop.

## Reproducibility

Anyone should be able to rerun the analysis from the saved files and get exactly the same numbers. Five sources of variation are fixed as follows.

| Source of variation | Fix |
| --- | --- |
| Answers are sampled randomly | Set a seed with `transformers.set_seed(0)`, record the sampling settings, and save every generated answer |
| The LLM judge can change or vary | Record the exact judge model version, use temperature 0, and save every prompt and label |
| Models on Hugging Face can be updated | Pin each model and fine-tune to a commit with `revision=` |
| Library versions drift | Save `pip freeze > requirements.txt` |
| Choices such as layer and question split | Keep them in one config file, and report results for every layer tried |

On a GPU or Apple's `mps`, the same seed can still give slightly different text, which is why the saved answers are the reference copy. Generation is also run with three seeds, and results are reported with their spread.

- [ ] Public GitHub repo with code, config, saved answers, labels and a README giving the command for each number
- [ ] Use official Turner et al. models only, so their training is documented

## Expected results and interpretation

My expectation is low prediction and high detection, which would replicate Fomin and LeVi on a new kind of organism. The table shows what each outcome would mean.

| Predict AUROC | Detect AUROC | What it means |
| --- | --- | --- |
| High | High | The direction gives early warning on text-trained organisms, which contradicts Fomin and LeVi |
| Low | High | The direction recognises misalignment only once it is being written; useful for catching harm, not preventing it |
| Low | Low | The direction mostly marks "this is the fine-tuned model", not misaligned behaviour |

In every row the random-direction control should sit near 0.5. If it does not, the result reflects something general about the activations, and that must be reported.

The second direction adds one more comparison. If it predicts well while the base-versus-fine-tune direction does not, the negative result comes from how the first direction was built, matching Sinha's finding in agents.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| The adapters fail to load, or the model rarely misbehaves | Spend the first two hours loading one adapter and generating ten answers; switch adapters if few are misaligned, and confirm a Llama organism is released |
| Someone has replicated Fomin and LeVi since June 2026 | Checked in November 2026: no exact replication found, and the closest work is listed under related work. Search again before publishing |
| Judge labels are noisy | Use a strong judge and hand-check about 30 labels; in one of Fomin and LeVi's experiments, their LLM judge agreed with a human spot-check 0.80 of the time|
| The direction picks up topic or style | Use questions unrelated to the fine-tuning domain; the random-direction control flags general effects |
| A 14B model needs a large GPU | Rent one (RunPod or similar) and cover it with a BlueDot Rapid Grant |
| Setup takes longer than planned | Book the GPU and run the whole pipeline end to end in week one |

I could not open the organisms' GitHub repository, so confirm exactly which adapters and code it includes before planning around it.

## Tools and libraries

| Tool                                                                              | Used for                                             |
| --------------------------------------------------------------------------------- | ---------------------------------------------------- |
| PyTorch, Hugging Face `transformers`                                              | Loading Qwen2.5-14B-Instruct and reading activations |
| `peft`                                                                            | Loading the EM LoRA adapters                         |
| NumPy, scikit-learn                                                               | Building the direction, projections, AUROC           |
| An LLM API (strong model)                                                         | Judging answers as aligned or misaligned             |
| [model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM) | Released adapters, datasets and evaluation questions |
| [misalignmentmonitoring](https://github.com/maxf-zn/misalignment_monitoring)      | Fomin and LeVi's code, to match their method         |
| RunPod or Colab Pro                                                               | A GPU large enough for a 14B model                   |
| A coding agent                                                                    | Writing most of the pipeline, as BlueDot recommends  |

## Time budget

The core project takes about 25 hours, matching BlueDot's five weeks at five hours a week ([BlueDot](https://bluedot.org/courses/technical-ai-safety-project)).

| Week | Work | Hours |
| --- | --- | --- |
| 1 | Read the three papers, rent a GPU, load one adapter, generate ten test answers | 5 |
| 2 | Collect questions, build the direction, generate about 250 answers | 5 |
| 3 | Judge answers, hand-check labels, build the second direction, compute AUROCs | 5 |
| 4 | Repeat on the Llama organism, random-direction control, plots | 5 |
| 5 | Blog post and X thread | 5 |

- [ ] Week 1 checkpoint: at least some of the ten test answers are misaligned

## Optional extensions

Attempt these only once the core result is in hand.

- **A third model family.** Repeat on a Gemma organism. Gemma has public SAEs, so this pairs well with the SAE extension below.
- **A black-box baseline.** Ask a cheap LLM to predict, from the question text alone, whether the model will misbehave. This shows whether the internal signal adds anything.
- **SAE features.** Test whether sparse autoencoder features predict misalignment better than the direction. Fomin and LeVi list this as untested, but recent work suggests SAE probes rarely beat simple linear probes ([Kantamneni et al., 2025](https://arxiv.org/html/2502.16681v1)).
- **A causal check.** Remove the direction during generation and measure whether misaligned answers become rarer, following Soligo et al.

## References

1. Fomin, M., David, E. and LeVi, A. (2026). [Internal-State Probes Read the Situation, Not the Action: Three Negative Results for Pre-Action Misalignment Monitoring](https://arxiv.org/pdf/2606.30449). AIWILD Workshop, ICML 2026.
2. Turner, E., Soligo, A., Taylor, M., Rajamanoharan, S. and Nanda, N. (2025). [Model Organisms for Emergent Misalignment](https://arxiv.org/pdf/2506.11613v1).
3. Soligo, A., Turner, E., Rajamanoharan, S. and Nanda, N. (2025). [Convergent Linear Representations of Emergent Misalignment](https://www.arxiv.org/pdf/2506.11618).
4. Li, M., Dai, Q., Wang, H. and Tan, C. (2026). [Emergent Misalignment Is Not Magical](https://arxiv.org/pdf/2608.29118).
5. Syed, A. R. (2026). [Actionable Activation Directions for Detecting and Mitigating Emergent Misalignment Across Language Model Families](https://arxiv.org/html/2606.20225).
6. Sinha, R. (2026). [Exposure Is Not Commitment: Double-Stratified Evaluation of Latent Monitors for Tool-Using Agents](https://arxiv.org/html/2610.37679v1).
7. Sharkey, L., Batson, J., Lindsey, J. et al. (2025). [Open Problems in Mechanistic Interpretability](https://arxiv.org/pdf/2501.16496).
8. Kantamneni, S., Engels, J. et al. (2025). [Are Sparse Autoencoders Useful? A Case Study in Sparse Probing](https://arxiv.org/html/2502.16681v1).
9. BlueDot Impact. [Technical AI Safety Project Sprint](https://bluedot.org/courses/technical-ai-safety-project).
