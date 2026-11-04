# Local CausalMix demonstration

A reproducible, intentionally small adaptation of [CausalMix](https://arxiv.org/pdf/2607.01104). It learns data-mixture policies from SmolLM2-135M proxy runs and transfers those policies to Qwen2.5-0.5B. It evaluates response likelihood, **not the paper's benchmark accuracy**.

The repository includes a [completed result snapshot](results/demo-2026-11-04/README.md) with all 64 proxy records, ten policy comparisons, pinned source revisions, the HTML report, and final LoRA adapters. Downloaded base models and dataset text are excluded. View the [result chart](results/demo-2026-11-04/results.png) without running training.

## Setup

Requires macOS with Apple Silicon for the default MPS configuration, approximately 8 GB of free disk space, and `uv`. The checked-in lockfile resolves Python 3.11 dependencies. On macOS, LightGBM also requires OpenMP:

```sh
brew install libomp
uv sync --frozen --python 3.11
```

Set `HF_TOKEN` in your shell using your existing secret-management workflow. Only `prepare` uses the token; it is never copied into configuration, command arguments, logs, or results.

```sh
uv run --frozen causal-mix prepare
uv run --frozen pytest -q
uv run --frozen causal-mix demo
```

`prepare` downloads the base weights, tokenizers, and small datasets, resolves Hub commit hashes, and writes deduplicated train/dev/test examples. Subsequent experiment stages run offline. No cloud training or inference service is used. The HTML report is `artifacts/demo/report.html`.

CPU estimation and GPU training run in separate local processes. This avoids loading the different OpenMP runtimes bundled with Homebrew LightGBM and PyTorch into the same macOS process.

## Commands

All commands accept `--config configs/demo.yaml`.

| Command | Behavior |
| --- | --- |
| `prepare` | Download and pin assets; select 256 train, 32 dev, 32 test examples per domain |
| `synthetic` | Recover known effects and verify policy gains on a synthetic response surface |
| `benchmark` | Time both models and select an immutable compute profile |
| `collect` | Score candidate pools and collect fresh-adapter proxy runs |
| `fit` | Fit grouped DML, diagnose held-out residual prediction, freeze five policies |
| `compare` | Train/evaluate all policies on each model and save final adapters |
| `report` | Render complete or partial local results as HTML |
| `demo` | Run all stages, reusing completed compatible artifacts |

The main configuration exposes models, data revisions, sample counts, seeds, pool construction, LoRA settings, estimator settings, and budget profiles. Exact Hub revisions are recorded in the prepared manifest; rerunning `prepare` reuses that manifest rather than following updated `main` branches.

## Experiment design

- Domains: GSM8K math, MBPP coding, No Robots general instructions. Dataset IDs, licenses, source splits, and sample IDs are recorded in `manifest.json`. No Robots is CC-BY-NC-4.0; this is a local research demonstration.
- Reserve all normalized official test prompts before selecting any train/dev rows. Deduplicate globally, including across domains. Hold out 32 development rows from each official training split and 32 test rows from each official test split.
- Candidate pools contain 64 examples/domain. Independently sampled difficulty settings tilt sampling toward high/low frozen-proxy loss. Nine pre-treatment covariates summarize mean loss, response character length, and character diversity by domain.
- Draw two independent mixtures per candidate pool using `0.05 + 0.85 * Dirichlet(1,1,1)`. Domain assignments follow those probabilities; examples are sampled with replacement. Realized counts and response-token counts are recorded.
- Every training run starts from identical initial LoRA weights and a new AdamW optimizer. Base model parameters remain frozen. Default LoRA rank 8, alpha 16, q/v projections, zero dropout; effective batch size 4, learning rate 2e-4.
- Use plain `Instruction: ... Response: ...` text. Only response tokens contribute to loss. Preserve response tokens before prompt tokens when truncating; very long answers are also truncated. The report discloses truncation counts.
- Outcome: equally weighted domain means of per-example response NLL improvement against the untrained model. This is an adaptation, not accuracy or pass@1. Coding examples are never executed.
- Estimate effects of `log(T + 1e-6)` with LightGBM nuisance models and EconML CausalForestDML. Group all folds by pool. Outer grouped validation refits the full effect estimator and both nuisance functions, avoiding evaluation on in-sample effects.
- Compare uniform, best observed proxy, RegMix-style regression, positive-effect analytical extraction, and top-100 averaging over 10,000 search candidates. Positive-effect normalization is treated as a heuristic when coefficients are negative. The analytical policy uses the same domain floor as the sampled treatments.
- Freeze policies before opening test data. Transfer only mixture weights to Qwen; do not re-estimate causal effects on Qwen or select policies using test scores.

## Runtime and resume behavior

The 120-minute target excludes environment installation and asset downloads. Timings include synthetic checks, model benchmarking, collection, fitting, and comparison. Benchmark both lengths and models in FP32 before collecting real outcomes. Choose the first profile projected to fit with 20% headroom:

1. 64 proxy runs, 16 optimizer steps, 256 tokens.
2. 64 proxy runs, 8 steps, 128 tokens.
3. 32 proxy runs, 8 steps, 128 tokens.

If none fits, exit with code 2 and write a partial report. Runtime admission checks preserve a reserve for the final comparisons. The budget is based on measured estimates between atomic runs; a slow individual operation may overshoot its estimate. No automatic CPU fallback changes the recipe.

Completed runs are reused. An interrupted `running` record is retried from fresh initialization; partial optimizer state is not resumed. Writes are atomic and configuration/signature mismatches fail rather than mixing experiments. The active-time ledger persists across commands; time between commands is excluded. An exclusive lock rejects concurrent commands against the same artifact directory.

```sh
uv run --frozen causal-mix demo  # resume an interrupted compatible run
uv run --frozen causal-mix report  # inspect partial results without training
```

A budget-limited experiment remains inspectable. To change the budget, profile, seed, or other configuration, create a new configuration with a different artifact directory. Changing an existing experiment in place is intentionally rejected.

## Artifacts

`artifacts/demo/` contains:

- `config.json`, `environment.json`, `manifest.json`: settings, package versions, exact asset revisions and data hashes.
- `synthetic.json`, `benchmark.json`, `profile.json`: validation and immutable timing-based profile selection.
- `pools.json`, `training_losses.json`, `runs/`: membership, covariates, assigned/realized mixtures, outcomes and timing.
- `policies.json`: frozen policies, target effects, conditioning and grouped out-of-fold diagnostics.
- `comparisons/`, `adapters/`: per-example test losses, aggregate results, and final adapters for both models.
- `status.json`, `clock.json`, `report.html`: completion state, active compute time and readable results.

Model weights and datasets are cached locally but ignored by Git. Intermediate proxy adapters are not retained. Final adapters reference the corresponding base model under `models/`.

To publish a portable snapshot of a completed run, use `python scripts/package_results.py --source artifacts/demo --output results/<new-snapshot-name>`. The exporter copies numerical records and final adapters, replaces machine-local adapter paths with pinned Hugging Face references, and writes a SHA-256 inventory. The checked-in snapshot is read-only; new experiments continue to write under `artifacts/`.

## Interpretation

This demonstrates the method at toy scale. It differs from the paper in model family, LoRA training, dataset size and domains, short sequences, covariates, estimator capacity, candidate count, and outcome metric. One seed and few pools are not a significance study. A gain, tie, or regression should all be reported. Public data may have been seen during base-model pretraining; deduplication prevents only leakage introduced by this experiment.

The synthetic test validates implementation against a known response surface. It does not establish that the assumptions or learned effects hold for these language models.

## Signal analysis: covariates and incremental gain

The relevant question is **which mixture change improves the evaluation objective for a given data pool and model**. A sample-quality score alone does not answer that question. This section connects the [OpenDataArena-scored-data-2603 score definitions](https://huggingface.co/datasets/OpenDataArena/OpenDataArena-scored-data-2603#-scoring-dimensions) to the signals actually used here, then decomposes the observed gains. No OpenDataArena scores were downloaded or fitted in the completed toy run.

### OpenDataArena signals versus the implemented covariates

The [CausalMix paper, Section 4.3](https://arxiv.org/pdf/2607.01104) selects `HES`, `Normalized_Loss`, and `Writing_Style` to represent complementary properties. Its reported mixture-ranking Spearman correlation rises from 0.7307 with HES alone to 0.7436 with Normalized_Loss added, then 0.7557 with Writing_Style added. These are the paper's prediction results, not locally measured downstream gains or causal effects of improving a score.

| Signal family | OpenDataArena definition | What this implementation actually measures |
| --- | --- | --- |
| Difficulty / predictability | `Normalized_Loss`: Qwen3-8B normalized cross-entropy, described in the dataset card as bits per token. | Frozen SmolLM2 response NLL in natural-log units, averaged over examples within each domain. Different model, tokenizer, units, and preprocessing; not the same score. |
| Reasoning complexity | `HES`: summed entropy of the highest-entropy 0.5% of reasoning-trace tokens, scored with Qwen3-8B. | Mean response character length. This is a size and truncation signal, **not a measurement of HES or reasoning quality**. |
| Text quality / style | `Writing_Style`: QuRater-1.3B assessment of clarity, coherence, and style. | Mean `unique_characters / response_characters`. This is a length-sensitive diversity proxy, **not a style or correctness assessment**. |

The dataset supplies `raw_scores` and `processed_scores`; the latter are winsorized, scaled to [0,1], and reversed for metrics where lower is better. Consequently, a high processed score must not automatically be read as high raw loss or difficulty. Any extension should preserve the score representation, direction, scorer, and dataset revision. See the [dataset's format documentation](https://huggingface.co/datasets/OpenDataArena/OpenDataArena-scored-data-2603#-data-format--structure).

The local feature vector has **nine covariates**: the three implemented measurements above, separately aggregated for math, coding, and instructions. The full target training pool has:

| Domain | Frozen-proxy response NLL | Mean response characters | Mean unique-character ratio |
| --- | ---: | ---: | ---: |
| Math | 1.6998 | 290.6 | 0.1606 |
| Coding | 1.5115 | 183.2 | 0.2203 |
| Instructions | 2.6101 | 816.2 | 0.1203 |

Loss is computed on the tokenized, potentially truncated examples; character statistics use the original full responses. The instruction pool is harder for this proxy to predict and contains longer responses, but those observations do not establish lower quality. Character diversity is also entangled with length: across the 32 candidate pools, their Pearson correlations are -0.970 for math and -0.979 for coding. These redundant signals cannot support a confident ranking of individual covariate importance.

### How signals enter the gain estimate

Pool covariates `X` are computed **before** assigning the mixture `T` and remain unchanged for both treatments of that pool. Mixture assignment uses an independent random stream. DML residualizes the outcome and `Z = log(T + 1e-6)` against `X`, with all folds grouped by pool. The fitted response to a feasible mixture change is:

```text
predicted_gain(T versus uniform | X)
    = sum_k theta_k(X) * [log(T_k + 1e-6) - log(1/3 + 1e-6)]
```

Here, `theta` is a domain-treatment coefficient conditional on pool state; it is **not a feature-importance score or the causal effect of changing a covariate**. Because proportions sum to one, a domain cannot be increased while holding every other domain fixed. Interpret the complete mixture contrast, rather than isolated coefficients as independent interventions.

At the target pool, the fitted coefficients and the contributions to the predicted development-outcome contrast are:

| Domain | Estimated `theta` | CausalMix-A contribution | CausalMix-S contribution |
| --- | ---: | ---: | ---: |
| Math | +0.008532 | +0.007015 | +0.006921 |
| Coding | +0.001704 | -0.000944 | -0.000924 |
| Instructions | -0.005367 | +0.010181 | +0.009572 |
| **Predicted total gain over uniform** | — | **+0.016252** | **+0.015569** |

The model favors more math and fewer instruction examples; reducing coding incurs a small predicted penalty. The positive instruction contribution comes from reducing a domain with a negative fitted coefficient. It does not mean instruction examples are intrinsically harmful. These terms are a decomposition of the fitted model, not separately measured causal effects or held-out gains.

### What actually drove the observed gain

The table below reports **uniform test NLL minus CausalMix test NLL**, so positive values favor CausalMix. A and S have the same realized training examples and results in this run.

| Evaluation domain | SmolLM2 incremental reduction | Qwen incremental reduction |
| --- | ---: | ---: |
| Math | +0.05404 | +0.02808 |
| Coding | -0.00965 | -0.08608 |
| Instructions | -0.00285 | -0.02410 |
| **Equal-domain average** | **+0.01384** | **-0.02737** |

**The measured proxy benefit is a math gain that outweighs losses elsewhere.** The CausalMix policies sampled 49 math, 13 coding, and 2 instruction examples, versus 20, 21, and 23 for uniform. On Qwen, the math gain was smaller while the coding penalty was much larger, reversing the aggregate benefit. This breakdown identifies where the score changed; it does not isolate which training-domain intervention caused each evaluation-domain change.

**Token exposure is an additional mechanism to control.** Although optimizer steps and example counts matched, supervised response-token totals did not: SmolLM2 saw 6,566 with CausalMix versus 7,625 with uniform; Qwen saw 6,330 versus 7,335. Mixture changes therefore altered both content and token exposure. Length, truncation, and token weighting are plausible contributors to the result, not established explanations for it. The response-loss covariate was also computed with SmolLM2, so it does not directly describe Qwen's learning difficulty.

### What would establish incremental value from the covariates?

The current evidence supports a useful mixture contrast on the proxy, but **does not show that HES, style, or any individual local covariate caused the gain**. The heterogeneous forest's held-out residual loss was slightly worse than the constant-effect baseline, and no local covariate ablation was performed.

To test the additional signal from OpenDataArena, a follow-up should:

1. Use exact matching scored examples, or rescore the local examples with a documented pipeline. Do not attach scores from similarly named datasets or align rows by position. Compute pool-level features before mixture assignment and record missing-score coverage.
2. Compare constant effects, the current nine features, each paper-selected signal, and their combinations using the same grouped outer folds and proxy-run budget. Choose features on development diagnostics; reserve fresh pools and test evaluations for the final comparison.
3. Repeat the selected-policy training across seeds, match supervised-token budgets, and report per-domain gains alongside the equal-domain average. Test whether the observed improvements survive the token-budget control.
4. Evaluate transfer separately, including a documented model-specific difficulty signal. Treat quality/complexity interactions as hypotheses until they predict improvements on new pools and model runs.

Local numbers above are derived from `pools.json`, `policies.json`, and the per-domain losses and token counts in `comparisons/`. The OpenDataArena definitions provide a framework for interpreting and extending the experiment, not retrospective measurements of signals absent from this run.

## Sample results: efficacy of CausalMix-A and CausalMix-S

In the local run on **2026-11-04**, both CausalMix variants improved held-out response loss over uniform sampling on SmolLM2-135M. Transferring those mixtures to Qwen2.5-0.5B did not preserve that advantage. This demonstrates useful mixture selection on the proxy in this sample, while exposing a limitation of transfer across models.

The experiment completed **64 proxy runs and ten final comparisons** on an M4 Pro in about **10.3 active experiment minutes**. Each final comparison used the same 16-step, 256-token LoRA training budget within its model, with seed 42. Policies were selected using development data and then evaluated on 96 held-out examples: 32 each for math, coding, and general instructions.

**CausalMix-A** converts positive estimated marginal effects into mixture weights, with a 5% minimum allocation per domain. **CausalMix-S** scores 10,000 candidate mixtures and averages the top 100. Both favored math over general instructions:

| Policy | Math | Coding | General instructions |
| --- | ---: | ---: | ---: |
| Uniform | 33.33% | 33.33% | 33.33% |
| CausalMix-A | 75.85% | 19.15% | 5.00% |
| CausalMix-S | 75.02% | 19.38% | 5.60% |

The following values are **held-out response negative log-likelihood (NLL) reductions relative to each model's untrained baseline**; larger is better. The outcome equally averages the three domain scores. These are loss reductions, not accuracy percentages. Compare policies within a model, since the tokenizers differ.

| Policy | SmolLM2-135M | Qwen2.5-0.5B |
| --- | ---: | ---: |
| Uniform | 0.02698 | 0.12814 |
| Best proxy | 0.04203 | 0.15816 |
| RegMix-style | 0.04224 | 0.13306 |
| CausalMix-A | 0.04082 | 0.10078 |
| CausalMix-S | 0.04082 | 0.10078 |

**Proxy efficacy.** Both CausalMix-A and CausalMix-S achieved a loss reduction of **0.04082**, compared with **0.02698** for uniform sampling. Their additional reduction was **0.01384 NLL**, taking the final held-out loss from 2.03445 to 2.02061: **0.68% lower loss than uniform** under the same training budget. Both approaches therefore helped in this proxy experiment, although RegMix-style and best-proxy selection scored slightly better.

**Transfer efficacy.** On Qwen, both variants improved over the untrained model by **0.10078 NLL**, but uniform sampling improved by **0.12814**. Their final loss was 1.60931 versus 1.58195 for uniform: **1.73% higher loss**. Thus, fine-tuning with the selected mixtures helped relative to the base model, but CausalMix mixture selection did not outperform uniform sampling on the transfer model. Best-proxy selection performed best here.

**A versus S.** Their nearby weights produced exactly the same ordered training examples under this seed and short budget, explaining their identical scores. This run cannot establish that analytical extraction and search are equally effective in general, or that either variant is superior.

**Evidence and limits.** Fourteen tests passed, and synthetic effect recovery achieved RMSE 0.00332. On the real proxy data, grouped residual R-score versus zero effects was 0.724; however, forest R-loss (1.498e-5) was slightly worse than the constant-effect baseline (1.419e-5). These checks support a functioning implementation but do not establish an advantage from heterogeneous effects. One seed, short training, and a small test set do not establish statistical significance or replicate the paper's transfer claim.

The recorded measurements are in `artifacts/demo/comparisons/`, mixture weights and diagnostics in `artifacts/demo/policies.json`, and the visual report in `artifacts/demo/report.html`. These local artifacts also preserve per-domain losses, truncation rates, model and dataset revisions, and final adapters.
