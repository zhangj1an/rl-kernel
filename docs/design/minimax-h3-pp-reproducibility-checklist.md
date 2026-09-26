# MiniMax-H3 P/P reproducibility checklist

Status: experiment protocol, not an executed validation. Instrumentation and runtime setup are still pending. Unchecked items require evidence from the actual run.

## Scope

Measure native rollout versus pre-update training replay for UniRL's MiniMax-H3 Flow-GRPO recipe. P/P means native implementations on both sides, without RL-Kernel operator replacements.

- Rollout: `TrainsideRolloutEngine`, using the training actor's model and pipeline.
- Training replay: PyTorch FSDP2 with LoRA, through the actual training replay path.
- Compared quantity: video/audio diffusion-transition log probabilities, not vocabulary-token log probabilities.
- Follow the [module debug matrix](ws2-module-debug-matrix.md): fixed replay, comparability gates, then one-factor-at-a-time diagnosis. The diffusion-specific checks below are an adaptation of that methodology, not an existing executable validator.

This is not a Megatron/vLLM comparison or a DiffusionNFT experiment. Sharing a model does not establish numerical equality between generation and replay.

## Baseline and storage

Keep this document in `rl-kernel/docs/design/` on `test_unirl_minimax_h3`. All generated artifacts, caches, logs, trajectories, and checkpoints belong on the `/data` volume.

| Item | Baseline |
| --- | --- |
| RL-Kernel source commit | `01b4ae410ae27aa0ff93bb48300c05158ef2968e`; record subsequent changes separately |
| UniRL source commit | `486284707e4e48372689e099b9a3e4788a9caba5` |
| UniRL branch | `test_rl_kernel_minimax_h3` |
| Recipe | `examples/diffusion/minimax_h3/minimax_h3_t2va_trainside.yaml` |
| Model | `MiniMaxAI/MiniMax-H3` |
| Model revision | `42ed227ee7df40d41602854ae760620d6eb651fe` |
| Local checkpoint | `/data/ellm/zhangjian/MiniMax-H3` |
| Run directory | `/data/ellm/zhangjian/experiments/minimax-h3-pp/<run-id>/` |

The downloaded recipe components contain 63 files totaling 144,051,199,021 bytes. File sizes and references in three shard indexes were verified. This does not establish content-hash integrity or successful model loading.

## 1. Freeze configuration and provenance

These are current recipe settings. Save the fully resolved configuration and every override at launch; a YAML filename alone is insufficient.

| Factor | Baseline or requirement |
| --- | --- |
| Topology | One node, eight H100 80GB GPUs, full FSDP sharding; record actual mesh and rank/device/sample ownership |
| Batching | `batch_size=8`, rollout `forward_batch_size=1`, training `micro_batch_size=1` |
| LoRA | Rank 32, alpha 64, dropout 0; `module_prefix=transformer_blocks`; preserve the complete target-module list |
| Model precision | BF16 block computation, checkpoint-specific FP32 modules, `master_dtype=fp32` |
| FSDP | `root_wrap=false`, `cast_forward_inputs=false`, `reshard_after_forward=true`, activation checkpointing on, compile off |
| Stored trajectory / logp | FP16 / FP32; also record actual intermediate dtypes |
| Geometry | 768×768, 124 frames; do not reduce geometry below the checkpoint's supported constraints |
| Sampling | CPS, 10 model evaluations, `eta=0.7`, `sde_indices=[0,3,6]` |
| Schedules | Video shift 12, audio shift 3; save actual sigma tensors, not just schedule parameters |
| Joint probability | `audio_joint_sde=true`; record modality-specific logp, element counts, and element-weighted joint logp |
| Inputs and randomness | Fixed prompts, order, sample IDs, embeddings, initial video/audio noise, and RNG seeds/states |

- [ ] Record both repository commits, dirty status, and patches used by the run.
- [ ] Record model component configurations and weight identities/checksums, including initialized LoRA state.
- [ ] Record Python, PyTorch, CUDA runtime/driver, NCCL, Transformers, Diffusers, PEFT, and actual attention implementation versions.
- [ ] Record TF32, autocast, deterministic flags, matmul precision, attention dispatch, and collective settings; do not assume defaults match.
- [ ] Save actual conditioning tensors, masks, positions, and packing metadata with shapes, dtypes, and digests.
- [ ] Record the actual distributed layout. All required ranks must participate in FSDP collectives during replay.
- [ ] Route HF/compilation caches, Ray temporary files, `TMPDIR`, logs, and tracker outputs to `/data`; verify that the paths take effect. Disable unnecessary online reporting.

## 2. Freeze weights while preserving real execution paths

- [ ] Use the same initialized model, LoRA weights, and adapter selection for rollout and every baseline replay.
- [ ] Prevent optimizer, scheduler, EMA, and adapter updates until comparison finishes. Check parameter identity before and after.
- [ ] Do not launch the recipe's default training loop unchanged: it requests two updates per batch. Intercept before updates; do not assume setting the count to zero is supported.
- [ ] Preserve and record native rollout behavior: `TrainsideRolloutEngine` calls `eval()`, generates under `torch.no_grad()`, and restores the previous training flags afterward.
- [ ] Use and record the actual training replay mode, grad-enabled state, autocast, and checkpointing configuration. Do not force both paths into eval/no-grad just to obtain equality.
- [ ] Treat train/eval, autograd, and checkpointing differences as explicit execution factors to investigate, not undocumented changes.

## 3. Preserve original rollout evidence

**The recipe sets `old_logp_source: replay`. `FlowGRPO.prepare_segment()` replaces `segment.sde_logp` with replay results. The ordinary PPO ratio can therefore compare replay against replay and hide rollout/replay differences.**

- [ ] Before `prepare_segment()`, independently clone and save `segment.sde_logp` as `rollout_logp_original`. A reference to a mutable segment is insufficient.
- [ ] Save `latents`, `aux_latents`, `indices`, `sde_indices`, and `sigmas`, plus conditions and sampling parameters.
- [ ] Align observations by sample ID, diffusion step, and modality; record joint-logp weights and reduction dimensions.
- [ ] Capture the transition destination used to calculate generation logp and the destination stored after the trajectory cast, or establish that they are exactly equal.
- [ ] Keep training-anchor logp separate from original rollout logp. Changing `old_logp_source` is not a substitute for independent capture.
- [ ] Generate one rollout batch initially, then replay that stored trajectory. Regenerating with the same seed is not fixed-input replay.

## 4. Comparability gates and claim boundaries

Maintain two distinct comparisons:

1. **Native end-to-end difference:** original rollout logp versus actual training replay, retaining the recipe's trajectory precision. This can include storage/casting effects.
2. **Same-input computation drift:** differences attributed to execution under identical inputs, only after all gates below pass.

- [ ] Identical checkpoint, LoRA parameters, and active adapter; no parameter update.
- [ ] Identical sample/step identities, conditioning, masks, positions, packing metadata, and logical ownership.
- [ ] Identical video/audio current latents, scored next latents, sigma/sigma-next, eta, and probability definition.
- [ ] Matching input shapes, dtypes, and values; record layout/stride differences separately.
- [ ] Complete, finite data. Missing audio trajectories, missing steps, or mismatched dimensions must not be silently truncated or skipped.

A failed gate yields `comparable=false` with a reason and no claimed kernel-drift metric. End-to-end measurements may still be retained under their separate label, without attributing them to same-input kernel arithmetic.

Generation obtains logp from `strategy.denoise()` before casting `x_next` and `a_next` to the trajectory dtype. Replay scores the stored destinations. This is a candidate rounding boundary, not a demonstrated cause of error. Generation also casts initial/current latents to the trajectory dtype, so it would be incorrect to describe the entire rollout as higher precision.

## 5. Measurements and verdicts

- [ ] Report shapes, element counts, nonfinite counts, mean/max absolute difference, and unequal-value fraction by sample, step, and video/audio/joint output.
- [ ] Define `delta = replay_logp - rollout_logp_original`; report delta and `exp(delta)`, including overflow/nonfinite flags.
- [ ] Label the ratio as derived from the model's reduced logp, not an unreduced full-trajectory joint probability.
- [ ] Identify the sample/step with maximum error; retain intermediate evidence sufficient for diagnosis.
- [ ] Distinguish numerical equality (`torch.equal` or elementwise comparison) from bitwise equality. A bitwise claim requires identical dtypes and comparison of raw representations.
- [ ] Separate execution failure, incomplete evidence, non-comparability, comparable zero drift, and comparable nonzero drift.
- [ ] Nonzero P/P drift is an experimental result, not automatically a failed baseline. Report observed values first; define any later acceptance tolerance before judging a run.

## 6. Execution and diagnosis sequence

1. Verify dependencies, model loading, distributed initialization, input data, and memory headroom. Do not start the default 10,000-rollout training run.
2. Generate one native rollout and preserve immutable original evidence immediately.
3. Replay before updates through the training path; evaluate gates and report the measurements above.
4. Repeat replay on the same stored trajectory to measure replay's own repeatability, verifying that weights and configuration remain unchanged.
5. If differences occur, compare video/audio denoiser predictions, then transition mean/std, scored destinations, and logp reductions. This distinguishes model-forward differences from probability-calculation differences.
6. Change only one factor per diagnostic variant: destination casting first, then execution mode, autograd/autocast, attention backend, batching/FSDP, or reduction ordering as evidence warrants. Save each variant independently.

Raising `trajectory_precision` changes both storage and subsequent model inputs. A newly generated FP32 trajectory is an end-to-end precision variant, not a pure storage ablation on the original trajectory. To isolate rounding, retain pre/post-cast destinations at a fixed step and rescore them with the same denoiser prediction.

A first memory smoke test may replay one selected SDE step, explicitly labeled as partial coverage. It cannot establish coverage of all `[0,3,6]` steps. Preserve the recipe's topology, valid geometry, and batch-1 requirements; actual H100 memory requirements remain to be verified.

## 7. Required artifacts

The proposed layout below is a diagnostic convention, not an implemented RL-Kernel sealed-attempt schema:

```text
/data/ellm/zhangjian/experiments/minimax-h3-pp/<run-id>/
  manifest.json          # Revisions, weights, command, seeds, ranks, run status
  resolved-config.yaml  # Fully resolved configuration and overrides
  environment.json      # Hardware, software, actual backends and numeric settings
  source.patch          # Uncommitted changes used by this run
  logs/                 # Per-rank logs
  rollout/              # Immutable trajectory and original rollout logp
  replay/               # Pre-update replay and diagnostic intermediates
  comparison.json       # Gates, metrics, missing evidence, failure reasons
  summary.md            # Findings, limitations, next single-factor experiment
```

- [ ] Mark completion only after every required rank finishes and evidence is complete. A success string in a log is not numerical validation.
- [ ] Exclude credentials and unfiltered environment dumps.
- [ ] State whether only forward/logp was tested. Do not infer backward, gradient, or multi-update consistency from this experiment.

## Source references

RL-Kernel:

- `rl_engine/integrations/ablation.py`
- `rl_engine/alignment/cross_config/debug_matrix.py`
- `examples/vime_qwen3_8b_tp2_cp2/validate_artifacts.py`

The VIME validator's input schema and strict RL-Kernel execution requirements do not directly apply to this P/P diffusion experiment.

UniRL:

- `examples/diffusion/minimax_h3/minimax_h3_t2va_trainside.yaml`
- `unirl/rollout/engine/trainside/engine.py`
- `unirl/models/minimax_h3/diffusion.py`
- `unirl/algorithms/flowgrpo.py`

Recheck capture boundaries after source changes, particularly replacement of original rollout logp by the replay anchor.
