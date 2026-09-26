# MiniMax-H3 P/P: real backward, one update, and next-rollout consistency

Completed on 2026-09-26. The GPU run and the offline comparator both exited with code 0. No repository source files were changed. All eight GPUs were released after completion.

## Outcome

All six diagnostic stages were exercised. All compared same-weight forward tensors and CPS scores were bitwise identical: 478 metric groups, zero unequal elements, zero byte-mismatched tensor pairs, zero nonfinite elements, and zero maximum absolute difference. There were no metadata comparison issues.

This does **not** establish gradient equivalence to another implementation or long-run convergence.

| Stage | MiniMax-H3 result | Existing Qwen3-8B / VIME evidence (not rerun here) |
| --- | --- | --- |
| 1. Generate and preserve original rollout scores | Completed for 32 trajectories at each of two weight versions | Completed; generation alone is not an equality test |
| 2. No-grad replay at generating weights | Bitwise identical to original rollout in both rounds | P/P mismatch; R/R strict had zero mismatches in the three-round report |
| 3. Training-mode, grad-enabled replay | Bitwise identical to original rollout in both rounds | Not independently isolated by that report |
| 4. Real loss/backward, without an optimizer update yet | Completed on all 32 first-round trajectories and steps [0,3,6]; gradients finite and nonzero; weights unchanged; subsequent replay bitwise identical and did not mutate accumulated gradients | Training occurred, but backward/gradient equality was not independently validated |
| 5. Optimizer step | Exactly one committed update; LoRA parameters changed, frozen parameters unchanged | Updates executed; not a standalone optimizer-equivalence test |
| 6. New rollout with updated weights, then same-weight replay | Completed for 32 new trajectories; no-grad and training-mode replay both bitwise identical; no second update | Three-round aggregate: P/P mismatches, R/R zero mismatches |

The historical VIME report is `/data/ellm/vime_qwen3_8b_tp4_cp2_200_experiment/results/pr424-eval-3r/REPORT.md`: P/P had 73,405 mismatches out of 123,083 compared logprobs, max absolute difference 0.491172791; R/R had 0 out of 124,226. The two arms used distinct generated workloads. Those text-token counts are not interchangeable with H3 diffusion-score counts.

## Fixed configuration and test scope

- UniRL commit: `9044f5b9f61dc4e7d8226316a9fe519ec5c3d134`.
- Eight H100 80 GB GPUs; native TrainsideRolloutEngine, PyTorch FSDP2, LoRA rank 32 / alpha 64 / dropout 0.
- 768x768, 124 frames, 10 denoising evaluations, CPS eta 0.7, selected SDE steps [0,3,6].
- 8 prompts x 4 samples = 32 trajectories per round, 64 total; 192 sampled transition positions across both rounds.
- Sampling seed 43; data-source seed 42. Model checkpoint: `/data/ellm/zhangjian/MiniMax-H3`.
- Native VideoPickScore/CLAP rewards and computed advantages; native FlowGRPO clipped loss, clip range 0.005, replay anchor, beta 0.
- Native optimizer/scheduler path, learning rate 0.0003, Adam epsilon 1e-12; one update after round 0, none after round 1.
- Mixed precision and original trajectory storage retained. Activation checkpointing enabled. `NCCL_NVLS_ENABLE=0` retained from the working baseline.

Three SDE steps were backwarded sequentially with each native loss scaled by `loss_scale / 3`, preserving the mean objective while avoiding three simultaneously retained graphs. A CPU float64 toy test verified the loss/gradient scaling against the all-steps native mean. This is not a GPU numerical-equivalence test against the simultaneous-graph implementation.

## Numerical and weight evidence

- 416 completed invocation manifests: 256 at weight version 0 and 160 at version 1.
- Compared actual trajectories, conditioning, geometry/layout, schedules, video/audio predictions, transition means, destinations, modality scores, and joint CPS scores.
- Original rollout scores were preserved before the replay-anchor replacement.
- Each comparison stays within its generating weight version; old-weight vs new-weight differences are not called inference inconsistency.
- Across the two rounds, each generation-vs-anchor prediction comparison covers 392,822,784 video elements and 2,543,616 audio elements in total over the three selected steps.
- Global gradient norm reported identically by all ranks: `1.2560325558297336e-05`.
- Sum of changed local LoRA-shard elements: `97,484,788`; largest absolute parameter change: `0.00029999788966961205`.
- Full transformer named-parameter hashes remained unchanged across each backward/replay micro. Frozen parameter hashes matched before/after the optimizer update. Auxiliary named-parameter hashes matched from the first anchor boundary through final replay.
- Forward captures verified local trainable parameter identity before/after each invocation; all invocations within a rank and weight version shared the same trainable fingerprint, and the fingerprint changed between versions.
- Allocator retries: 72 on ranks 0-6, 79 on rank 7. Final `num_ooms` was 0 on all ranks; allocation warnings were recovered, not terminal failures.

## Artifacts and reproduction

Related protocol: [MiniMax-H3 P/P reproducibility checklist](minimax-h3-pp-reproducibility-checklist.md).

All artifact paths below are relative to `/data/ellm/zhangjian/experiments/minimax-h3-pp/smoke-GVVKCy/` on the experiment host. Raw tensors and diagnostic scripts remain there and are not included in this documentation PR.

- `update-comparison.json`: complete comparison metrics and per-rank update evidence.
- `capture/rank-*/`: raw tensors, manifests, per-sample backward summaries, optimizer evidence, completion markers.
- `update_pp.py`: exact one-off diagnostic harness snapshot.
- `compare_update_pp.py`: exact offline comparator snapshot.
- `launcher.sh`, `hydra/.hydra/`, `driver.log`: launcher, resolved configuration/overrides, runtime log.

Launch command used:

```bash
bash /data/ellm/zhangjian/run_minimax_pp_smoke.sh \
  'sampling.sde_indices=[0,3,6]' sampling.seed=43 num_rollouts=2 \
  algorithm._target_=update_pp.UpdateFlowGRPO \
  backend._target_=update_pp.UpdateBackend \
  'logging.tags=[minimax-h3,pp,real-backward,one-update]'
```

The run occupies approximately 29 GB under `/data`. `save_interval=0`; this run retained diagnostic evidence, not a resumable updated-model checkpoint.

## Limits

Only one optimizer update was tested. Gradient finiteness/nonzero values and post-backward replay were verified, but independent reference gradients, repeated-backward gradient equality, simultaneous three-step graphs, and long-run training were not tested. Model buffers and plain tensor attributes were not fingerprinted. CPS transition scores are negative mean squared error, not normalized Gaussian log densities or text-token logprobs. These results apply to this native shared-model P/P configuration, not every inference/training engine pair.
