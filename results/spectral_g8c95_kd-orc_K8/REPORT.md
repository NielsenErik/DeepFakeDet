# PC deepfake detection — results (spectral_g8c95_kd-orc_K8)

Protocol: the circuit and every one-class baseline are fitted on FF++ c23 REAL training faces only (official identity-disjoint 720/140/140 split); no forgery is seen during fitting. Cross-dataset test sets are scored by the same frozen models.

## Detection (video-level AUC)

| model       | ffpp                       |
|-------------|----------------------------|
| PC          | 0.5541 (patch_cond_lowmax) |
| SBI         | 0.8596 (prob)              |
| flow        | 0.5568 (patch_lowmax)      |
| gmm         | 0.5916 (patch_lowmax)      |
| mahalanobis | 0.5685 (patch_lowmax)      |
| patchcore   | 0.5556 (patch_lowmax)      |

### ffpp — per manipulation method

| model (score)              | Deepfakes | Face2Face | FaceShifter | FaceSwap | NeuralTextures |
|----------------------------|-----------|-----------|-------------|----------|----------------|
| PC (patch_cond_lowmax)     | 0.5852    | 0.5325    | 0.5949      | 0.4887   | 0.5689         |
| mahalanobis (patch_lowmax) | 0.6776    | 0.5000    | 0.6828      | 0.4009   | 0.5811         |
| gmm (patch_lowmax)         | 0.6887    | 0.5268    | 0.6516      | 0.4706   | 0.6202         |
| patchcore (patch_lowmax)   | 0.5943    | 0.4832    | 0.6662      | 0.4436   | 0.5910         |
| flow (patch_lowmax)        | 0.6281    | 0.5104    | 0.6131      | 0.4597   | 0.5729         |
| SBI (prob)                 | 0.9401    | 0.9259    | 0.8456      | 0.7625   | 0.8236         |

## Localization (exact per-patch queries)

Masks: `derived_frame_diff` over 1711 manipulated frames.

| model          | patch AUC (pooled) | patch AUC (per image) | best IoU | pointing acc |
|----------------|--------------------|-----------------------|----------|--------------|
| PC_conditional | 0.5174             | 0.5118                | 0.1759   | 0.2404       |
| PC_marginal    | 0.5171             | 0.5115                | 0.1758   | 0.2410       |
| mahalanobis    | 0.5097             | 0.4973                | 0.1788   | 0.2201       |
| gmm            | 0.5066             | 0.4975                | 0.1730   | 0.2069       |
| patchcore      | 0.5159             | 0.5115                | 0.1740   | 0.2362       |
| flow           | 0.5091             | 0.4995                | 0.1725   | 0.2201       |

### Conditional surprisal by manipulation

| method         | patch AUC | manipulated fraction |
|----------------|-----------|----------------------|
| Deepfakes      | 0.4140    | 0.2177               |
| Face2Face      | 0.5617    | 0.0908               |
| FaceShifter    | 0.5072    | 0.1550               |
| FaceSwap       | 0.6299    | 0.1991               |
| NeuralTextures | 0.4929    | 0.0605               |

## Circuit property audit

```
{
  "log_partition": -4.935271863359958e-05,
  "max_abs_diff_marginal_none_vs_logprob": 0.0,
  "max_abs_marginal_all": 5.108125333208591e-05,
  "finite_partial_marginal": 1.0,
  "structured_decomposable": 1.0,
  "structure_n_regions": 11903,
  "structure_n_leaf_regions": 6080,
  "structure_max_arity": 4,
  "structure_structured_decomposable": 1,
  "structure_mean_region_scope": 7.629673191632362,
  "structure_max_region_scope": 6080,
  "structure_multi_partition_regions": 0,
  "size_features": 6080,
  "size_leaf_units": 48640,
  "size_sum_units": 46577,
  "size_product_units": 361920,
  "size_parameters": 3478592,
  "size_levels": 21,
  "size_einsum_groups": 22,
  "size_structured_decomposable": 1
}
```

## Circuit throughput (tensorized vs reference object graph)

| d    | K | einsum s/step | reference s/step | speedup | params  | log Z   |
|------|---|---------------|------------------|---------|---------|---------|
| 128  | 4 | 0.0034        | 0.1519           | 44.5281 | 14224   | -0.0000 |
| 1024 | 8 | 0.0136        | —                | —       | 621632  | -0.0000 |
| 4096 | 8 | 0.0424        | —                | —       | 2489408 | -0.0000 |

## Structure ablation

| structure     | val NLL   | video AUC | fit s   |
|---------------|-----------|-----------|---------|
| random/random | 1099.0868 | 0.5203    | 36.9719 |
| kd/chow_liu   | 1048.2886 | 0.5163    | 63.0594 |
| kd/orc        | 1040.3938 | 0.5173    | 67.6936 |
| kd/forman     | 888.6292  | 0.5187    | 48.9575 |

## Pre-registered decision rubric

| gate                          | metric                                    | threshold | observed | pass |
|-------------------------------|-------------------------------------------|-----------|----------|------|
| G1_detection                  | ffpp in-dataset video AUC                 | 0.9000    | 0.5541   | FAIL |
| G2_generalization             | cross-dataset AUC minus SBI               | -0.0300   | —        | —    |
| G3_circuit_value_detection    | AUC minus best baseline                   | 0.0200    | -0.0375  | FAIL |
| G3_circuit_value_localization | patch AUC minus best baseline             | 0.0300    | 0.0016   | FAIL |
| G4_structure                  | val NLL gain over random structure (nats) | 2.0000    | 210.4576 | PASS |
| G5_scale                      | circuit fit seconds                       | 3600.0000 | 67.6936  | PASS |

**Verdict: STOP — the detector does not work in-dataset; fix the representation first**
