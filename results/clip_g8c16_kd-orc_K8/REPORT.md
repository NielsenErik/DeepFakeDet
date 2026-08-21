# PC deepfake detection — results (clip_g8c16_kd-orc_K8)

Protocol: the circuit and every one-class baseline are fitted on FF++ c23 REAL training faces only (official identity-disjoint 720/140/140 split); no forgery is seen during fitting. Cross-dataset test sets are scored by the same frozen models.

## Detection (video-level AUC)

| model       | ffpp                       |
|-------------|----------------------------|
| PC          | 0.5360 (patch_cond_lowmax) |
| flow        | 0.5359 (patch_max)         |
| gmm         | 0.5307 (patch_lowmax)      |
| mahalanobis | 0.5191 (patch_lowmax)      |
| patchcore   | 0.5174 (patch_lowmax)      |

### ffpp — per manipulation method

| model (score)              | Deepfakes | Face2Face | FaceShifter | FaceSwap | NeuralTextures |
|----------------------------|-----------|-----------|-------------|----------|----------------|
| PC (patch_cond_lowmax)     | 0.5386    | 0.5081    | 0.5862      | 0.5557   | 0.4914         |
| mahalanobis (patch_lowmax) | 0.4999    | 0.5099    | 0.5674      | 0.5306   | 0.4877         |
| gmm (patch_lowmax)         | 0.4982    | 0.5169    | 0.5992      | 0.5432   | 0.4961         |
| patchcore (patch_lowmax)   | 0.4591    | 0.5227    | 0.5717      | 0.5281   | 0.5052         |
| flow (patch_max)           | 0.6386    | 0.4945    | 0.4572      | 0.6105   | 0.4785         |

## Localization (exact per-patch queries)

Masks: `derived_frame_diff` over 1711 manipulated frames.

| model          | patch AUC (pooled) | patch AUC (per image) | best IoU | pointing acc |
|----------------|--------------------|-----------------------|----------|--------------|
| PC_conditional | 0.5502             | 0.5225                | 0.1789   | 0.2022       |
| PC_marginal    | 0.5432             | 0.5161                | 0.1753   | 0.1830       |
| mahalanobis    | 0.5500             | 0.5340                | 0.1779   | 0.1609       |
| gmm            | 0.5451             | 0.5260                | 0.1789   | 0.1615       |
| patchcore      | 0.5106             | 0.4914                | 0.1581   | 0.1848       |
| flow           | 0.5362             | 0.5044                | 0.1706   | 0.2117       |

### Conditional surprisal by manipulation

| method         | patch AUC | manipulated fraction |
|----------------|-----------|----------------------|
| Deepfakes      | 0.6057    | 0.2177               |
| Face2Face      | 0.5034    | 0.0908               |
| FaceShifter    | 0.5032    | 0.1550               |
| FaceSwap       | 0.5636    | 0.1991               |
| NeuralTextures | 0.4742    | 0.0605               |

## Circuit property audit

```
{
  "log_partition": -7.689008270972408e-06,
  "max_abs_diff_marginal_none_vs_logprob": 0.0,
  "max_abs_marginal_all": 7.5101943366462365e-06,
  "finite_partial_marginal": 1.0,
  "structured_decomposable": 1.0,
  "structure_n_regions": 2047,
  "structure_n_leaf_regions": 1024,
  "structure_max_arity": 2,
  "structure_structured_decomposable": 1,
  "structure_mean_region_scope": 6.784562774792379,
  "structure_max_region_scope": 1024,
  "structure_multi_partition_regions": 0,
  "size_features": 1024,
  "size_leaf_units": 8192,
  "size_sum_units": 8177,
  "size_product_units": 65472,
  "size_parameters": 621632,
  "size_levels": 15,
  "size_einsum_groups": 15,
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
| random/random | 1255.6965 | 0.5327    | 24.9602 |
| kd/random     | 1253.7290 | 0.5297    | 32.9499 |
| kd/chow_liu   | 1250.4683 | 0.5315    | 25.9162 |
| kd/orc        | 1209.7200 | 0.5245    | 29.0506 |
| kd/forman     | 1086.2600 | 0.5192    | 21.2601 |
| chow_liu/orc  | 1229.9288 | 0.5299    | 31.5614 |
| orc/orc       | 1235.7251 | 0.5290    | 37.6942 |

## Pre-registered decision rubric

| gate                          | metric                                    | threshold | observed | pass |
|-------------------------------|-------------------------------------------|-----------|----------|------|
| G1_detection                  | ffpp in-dataset video AUC                 | 0.9000    | 0.5360   | FAIL |
| G2_generalization             | cross-dataset AUC minus SBI               | -0.0300   | —        | —    |
| G3_circuit_value_detection    | AUC minus best baseline                   | 0.0200    | 0.0001   | FAIL |
| G3_circuit_value_localization | patch AUC minus best baseline             | 0.0300    | 0.0002   | FAIL |
| G4_structure                  | val NLL gain over random structure (nats) | 2.0000    | 169.4365 | PASS |
| G5_scale                      | circuit fit seconds                       | 3600.0000 | 37.6942  | PASS |

**Verdict: STOP — the detector does not work in-dataset; fix the representation first**
