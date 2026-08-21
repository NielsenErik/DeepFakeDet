# PC deepfake detection — results (srm_g8c16_kd-orc_K8)

Protocol: the circuit and every one-class baseline are fitted on FF++ c23 REAL training faces only (official identity-disjoint 720/140/140 split); no forgery is seen during fitting. Cross-dataset test sets are scored by the same frozen models.

## Detection (video-level AUC)

| model       | ffpp                    |
|-------------|-------------------------|
| PC          | 0.6242 (patch_cond_max) |
| gmm         | 0.5314 (patch_max)      |
| mahalanobis | 0.6283 (patch_max)      |
| patchcore   | 0.6268 (patch_max)      |

### ffpp — per manipulation method

| model (score)           | Deepfakes | Face2Face | FaceShifter | FaceSwap | NeuralTextures |
|-------------------------|-----------|-----------|-------------|----------|----------------|
| PC (patch_cond_max)     | 0.4505    | 0.6544    | 0.3361      | 0.6925   | 0.4069         |
| mahalanobis (patch_max) | 0.4502    | 0.6558    | 0.3501      | 0.6977   | 0.4139         |
| gmm (patch_max)         | 0.4384    | 0.6489    | 0.3181      | 0.6744   | 0.4064         |
| patchcore (patch_max)   | 0.4503    | 0.6575    | 0.3477      | 0.7006   | 0.4178         |

## Localization (exact per-patch queries)

Masks: `derived_frame_diff` over 1293 manipulated frames.

| model          | patch AUC (pooled) | patch AUC (per image) | best IoU | pointing acc |
|----------------|--------------------|-----------------------|----------|--------------|
| PC_conditional | 0.5193             | 0.5067                | 0.1694   | 0.1856       |
| PC_marginal    | 0.5194             | 0.5069                | 0.1699   | 0.1848       |
| mahalanobis    | 0.5630             | 0.5596                | 0.1852   | 0.1651       |
| gmm            | 0.5250             | 0.5071                | 0.1818   | 0.2180       |
| patchcore      | 0.5278             | 0.5192                | 0.1718   | 0.1912       |

### Conditional surprisal by manipulation

| method         | patch AUC | manipulated fraction |
|----------------|-----------|----------------------|
| Deepfakes      | 0.3826    | 0.2178               |
| Face2Face      | 0.5946    | 0.0900               |
| FaceShifter    | 0.5157    | 0.1544               |
| FaceSwap       | 0.6346    | 0.1986               |
| NeuralTextures | 0.5264    | 0.0600               |

## Circuit property audit

```
{
  "log_partition": -1.0728907682278077e-06,
  "max_abs_diff_marginal_none_vs_logprob": 0.0,
  "max_abs_marginal_all": 1.7881492340166005e-06,
  "finite_partial_marginal": 1.0,
  "structured_decomposable": 1.0,
  "structure_n_regions": 2047,
  "structure_n_leaf_regions": 1024,
  "structure_max_arity": 2,
  "structure_structured_decomposable": 1,
  "structure_mean_region_scope": 6.534440644846116,
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

## Pre-registered decision rubric

| gate                          | metric                                    | threshold | observed | pass |
|-------------------------------|-------------------------------------------|-----------|----------|------|
| G1_detection                  | ffpp in-dataset video AUC                 | 0.9000    | 0.6242   | FAIL |
| G2_generalization             | cross-dataset AUC minus SBI               | -0.0300   | —        | —    |
| G3_circuit_value_detection    | AUC minus best baseline                   | 0.0200    | -0.0041  | FAIL |
| G3_circuit_value_localization | patch AUC minus best baseline             | 0.0300    | -0.0436  | FAIL |
| G4_structure                  | val NLL gain over random structure (nats) | 2.0000    | —        | —    |
| G5_scale                      | circuit fit seconds                       | 3600.0000 | —        | —    |

**Verdict: STOP — the detector does not work in-dataset; fix the representation first**
