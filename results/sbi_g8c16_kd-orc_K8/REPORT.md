# PC deepfake detection — results (sbi_g8c16_kd-orc_K8)

Protocol: the circuit and every one-class baseline are fitted on FF++ c23 REAL training faces only (official identity-disjoint 720/140/140 split); no forgery is seen during fitting. Cross-dataset test sets are scored by the same frozen models.

## Detection (video-level AUC)

| model       | ffpp                       |
|-------------|----------------------------|
| PC          | 0.8125 (patch_cond_lowmax) |
| SBI         | 0.8596 (prob)              |
| flow        | 0.8038 (patch_lowmax)      |
| gmm         | 0.7879 (patch_lowmax)      |
| mahalanobis | 0.3863 (patch_lowmax)      |
| patchcore   | 0.4613 (patch_mean)        |

### ffpp — per manipulation method

| model (score)              | Deepfakes | Face2Face | FaceShifter | FaceSwap | NeuralTextures |
|----------------------------|-----------|-----------|-------------|----------|----------------|
| PC (patch_cond_lowmax)     | 0.8754    | 0.8560    | 0.8261      | 0.7069   | 0.7980         |
| mahalanobis (patch_lowmax) | 0.3075    | 0.3429    | 0.3775      | 0.4596   | 0.4442         |
| gmm (patch_lowmax)         | 0.8245    | 0.8322    | 0.8033      | 0.6959   | 0.7835         |
| patchcore (patch_mean)     | 0.5601    | 0.4730    | 0.4148      | 0.4443   | 0.4140         |
| flow (patch_lowmax)        | 0.8602    | 0.8490    | 0.8207      | 0.6971   | 0.7923         |
| SBI (prob)                 | 0.9401    | 0.9259    | 0.8456      | 0.7625   | 0.8236         |

## Localization (exact per-patch queries)

Masks: `derived_frame_diff` over 1711 manipulated frames.

| model          | patch AUC (pooled) | patch AUC (per image) | best IoU | pointing acc |
|----------------|--------------------|-----------------------|----------|--------------|
| PC_conditional | 0.5679             | 0.5629                | 0.2033   | 0.2183       |
| PC_marginal    | 0.5665             | 0.5610                | 0.2029   | 0.2207       |
| mahalanobis    | 0.5399             | 0.5140                | 0.1841   | 0.0831       |
| gmm            | 0.5400             | 0.5288                | 0.1840   | 0.2063       |
| patchcore      | 0.6701             | 0.6725                | 0.2881   | 0.4384       |
| flow           | 0.5104             | 0.5049                | 0.1656   | 0.1417       |

### Conditional surprisal by manipulation

| method         | patch AUC | manipulated fraction |
|----------------|-----------|----------------------|
| Deepfakes      | 0.6506    | 0.2177               |
| Face2Face      | 0.5317    | 0.0908               |
| FaceShifter    | 0.4842    | 0.1550               |
| FaceSwap       | 0.5562    | 0.1991               |
| NeuralTextures | 0.4836    | 0.0605               |

## Circuit property audit

```
{
  "log_partition": -7.808224836480804e-06,
  "max_abs_diff_marginal_none_vs_logprob": 0.0,
  "max_abs_marginal_all": 7.212180662463652e-06,
  "finite_partial_marginal": 1.0,
  "structured_decomposable": 1.0,
  "structure_n_regions": 2047,
  "structure_n_leaf_regions": 1024,
  "structure_max_arity": 2,
  "structure_structured_decomposable": 1,
  "structure_mean_region_scope": 6.47191011235955,
  "structure_max_region_scope": 1024,
  "structure_multi_partition_regions": 0,
  "size_features": 1024,
  "size_leaf_units": 8192,
  "size_sum_units": 8177,
  "size_product_units": 65472,
  "size_parameters": 621632,
  "size_levels": 14,
  "size_einsum_groups": 14,
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

| structure     | val NLL  | video AUC | fit s   |
|---------------|----------|-----------|---------|
| random/random | 729.6938 | 0.7274    | 23.8569 |
| kd/chow_liu   | 730.7589 | 0.7265    | 24.6926 |
| kd/orc        | 678.0132 | 0.7226    | 28.4073 |
| kd/forman     | 565.4110 | 0.7258    | 20.5698 |

## Pre-registered decision rubric

| gate                          | metric                                    | threshold | observed | pass |
|-------------------------------|-------------------------------------------|-----------|----------|------|
| G1_detection                  | ffpp in-dataset video AUC                 | 0.9000    | 0.8125   | FAIL |
| G2_generalization             | cross-dataset AUC minus SBI               | -0.0300   | —        | —    |
| G3_circuit_value_detection    | AUC minus best baseline                   | 0.0200    | 0.0087   | FAIL |
| G3_circuit_value_localization | patch AUC minus best baseline             | 0.0300    | -0.1022  | FAIL |
| G4_structure                  | val NLL gain over random structure (nats) | 2.0000    | 164.2828 | PASS |
| G5_scale                      | circuit fit seconds                       | 3600.0000 | 28.4073  | PASS |

**Verdict: STOP — the detector does not work in-dataset; fix the representation first**
