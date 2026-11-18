# Release gate report

Decision: **pass**. CI exit code: **0**. Model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B`. Thinking: **on**.

Synthetic hand-authored tickets. This gate decides whether a prompt change may proceed; it deploys nothing.

| Variant | Split | Correct / cases | Accuracy | Median ms | Output tokens | Complete |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | all | 64 / 64 | 1.000 | 3013.9 | 24970 | 64 |
| baseline | development | 16 / 16 | 1.000 | 2893.4 | 6140 | 16 |
| baseline | heldout | 48 / 48 | 1.000 | 3038.5 | 18830 | 48 |
| candidate | all | 64 / 64 | 1.000 | 4389.2 | 34542 | 64 |
| candidate | development | 16 / 16 | 1.000 | 4179.9 | 8270 | 16 |
| candidate | heldout | 48 / 48 | 1.000 | 4586.2 | 26272 | 48 |

Negative control (deliberately bad prompt): 2 / 8 correct, rejected by the gate.

Policy checks:

- valid_membership: pass
- required_heldout_count: pass
- complete_responses: pass
- strict_label_contract: pass
- candidate_absolute_floor: pass
- every_category_floor: pass
- candidate_delta_floor: pass
- paired_regression_budget: pass
- negative_control_rejected: pass
