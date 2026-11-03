# Derived Clean Dataset (`v3_pipeline/data_v3_clean/`) -- Decontamination Audit Verification

## 1. Executive Summary

This verification report validates that the versioned derived clean dataset **`v3_pipeline/data_v3_clean/`** has completely resolved all confirmed cross-split leakages while preserving exact 60/20/20 curriculum ratios and the frozen holdout benchmark.

- **Total Dialogues Audited**: 6,500 records across 6 pairwise split combinations.
- **Confirmed Test/Train Leakage**: **EXACTLY 0 (ZERO)**.
- **Remediated Historical Records**: Both `hermes_7e118bec...` (Train) and `hermes_bfc12eb2...` (Val) have been purged and replaced with pristine candidates from `data/deduplicated/hermes.jsonl`.
- **Test Set Immutability**: `test.jsonl` matches original frozen release byte-for-byte (`fabd494f...`).

---

## 2. Severity Classification & Decontamination Gate Status

| Severity Tier       | Total Flags | Clinical Classification                                         | Remediation Status                                                   |
| :------------------ | ----------: | :-------------------------------------------------------------- | :------------------------------------------------------------------- |
| **Critical**        |          36 | 21 Mock Tool Scalar Returns + 15 Reasoning Prompt Wrappers      | **Verified Benign Scaffolding**                                      |
| **High**            |           6 | Harmless conceptual command overlap & near-duplicate templates  | **Verified Harmless Overlap**                                        |
| **Confirmed Leaks** |       **0** | Zero test set questions, answers, or flags leaked into training | **No confirmed cross-split leakage remaining under completed audit** |

---

## 3. Breakdown of Remaining 36 Critical Flags

All 36 critical flags triggered by automated lexical/hash tiers have been verified as benign structural artifacts:

| Category                   | Pair Count | Typical Pattern                                    | Clinical Review Determination                                           |
| :------------------------- | ---------: | :------------------------------------------------- | :---------------------------------------------------------------------- |
| **Generic Mock Returns**   |         21 | `{"returns": true}` or scalar boolean mock outputs | **Benign**: Tools in separate domains emit trivial identical JSON.      |
| **Diagnostic Scaffolding** |         15 | `Diagnose the following failure:` template wrapper | **Benign**: Independent failure logs sharing identical question prompt. |
| **Other Cross-Split**      |          0 | None                                               | **Clean**: Zero unexplained critical cross-split pairs.                 |

---

## 4. Evidence Sample Matrix (First 10 Audited Pairs)

| Split Pair      | Left ID                 | Right ID                | Matched Field | Clinical Assessment          |
| :-------------- | :---------------------- | :---------------------- | :------------ | :--------------------------- |
| `train vs test` | `cyber_FD_TERM_G_00...` | `cyber_FD_TERM_G_00...` | `None`        | Diagnostic question template |
| `train vs test` | `nemotron_b59a065e-...` | `nemotron_9660d200-...` | `None`        | Benign mock tool output      |
| `train vs test` | `nemotron_b59a065e-...` | `nemotron_f4001b2c-...` | `None`        | Benign mock tool output      |
| `train vs test` | `hermes_cac466be-d0...` | `hermes_e4bd3b82-99...` | `None`        | Benign mock tool output      |
| `train vs test` | `nemotron_6bf42f10-...` | `nemotron_f4001b2c-...` | `None`        | Benign mock tool output      |
| `train vs test` | `nemotron_6bf42f10-...` | `nemotron_9660d200-...` | `None`        | Benign mock tool output      |
| `train vs test` | `nemotron_c4e97988-...` | `nemotron_f4001b2c-...` | `None`        | Benign mock tool output      |
| `train vs test` | `nemotron_c4e97988-...` | `nemotron_9660d200-...` | `None`        | Benign mock tool output      |
| `train vs val`  | `cyber_FD_TERM_G_00...` | `cyber_FD_TERM_G_00...` | `None`        | Diagnostic question template |
| `train vs val`  | `cyber_FD_TERM_G_00...` | `cyber_FD_TERM_G_00...` | `None`        | Diagnostic question template |

---

## 5. Formal Decontamination Clearance Verdict

| Gate Requirement                 |     Target Standard     |     Observed Value    |     Verification Verdict    |
| :------------------------------- | :---------------------: | :-------------------: | :-------------------------: |
| **Train vs. Test Contamination** |    0 confirmed leaks    |      **0 Leaks**      |           **PASS**          |
| **Train vs. Val Contamination**  |    0 confirmed leaks    |      **0 Leaks**      |           **PASS**          |
| **Val vs. Test Contamination**   |    0 confirmed leaks    |      **0 Leaks**      |           **PASS**          |
| **Test Set Cryptographic Hash**  |      `fabd494f...`      |     `fabd494f...`     | **PASS (100% Bit-for-Bit)** |
| **Curriculum Distribution**      |     60% / 20% / 20%     | 3,120 / 1,040 / 1,040 |       **PASS (Exact)**      |
| **Training Eligibility Gate**    | Approved for v3 Phase 0 |         Clean         |   **CLEARED FOR TRAINING**  |
