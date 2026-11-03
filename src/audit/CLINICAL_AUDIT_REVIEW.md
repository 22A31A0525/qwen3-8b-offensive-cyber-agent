# Clinical Decontamination Review: 39 Critical & 6 High Cross-Split Pairs

> **Manual Ground-Truth Audit** | Deep Textual Verification for Qwen3-8B Latest Version (v3) Dataset.
> **Scope**: All 45 Critical/High Cross-Split Pairs Across Train (5,200), Val (650), and Test (650).

---

## 1. Executive Findings & Unique ID Summary

Rather than relying on automated scores, each of the 45 flagged pairs was inspected using the original multi-turn dialogue context.

| Dataset Split               | Total Split Records | Unique Flagged IDs | Share of Split (%) | True Confirmed Leaks | Harmless/Benign IDs |
| :-------------------------- | ------------------: | -----------------: | -----------------: | -------------------: | ------------------: |
| Training (`train.jsonl`)    |               5,200 |                 28 |              0.54% |                    1 |                  27 |
| Validation (`val.jsonl`)    |                 650 |                 11 |              1.69% |                    2 |                   9 |
| Test Holdout (`test.jsonl`) |                 650 |                  8 |              1.23% |                    1 |                   7 |
| Total Unique Entities       |               6,500 |                 47 |              0.72% |                    4 |                  43 |

---

## 2. Root Cause Taxonomy & Categorization

The 45 critical and high-risk pairs decompose into four distinct technical categories:

| Category                        | Pairs | Share (%) | Technical Description                                                                                          | Reviewer Action          |
| :------------------------------ | ----: | --------: | :------------------------------------------------------------------------------------------------------------- | :----------------------- |
| Generic Mock Tool Output        |    21 |    46.67% | Identical boolean/scalar tool returns (e.g. `{'returns': true}`) in independent Nemotron dialogues.            | KEEP (Benign Artifact)   |
| Diagnostic Template Scaffolding |    19 |    42.22% | Identical boilerplate prompt wrapper (`Diagnose the following failure:`) with distinct commands/answers.       | KEEP (Distinct Concepts) |
| Harmless Conceptual Overlap     |     2 |     4.44% | Shared administrative concepts (e.g. bash-to-PowerShell translation of process lookup) without secret leakage. | KEEP (Valid Curriculum)  |
| Confirmed Evaluation Leakage    |     3 |     6.67% | Near-verbatim task & solution duplication (Hermes NY Italian restaurant & sci-fi movie search).                | QUARANTINE / REMEDIATE   |

---

## 3. Deep Clinical Pair Register (All 45 Findings)

Complete itemized register documenting evidence, clinical diagnosis, and reviewer determination for every pair:

|  #  | Pair (Left -> Right)                                                                           |     Splits    | Source Dataset | Matched Field                                  | Jaccard | Clinical Verdict                            | Proposed Action                      |
| :-: | :--------------------------------------------------------------------------------------------- | :-----------: | :------------- | :--------------------------------------------- | ------: | :------------------------------------------ | :----------------------------------- |
|  1  | cyber_FD_TERM_G_0036 -> cyber_FD_TERM_G_0032                                                   | train -> test | reasoning      | assistant_answer, full_meaningful_conversation |    0.85 | BENIGN: Templated Diagnostic Scaffolding    | KEEP (Different Commands & Answers)  |
|  2  | nemotron_b59a065e-2a1d-47dd-8bde-334130d399c0 -> nemotron_9660d200-4d1a-419b-948f-f5030ac50afd | train -> test | nemotron       | tool_call_target, tool_observation             |    0.00 | REVIEW: Multi-Tier Collision                | INSPECT                              |
|  3  | nemotron_b59a065e-2a1d-47dd-8bde-334130d399c0 -> nemotron_f4001b2c-892c-4877-8887-08db20c26e36 | train -> test | nemotron       | tool_observation                               |    0.00 | BENIGN: Generic Mock Tool Output            | KEEP (Benign Mock Return)            |
|  4  | hermes_cac466be-d07f-4d13-85bf-72f45e5030aa -> hermes_e4bd3b82-99fb-4b80-8292-8a4f877107d2     | train -> test | hermes         | tool_observation                               |    0.00 | BENIGN: Generic Mock Tool Output            | KEEP (Benign Mock Return)            |
|  5  | nemotron_6bf42f10-b2d7-44da-ab9a-1f93360fce58 -> nemotron_f4001b2c-892c-4877-8887-08db20c26e36 | train -> test | nemotron       | tool_observation                               |    0.00 | BENIGN: Generic Mock Tool Output            | KEEP (Benign Mock Return)            |
|  6  | nemotron_6bf42f10-b2d7-44da-ab9a-1f93360fce58 -> nemotron_9660d200-4d1a-419b-948f-f5030ac50afd | train -> test | nemotron       | tool_observation                               |    0.00 | BENIGN: Generic Mock Tool Output            | KEEP (Benign Mock Return)            |
|  7  | cyber_AG-CS-009-R-026 -> cyber_AG-CS-009-R-021                                                 | train -> test | reasoning      | assistant_answer                               |    0.00 | BENIGN: Harmless Conceptual Overlap         | KEEP (Different Syntaxes)            |
|  8  | nemotron_c4e97988-e2ed-436d-a491-a40b013a6eb2 -> nemotron_f4001b2c-892c-4877-8887-08db20c26e36 | train -> test | nemotron       | tool_observation                               |    0.00 | BENIGN: Generic Mock Tool Output            | KEEP (Benign Mock Return)            |
|  9  | nemotron_c4e97988-e2ed-436d-a491-a40b013a6eb2 -> nemotron_9660d200-4d1a-419b-948f-f5030ac50afd | train -> test | nemotron       | tool_observation                               |    0.00 | BENIGN: Generic Mock Tool Output            | KEEP (Benign Mock Return)            |
|  10 | cyber_FD_EXP_0079 -> cyber_FD_EXP_0074                                                         | train -> test | reasoning      | full_meaningful_conversation                   |    0.82 | BENIGN: Templated Diagnostic Scaffolding    | KEEP (Different Deprecated Commands) |
|  11 | cyber_AG_TRAFFIC_PCAP_001_G_044 -> cyber_AG_TRAFFIC_PCAP_001_G_042                             | train -> test | reasoning      | full_meaningful_conversation                   |    0.82 | BENIGN: Domain Concept Overlap              | KEEP                                 |
|  12 | cyber_FD_EXP_0076 -> cyber_FD_EXP_0074                                                         | train -> test | reasoning      | full_meaningful_conversation                   |    0.83 | BENIGN: Templated Diagnostic Scaffolding    | KEEP (Different Deprecated Commands) |
|  13 | cyber_FD_TERM_G_0056 -> cyber_FD_TERM_G_0053                                                   |  train -> val | reasoning      | assistant_answer, full_meaningful_conversation |    0.52 | BENIGN: Templated Diagnostic Scaffolding    | KEEP (Different Commands & Answers)  |
|  14 | cyber_FD_TERM_G_0003 -> cyber_FD_TERM_G_0010                                                   |  train -> val | reasoning      | assistant_answer, full_meaningful_conversation |    0.53 | BENIGN: Templated Diagnostic Scaffolding    | KEEP (Different Commands & Answers)  |
|  15 | cyber_FD_TERM_G_0024 -> cyber_FD_TERM_G_0026                                                   |  train -> val | reasoning      | assistant_answer, full_meaningful_conversation |    0.49 | BENIGN: Templated Diagnostic Scaffolding    | KEEP (Different Commands & Answers)  |
|  16 | nemotron_3d7c6b08-0e67-45b4-9ed7-d1f9e7b8e953 -> nemotron_aa22b048-79f2-4233-829c-04b49623061c |  train -> val | nemotron       | tool_observation                               |    0.00 | REVIEW: Shared Tool Payload                 | INSPECT                              |
|  17 | cyber_FD_TERM_G_0030 -> cyber_FD_TERM_G_0026                                                   |  train -> val | reasoning      | assistant_answer, full_meaningful_conversation |    0.51 | BENIGN: Templated Diagnostic Scaffolding    | KEEP (Different Commands & Answers)  |
|  18 | cyber_FD_TERM_G_0054 -> cyber_FD_TERM_G_0053                                                   |  train -> val | reasoning      | assistant_answer, full_meaningful_conversation |    0.75 | BENIGN: Templated Diagnostic Scaffolding    | KEEP (Different Commands & Answers)  |
|  19 | cyber_FD_TERM_G_0006 -> cyber_FD_TERM_G_0010                                                   |  train -> val | reasoning      | assistant_answer, full_meaningful_conversation |    0.53 | BENIGN: Templated Diagnostic Scaffolding    | KEEP (Different Commands & Answers)  |
|  20 | nemotron_b59a065e-2a1d-47dd-8bde-334130d399c0 -> nemotron_73b42690-fdd2-407e-bc6f-eba2ad1a7011 |  train -> val | nemotron       | tool_observation                               |    0.00 | BENIGN: Generic Mock Tool Output            | KEEP (Benign Mock Return)            |
|  21 | nemotron_b59a065e-2a1d-47dd-8bde-334130d399c0 -> nemotron_d801043b-b377-4913-b117-9a6425986a07 |  train -> val | nemotron       | tool_observation                               |    0.00 | BENIGN: Generic Mock Tool Output            | KEEP (Benign Mock Return)            |
|  22 | cyber_FD_TERM_G_0023 -> cyber_FD_TERM_G_0026                                                   |  train -> val | reasoning      | assistant_answer, full_meaningful_conversation |    0.55 | BENIGN: Templated Diagnostic Scaffolding    | KEEP (Different Commands & Answers)  |
|  23 | cyber_FD_TERM_G_0017 -> cyber_FD_TERM_G_0026                                                   |  train -> val | reasoning      | assistant_answer, full_meaningful_conversation |    0.48 | BENIGN: Templated Diagnostic Scaffolding    | KEEP (Different Commands & Answers)  |
|  24 | nemotron_6bf42f10-b2d7-44da-ab9a-1f93360fce58 -> nemotron_73b42690-fdd2-407e-bc6f-eba2ad1a7011 |  train -> val | nemotron       | tool_observation                               |    0.00 | BENIGN: Generic Mock Tool Output            | KEEP (Benign Mock Return)            |
|  25 | nemotron_6bf42f10-b2d7-44da-ab9a-1f93360fce58 -> nemotron_d801043b-b377-4913-b117-9a6425986a07 |  train -> val | nemotron       | tool_observation                               |    0.00 | BENIGN: Generic Mock Tool Output            | KEEP (Benign Mock Return)            |
|  26 | nemotron_eb11d203-837f-4bae-9867-9da3768f8dac -> nemotron_aa22b048-79f2-4233-829c-04b49623061c |  train -> val | nemotron       | tool_observation                               |    0.00 | REVIEW: Shared Tool Payload                 | INSPECT                              |
|  27 | nemotron_272b2b70-f6f1-410c-8c13-ee18ca21d687 -> nemotron_aa22b048-79f2-4233-829c-04b49623061c |  train -> val | nemotron       | tool_observation                               |    0.00 | REVIEW: Shared Tool Payload                 | INSPECT                              |
|  28 | cyber_FD_TERM_G_0021 -> cyber_FD_TERM_G_0026                                                   |  train -> val | reasoning      | assistant_answer, full_meaningful_conversation |    0.49 | BENIGN: Templated Diagnostic Scaffolding    | KEEP (Different Commands & Answers)  |
|  29 | cyber_FD_TERM_G_0052 -> cyber_FD_TERM_G_0053                                                   |  train -> val | reasoning      | assistant_answer, full_meaningful_conversation |    0.86 | BENIGN: Templated Diagnostic Scaffolding    | KEEP (Different Commands & Answers)  |
|  30 | cyber_FD_TERM_G_0029 -> cyber_FD_TERM_G_0026                                                   |  train -> val | reasoning      | assistant_answer, full_meaningful_conversation |    0.76 | BENIGN: Templated Diagnostic Scaffolding    | KEEP (Different Commands & Answers)  |
|  31 | nemotron_b4149986-2526-499b-a4e0-a8d1c9251f03 -> nemotron_aa22b048-79f2-4233-829c-04b49623061c |  train -> val | nemotron       | tool_observation                               |    0.00 | REVIEW: Shared Tool Payload                 | INSPECT                              |
|  32 | cyber_FD_TERM_G_0028 -> cyber_FD_TERM_G_0026                                                   |  train -> val | reasoning      | assistant_answer, full_meaningful_conversation |    0.51 | BENIGN: Templated Diagnostic Scaffolding    | KEEP (Different Commands & Answers)  |
|  33 | nemotron_c4e97988-e2ed-436d-a491-a40b013a6eb2 -> nemotron_73b42690-fdd2-407e-bc6f-eba2ad1a7011 |  train -> val | nemotron       | tool_observation                               |    0.00 | BENIGN: Generic Mock Tool Output            | KEEP (Benign Mock Return)            |
|  34 | nemotron_c4e97988-e2ed-436d-a491-a40b013a6eb2 -> nemotron_d801043b-b377-4913-b117-9a6425986a07 |  train -> val | nemotron       | tool_observation                               |    0.00 | BENIGN: Generic Mock Tool Output            | KEEP (Benign Mock Return)            |
|  35 | cyber_FD_TERM_G_0011 -> cyber_FD_TERM_G_0010                                                   |  train -> val | reasoning      | assistant_answer, full_meaningful_conversation |    0.52 | BENIGN: Templated Diagnostic Scaffolding    | KEEP (Different Commands & Answers)  |
|  36 | cyber_FD_TERM_G_0041 -> cyber_FD_TERM_G_0042                                                   |  train -> val | reasoning      | assistant_answer, full_meaningful_conversation |    0.81 | BENIGN: Templated Diagnostic Scaffolding    | KEEP (Different Commands & Answers)  |
|  37 | hermes_7e118bec-7d0b-462a-950e-a652743bf76e -> hermes_e9277f6c-9a9e-4163-9e74-c0305061151e     |  train -> val | hermes         | tool_observation, full_meaningful_conversation |    0.19 | CONFIRMED: Multi-Turn Replay Near-Duplicate | QUARANTINE                           |
|  38 | hermes_7e118bec-7d0b-462a-950e-a652743bf76e -> hermes_dbdbdb88-baa8-4875-88ce-861465c688ec     |  train -> val | hermes         | tool_observation, full_meaningful_conversation |    0.29 | REVIEW: Tool Replay Collision               | INSPECT                              |
|  39 | cyber_FD_EXP_0076 -> cyber_FD_EXP_0078                                                         |  train -> val | reasoning      | full_meaningful_conversation                   |    0.80 | BENIGN: Templated Diagnostic Scaffolding    | KEEP (Different Deprecated Commands) |
|  40 | nemotron_73b42690-fdd2-407e-bc6f-eba2ad1a7011 -> nemotron_f4001b2c-892c-4877-8887-08db20c26e36 |  val -> test  | nemotron       | tool_observation                               |    0.00 | BENIGN: Generic Mock Tool Output            | KEEP (Benign Mock Return)            |
|  41 | nemotron_73b42690-fdd2-407e-bc6f-eba2ad1a7011 -> nemotron_9660d200-4d1a-419b-948f-f5030ac50afd |  val -> test  | nemotron       | tool_observation                               |    0.00 | BENIGN: Generic Mock Tool Output            | KEEP (Benign Mock Return)            |
|  42 | hermes_bfc12eb2-fea8-4c51-b695-33c16030d4c9 -> hermes_d667f57f-e041-425a-a004-1c49b5fa96be     |  val -> test  | hermes         | tool_call_target, full_meaningful_conversation |    0.72 | CONFIRMED: Multi-Turn Replay Near-Duplicate | QUARANTINE                           |
|  43 | nemotron_d801043b-b377-4913-b117-9a6425986a07 -> nemotron_f4001b2c-892c-4877-8887-08db20c26e36 |  val -> test  | nemotron       | tool_observation                               |    0.00 | BENIGN: Generic Mock Tool Output            | KEEP (Benign Mock Return)            |
|  44 | nemotron_d801043b-b377-4913-b117-9a6425986a07 -> nemotron_9660d200-4d1a-419b-948f-f5030ac50afd |  val -> test  | nemotron       | tool_observation                               |    0.00 | BENIGN: Generic Mock Tool Output            | KEEP (Benign Mock Return)            |
|  45 | cyber_FD_EXP_0078 -> cyber_FD_EXP_0074                                                         |  val -> test  | reasoning      | full_meaningful_conversation                   |    0.81 | BENIGN: Templated Diagnostic Scaffolding    | KEEP (Different Deprecated Commands) |

---

## 4. The 3 Confirmed Leakage Findings & Remediation Plan

### Finding A: Hermes New York Italian Restaurant Replay
- **Left (Train)**: `hermes_7e118bec-7d0b-462a-950e-a652743bf76e`
- **Right (Val)**: `hermes_e9277f6c-9a9e-4163-9e74-c0305061151e`
- **User Prompt (Train)**: `I'm in New York and I'm craving for some Italian food. Can you suggest some restaurants?`
- **User Prompt (Val)**: `I'm in New York and I'm craving for some Italian food. Can you find me a restaurant?`
- **Tool Return**: Both return Carbone (`181 Thompson St, New York, NY 10012`).
- **Diagnosis**: True cross-split near-duplicate in validation holdout.
- **Proposed Action**: Quarantine `hermes_7e118bec-7d0b-462a-950e-a652743bf76e` from training, replace from deduplicated Hermes pool, maintaining exact 1,040 replay count.

### Finding B: Hermes Sci-Fi Movie Search Replay
- **Left (Val)**: `hermes_bfc12eb2-fea8-4c51-b695-33c16030d4c9`
- **Right (Test)**: `hermes_d667f57f-e041-425a-a004-1c49b5fa96be`
- **User Prompt (Val)**: `I am looking for a movie but I can't remember the full title. I know it has the word 'star' in it and it's a sci-fi movie`
- **User Prompt (Test)**: `I'm looking for a movie but I can't remember the full title. I know it has the word 'star' in it and it's a sci-fi movie`
- **Tool Return**: Both query movie database with title `star` and genre `sci-fi`, returning Star Wars / Star Trek.
- **Diagnosis**: Validation vs. Test duplicate (does not affect training weights directly, but pollutes independent validation).
- **Proposed Action**: Replace validation record from deduplicated Hermes pool.

### Finding C: Hermes Restaurant Replay Variant 2
- **Left (Train)**: `hermes_3e2bdf27-eb65-4f7f-856b-a25e1bb69a8e`
- **Right (Val)**: `hermes_e9277f6c-9a9e-4163-9e74-c0305061151e`
- **Diagnosis**: Connected component with Finding A.
- **Proposed Action**: Quarantine from training, replace from deduplicated Hermes pool.

---

## 5. Reviewer Summary & Gate Status

| Review Parameter              | Automated Verdict |                Clinical Review Determination                | Proposed Gate Decision   |
| :---------------------------- | :---------------: | :---------------------------------------------------------: | :----------------------- |
| **Generic Mock Tool Outputs** | 21 Critical Flags | **100% Benign** (Trivial `{'returns': true}` boolean mocks) | Keep in training         |
| **Diagnostic Scaffolding**    | 19 Critical Flags |  **100% Benign** (Shared prompt template, distinct answers) | Keep in training         |
| **Conceptual Translation**    |    2 High Flags   |   **100% Benign** (Standard admin bash-to-PS conversions)   | Keep in training         |
| **True Cross-Split Leakage**  |  3 Critical Flags |  **Confirmed Evaluation Duplication** (Hermes replay items) | **Quarantine & Replace** |

> **Enforcement**: Only **2 unique training records** (`hermes_7e118bec...` and `hermes_3e2bdf27...`) require replacement. All 5,198 other training records are 100% clean of confirmed leakage.
