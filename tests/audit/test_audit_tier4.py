"""Unit tests for src.audit.tier4_minhash."""
import pytest
from src.audit.config import (
    AuditMatch,
    MINHASH_NUM_PERM,
    MINHASH_NUM_BANDS,
    MINHASH_ROWS_PER_BAND,
)
from src.audit.tier4_minhash import (
    compute_minhash,
    estimate_minhash_jaccard,
    MinHashLSHIndex,
    extract_security_entities,
    run_tier4_audit,
)


def test_compute_minhash_deterministic_and_length():
    tokens = ["sqlmap", "-u", "http://target.local/vuln.php?id=1", "--dbs", "--batch"]
    sig1 = compute_minhash(tokens)
    sig2 = compute_minhash(tokens)
    assert len(sig1) == MINHASH_NUM_PERM
    assert sig1 == sig2
    assert all(isinstance(x, int) for x in sig1)


def test_compute_minhash_empty():
    sig = compute_minhash([])
    assert len(sig) == MINHASH_NUM_PERM
    # Estimated jaccard with non-empty or empty should be 0.0
    assert estimate_minhash_jaccard(sig, sig) == 0.0


def test_estimate_minhash_jaccard_identical_and_divergent():
    tokens_base = [f"security_token_{i}" for i in range(100)]
    sig_base = compute_minhash(tokens_base)
    sig_identical = compute_minhash(tokens_base)
    assert estimate_minhash_jaccard(sig_base, sig_identical) == 1.0

    tokens_diff = [f"unrelated_token_{i}" for i in range(100)]
    sig_diff = compute_minhash(tokens_diff)
    j_diff = estimate_minhash_jaccard(sig_base, sig_diff)
    assert j_diff < 0.15


def test_minhash_lsh_index_candidate_retrieval():
    index = MinHashLSHIndex(num_bands=MINHASH_NUM_BANDS, rows_per_band=MINHASH_ROWS_PER_BAND)
    tokens_target = [f"word_{i}" for i in range(60)]
    # 95% similar tokens
    tokens_similar = [f"word_{i}" for i in range(57)] + ["diff_1", "diff_2", "diff_3"]
    # completely different tokens
    tokens_other = [f"different_{i}" for i in range(60)]

    sig_target = compute_minhash(tokens_target)
    sig_similar = compute_minhash(tokens_similar)
    sig_other = compute_minhash(tokens_other)

    index.insert("doc_target", sig_target)
    index.insert("doc_other", sig_other)

    candidates = index.query(sig_similar)
    assert "doc_target" in candidates
    assert "doc_other" not in candidates


def test_extract_security_entities():
    sample_text = (
        "Exploiting CVE-2023-38606 using metasploit module on host 192.168.1.100 "
        "targeting /api/v1/authenticate endpoint with sqlmap injection"
    )
    entities = extract_security_entities(sample_text)
    assert "cves" in entities
    assert "CVE-2023-38606" in entities["cves"]
    assert "192.168.1.100" in entities["hosts"]
    assert "/api/v1/authenticate" in entities["paths"]
    assert "metasploit" in entities["tools"]
    assert "sqlmap" in entities["tools"]


def test_run_tier4_audit_similarity_and_priority():
    # Record pair with very high similarity (J >= 0.80 -> high priority)
    common_content = " ".join([f"exploit_step_{i}" for i in range(80)])
    rec_train_high = {
        "id": "train_high_01",
        "messages": [
            {"role": "user", "content": f"Execute advanced exploit on target: {common_content}"},
            {"role": "assistant", "content": "Running payload to obtain remote code execution."},
        ]
    }
    rec_test_high = {
        "id": "test_high_01",
        "messages": [
            {"role": "user", "content": f"Execute advanced exploit on target: {common_content}"},
            {"role": "assistant", "content": "Running payload to obtain remote code execution."},
        ]
    }

    matches = run_tier4_audit([rec_train_high], [rec_test_high], "train", "test")
    assert len(matches) == 1
    m = matches[0]
    assert isinstance(m, AuditMatch)
    assert m.left_id == "train_high_01"
    assert m.right_id == "test_high_01"
    assert m.match_tier == "tier4_minhash"
    assert m.details["estimated_jaccard"] >= 0.80
    assert m.details["priority"] == "high"


def test_run_tier4_audit_secondary_entity_recall():
    # Record pair with lower general similarity but identical specific CVE and target host/path
    rec_train = {
        "id": "train_cve_01",
        "messages": [
            {"role": "user", "content": "Conduct assessment on 10.0.0.55 looking for CVE-2021-44228 via ldap queries"},
            {"role": "assistant", "content": "Injecting jndi ldap lookup payload into User-Agent header."},
        ]
    }
    rec_test = {
        "id": "test_cve_01",
        "messages": [
            {"role": "user", "content": "Inspect web server 10.0.0.55 for vulnerable Log4j CVE-2021-44228 exposure"},
            {"role": "assistant", "content": "Verifying DNS callback response from jndi lookup."},
        ]
    }

    matches = run_tier4_audit([rec_train], [rec_test], "train", "test")
    # Even if MinHash J is around medium or lower, secondary recall check links them via CVE-2021-44228 & host
    assert len(matches) >= 1
    m = next((match for match in matches if match.left_id == "train_cve_01" and match.right_id == "test_cve_01"), None)
    assert m is not None
    assert "shared_entities" in m.details
    assert "CVE-2021-44228" in m.details["shared_entities"].get("cves", [])


def test_run_tier4_audit_within_split_deduplication():
    tokens = " ".join([f"duplicate_token_{i}" for i in range(50)])
    rec_a = {"id": "rec_1", "messages": [{"role": "user", "content": tokens}]}
    rec_b = {"id": "rec_2", "messages": [{"role": "user", "content": tokens}]}

    matches = run_tier4_audit([rec_a, rec_b], [rec_a, rec_b], "train", "train")
    assert len(matches) == 1
    assert matches[0].left_id == "rec_1"
    assert matches[0].right_id == "rec_2"
