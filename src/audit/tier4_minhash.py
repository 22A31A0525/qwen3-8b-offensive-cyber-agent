"""Tier 4 MinHash-LSH & Semantic Recall Audit Module.

Implements deterministic MinHash signature generation (128 permutations, fixed seed 42)
modulo Mersenne prime 2^61 - 1, Locality Sensitive Hashing (LSH, 16 bands x 8 rows),
and secondary recall checks for shared security entities (CVEs, hosts, paths, tool families).
"""
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib
import random
import re

from src.audit.config import (
    AuditMatch,
    MINHASH_NUM_PERM,
    MINHASH_NUM_BANDS,
    MINHASH_ROWS_PER_BAND,
    DETERMINISTIC_SEED,
    JACCARD_HIGH_PRIORITY,
    JACCARD_MEDIUM_PRIORITY,
    SUSPECTED_HIGH,
    SUSPECTED_MEDIUM,
    FIELD_FULL_CONVERSATION,
)
from src.audit.normalizer import (
    normalize_text,
    extract_audit_fields,
)

# Mersenne prime 2^61 - 1
MERSENNE_PRIME: int = (1 << 61) - 1

# Pre-generate deterministic permutations using fixed seed 42
def _generate_permutations(
    num_perm: int = MINHASH_NUM_PERM,
    seed: int = DETERMINISTIC_SEED,
) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    perms: List[Tuple[int, int]] = []
    for _ in range(num_perm):
        a = rng.randint(1, MERSENNE_PRIME - 1)
        b = rng.randint(0, MERSENNE_PRIME - 1)
        perms.append((a, b))
    return perms


_PERMUTATIONS = _generate_permutations()


def _tokenize(text: Optional[str]) -> List[str]:
    """Tokenizes text into normalized word tokens."""
    if not text:
        return []
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def _hash_token(token: str) -> int:
    """Hashes a single token into a 61-bit integer modulo MERSENNE_PRIME."""
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    raw_int = int.from_bytes(digest[:8], byteorder="big")
    return raw_int % MERSENNE_PRIME


def compute_minhash(
    tokens: List[str],
    num_perm: int = MINHASH_NUM_PERM,
) -> List[int]:
    """Computes a deterministic MinHash signature of length num_perm for given tokens.

    Args:
        tokens: List of token strings.
        num_perm: Number of permutations (default 128).

    Returns:
        List of num_perm integer signatures. If tokens is empty, returns list of zeros.
    """
    if not tokens:
        return [0] * num_perm

    perms = _PERMUTATIONS if num_perm == MINHASH_NUM_PERM else _generate_permutations(num_perm, DETERMINISTIC_SEED)
    unique_tokens = set(tokens)
    token_hashes = [_hash_token(t) for t in unique_tokens]

    sig: List[int] = []
    for a, b in perms:
        min_val = MERSENNE_PRIME
        for x in token_hashes:
            val = (a * x + b) % MERSENNE_PRIME
            if val < min_val:
                min_val = val
        sig.append(min_val)

    return sig


def estimate_minhash_jaccard(sig_a: List[int], sig_b: List[int]) -> float:
    """Estimates Jaccard similarity between two MinHash signatures.

    Args:
        sig_a: First signature.
        sig_b: Second signature.

    Returns:
        Estimated Jaccard similarity in [0.0, 1.0]. Returns 0.0 if either signature is empty.
    """
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    if all(x == 0 for x in sig_a) or all(x == 0 for x in sig_b):
        return 0.0

    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


class MinHashLSHIndex:
    """Locality-Sensitive Hashing (LSH) index over MinHash signatures using banding."""

    def __init__(
        self,
        num_bands: int = MINHASH_NUM_BANDS,
        rows_per_band: int = MINHASH_ROWS_PER_BAND,
    ) -> None:
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.buckets: Dict[Tuple[int, Tuple[int, ...]], Set[str]] = {}
        self.signatures: Dict[str, List[int]] = {}

    def insert(self, doc_id: str, signature: List[int]) -> None:
        """Inserts a document ID and its MinHash signature into the LSH index."""
        self.signatures[doc_id] = signature
        if all(x == 0 for x in signature):
            return

        for b in range(self.num_bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            band_slice = tuple(signature[start:end])
            bucket_key = (b, band_slice)
            if bucket_key not in self.buckets:
                self.buckets[bucket_key] = set()
            self.buckets[bucket_key].add(doc_id)

    def query(self, signature: List[int]) -> Set[str]:
        """Queries the index for candidate document IDs colliding in at least one band."""
        if all(x == 0 for x in signature):
            return set()

        candidates: Set[str] = set()
        for b in range(self.num_bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            band_slice = tuple(signature[start:end])
            bucket_key = (b, band_slice)
            if bucket_key in self.buckets:
                candidates.update(self.buckets[bucket_key])
        return candidates


def extract_security_entities(text: Optional[str]) -> Dict[str, Set[str]]:
    """Extracts security entities including CVE IDs, target IP hosts, file/URL paths,

    and security tool family names.
    """
    if not text:
        return {"cves": set(), "hosts": set(), "paths": set(), "tools": set()}

    raw_text = str(text)

    # 1. CVE extraction
    cves = set(re.findall(r"CVE-\d{4}-\d{4,7}", raw_text, re.IGNORECASE))
    cves = {cve.upper() for cve in cves}

    # 2. Host / IPv4 extraction
    hosts = set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", raw_text))

    # 3. Path extraction (endpoint paths with at least two path segments)
    paths = set(re.findall(r"(?:/[a-zA-Z0-9_\-\.\~]+){2,}", raw_text))

    # 4. Security tool family extraction
    tool_pattern = r"\b(nmap|sqlmap|metasploit|hydra|nikto|gobuster|ffuf|wireshark|burp|john|hashcat|aircrack|responder|mimikatz|crackmapexec|impacket)\b"
    tools = set(re.findall(tool_pattern, raw_text, re.IGNORECASE))
    tools = {t.lower() for t in tools}

    return {
        "cves": cves,
        "hosts": hosts,
        "paths": paths,
        "tools": tools,
    }


def run_tier4_audit(
    left_records: List[Dict[str, Any]],
    right_records: List[Dict[str, Any]],
    left_split: str,
    right_split: str,
) -> List[AuditMatch]:
    """Executes Tier 4 MinHash-LSH audit and secondary entity recall check.

    Flags candidate pairs with estimated Jaccard >= 0.80 as high priority,
    0.65 <= J < 0.80 as medium priority, and checks secondary entity overlap.

    Args:
        left_records: Records from source split.
        right_records: Records from comparison split.
        left_split: Name of source split.
        right_split: Name of comparison split.

    Returns:
        List of AuditMatch objects meeting priority or entity recall thresholds.
    """
    matches: List[AuditMatch] = []
    is_within_split = (left_split == right_split)

    # 1. Build LSH index and entity index over right_records
    lsh_index = MinHashLSHIndex(num_bands=MINHASH_NUM_BANDS, rows_per_band=MINHASH_ROWS_PER_BAND)
    right_data: Dict[str, Dict[str, Any]] = {}
    cve_index: Dict[str, Set[str]] = {}
    host_index: Dict[str, Set[str]] = {}

    for idx, rec in enumerate(right_records):
        r_id = str(rec.get("id") or rec.get("dialogue_id") or rec.get("conv_id") or f"r_{idx}")
        fields = extract_audit_fields(rec)
        meaningful_text = fields[FIELD_FULL_CONVERSATION]
        tokens = _tokenize(meaningful_text)
        sig = compute_minhash(tokens)
        entities = extract_security_entities(meaningful_text)

        lsh_index.insert(r_id, sig)
        right_data[r_id] = {
            "signature": sig,
            "entities": entities,
        }

        for cve in entities["cves"]:
            if cve not in cve_index:
                cve_index[cve] = set()
            cve_index[cve].add(r_id)

        for host in entities["hosts"]:
            if host not in host_index:
                host_index[host] = set()
            host_index[host].add(r_id)

    # 2. Query left_records
    for l_idx, l_rec in enumerate(left_records):
        l_id = str(l_rec.get("id") or l_rec.get("dialogue_id") or l_rec.get("conv_id") or f"l_{l_idx}")
        l_fields = extract_audit_fields(l_rec)
        l_meaningful_text = l_fields[FIELD_FULL_CONVERSATION]
        l_tokens = _tokenize(l_meaningful_text)
        l_sig = compute_minhash(l_tokens)
        l_entities = extract_security_entities(l_meaningful_text)

        # Retrieve candidates from LSH
        candidate_ids = lsh_index.query(l_sig)

        # Secondary recall candidates from shared CVEs or Hosts
        for cve in l_entities["cves"]:
            if cve in cve_index:
                candidate_ids.update(cve_index[cve])
        for host in l_entities["hosts"]:
            if host in host_index:
                candidate_ids.update(host_index[host])

        for r_id in candidate_ids:
            if is_within_split and l_id >= r_id:
                continue

            r_info = right_data[r_id]
            r_sig = r_info["signature"]
            r_entities = r_info["entities"]

            j_est = estimate_minhash_jaccard(l_sig, r_sig)

            # Evaluate shared entities
            shared_cves = l_entities["cves"] & r_entities["cves"]
            shared_hosts = l_entities["hosts"] & r_entities["hosts"]
            shared_paths = l_entities["paths"] & r_entities["paths"]
            shared_tools = l_entities["tools"] & r_entities["tools"]

            has_entity_hit = bool(shared_cves or (shared_hosts and shared_paths))

            # Threshold and priority logic
            if j_est >= JACCARD_HIGH_PRIORITY:
                priority = "high"
                classification = SUSPECTED_HIGH
            elif j_est >= JACCARD_MEDIUM_PRIORITY:
                priority = "medium"
                classification = SUSPECTED_MEDIUM
            elif has_entity_hit:
                priority = "medium"
                classification = SUSPECTED_MEDIUM
            else:
                continue

            shared_entities_dict: Dict[str, List[str]] = {}
            if shared_cves:
                shared_entities_dict["cves"] = sorted(list(shared_cves))
            if shared_hosts:
                shared_entities_dict["hosts"] = sorted(list(shared_hosts))
            if shared_paths:
                shared_entities_dict["paths"] = sorted(list(shared_paths))
            if shared_tools:
                shared_entities_dict["tools"] = sorted(list(shared_tools))

            matches.append(
                AuditMatch(
                    left_id=l_id,
                    right_id=r_id,
                    left_split=left_split,
                    right_split=right_split,
                    match_tier="tier4_minhash",
                    field_matched=FIELD_FULL_CONVERSATION,
                    details={
                        "estimated_jaccard": round(j_est, 4),
                        "priority": priority,
                        "classification": classification,
                        "shared_entities": shared_entities_dict,
                    },
                )
            )

    return matches
