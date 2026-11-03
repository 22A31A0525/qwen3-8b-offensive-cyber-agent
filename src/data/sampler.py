"""sampler.py
Globally Deterministic Distributed Curriculum Sampler for Qwen3-8B (v3 Pipeline).

Enforces:
1. Exact 3:1:1 batch sampling ratio (3 Exploit : 1 Reasoning : 1 Tool Replay per 5 samples).
2. Globally deterministic schedule partitioned across DDP ranks (Rank r: global[r::world_size]).
3. Zero inter-rank sample duplication with guaranteed disjoint coverage.
4. Resumable step offset (skips already executed steps upon resumption).
5. Dynamic computation of optimizer steps from actual draws (no hardcoded step limits).
"""
import math
import random
from typing import Any, Dict, Iterator, List, Optional
from torch.utils.data import Sampler


class StratifiedCurriculumSampler(Sampler[int]):
    """Samples dataset indices maintaining strict 3:1:1 curriculum balance.
    
    For every 5 samples drawn globally:
    - 3 are drawn from the Exploit domain (60%)
    - 1 is drawn from the Reasoning domain (20%)
    - 1 is drawn from the Tool-Calling Replay domain (20%)
    """

    def __init__(
        self,
        dataset: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 42,
        resume_step: int = 0,
        drop_last: bool = False,
        dataset_records: Optional[List[Dict[str, Any]]] = None,
        world_size: Optional[int] = None,
    ):
        if dataset is None and dataset_records is not None:
            dataset = dataset_records
        elif dataset is None:
            raise ValueError("Must provide either 'dataset' or 'dataset_records'.")

        if num_replicas is None and world_size is not None:
            num_replicas = world_size

        self.dataset = dataset
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0
        self.seed = seed
        self.epoch = 0
        self.resume_step = resume_step
        self.drop_last = drop_last

        # Categorize indices
        self.exploit_indices: List[int] = []
        self.reasoning_indices: List[int] = []
        self.replay_indices: List[int] = []

        for idx, rec in enumerate(dataset):
            cat = (rec.get("metadata", {}).get("category") or rec.get("category", "")).lower()
            src = (rec.get("metadata", {}).get("source_dataset") or "").lower()
            if "exploit" in cat or "red_team" in src or "pentestr1" in src:
                self.exploit_indices.append(idx)
            elif "reasoning" in cat or "cybersecurity" in src:
                self.reasoning_indices.append(idx)
            elif "replay" in cat or "tool" in cat or "hermes" in src or "nemotron" in src:
                self.replay_indices.append(idx)
            else:
                self.replay_indices.append(idx)

        # Compute maximum complete 3:1:1 cycles
        num_cycles_exp = len(self.exploit_indices) // 3
        num_cycles_rsn = len(self.reasoning_indices)
        num_cycles_rpl = len(self.replay_indices)
        self.num_cycles = min(num_cycles_exp, num_cycles_rsn, num_cycles_rpl)

        self.total_size = self.num_cycles * 5
        self.effective_batch_size = self.num_replicas * self.batch_size * self.gradient_accumulation_steps
        self.dynamic_optimizer_steps = math.ceil(self.total_size / self.effective_batch_size)

        # Samples allocated to this rank
        self.num_samples = int(math.ceil(self.total_size / self.num_replicas))

    def compute_dynamic_optimizer_steps(self) -> int:
        """Returns exact number of optimizer steps required for one full epoch."""
        return self.dynamic_optimizer_steps

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def set_resume_step(self, resume_step: int):
        self.resume_step = resume_step

    def get_global_schedule(self) -> List[int]:
        """Generates the full deterministic 3:1:1 sequence for the current epoch."""
        rng = random.Random(self.seed + self.epoch)
        
        exp = list(self.exploit_indices)
        rsn = list(self.reasoning_indices)
        rpl = list(self.replay_indices)
        
        rng.shuffle(exp)
        rng.shuffle(rsn)
        rng.shuffle(rpl)

        ordered_indices: List[int] = []
        for i in range(self.num_cycles):
            ordered_indices.extend(exp[i * 3 : (i + 1) * 3])
            ordered_indices.append(rsn[i])
            ordered_indices.append(rpl[i])

        return ordered_indices

    def __iter__(self) -> Iterator[int]:
        ordered_indices = self.get_global_schedule()

        # Slice globally for current rank
        if self.num_replicas > 1:
            rank_indices = ordered_indices[self.rank : self.total_size : self.num_replicas]
        else:
            rank_indices = ordered_indices

        # Apply resume offset if resuming training midway
        if self.resume_step > 0:
            skip_count = self.resume_step * self.batch_size * self.gradient_accumulation_steps
            rank_indices = rank_indices[skip_count:]

        return iter(rank_indices)

    def __len__(self) -> int:
        if self.num_replicas > 1:
            total = len(range(self.rank, self.total_size, self.num_replicas))
        else:
            total = self.total_size
            
        if self.resume_step > 0:
            skip_count = self.resume_step * self.batch_size * self.gradient_accumulation_steps
            return max(0, total - skip_count)
        return total
