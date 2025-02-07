import torch
import gc
import sys
from typing import Optional, List, Set
from collections import defaultdict
import logging


class MemoryLeakDetector:
    def __init__(self, threshold_mb: float = 10.0):
        self.threshold_mb = threshold_mb
        self.previous_tensors: Set[int] = set()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_tensor_memory(self) -> float:
        """Return total PyTorch tensor memory usage in MB."""
        return torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    def get_tensor_references(self) -> defaultdict:
        """Get all current tensor objects and their reference counts."""
        tensor_refs = defaultdict(int)
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    tensor_refs[id(obj)] += 1
            except Exception:
                continue
        return tensor_refs

    def check_memory_leak(self, iteration: int, context: str = "") -> List[int]:
        """
        Check for potential memory leaks by comparing tensor references.
        Returns list of tensor IDs that might be leaking.
        """
        current_tensors = set(self.get_tensor_references().keys())

        # Find tensors that persist across iterations
        persistent_tensors = current_tensors.intersection(self.previous_tensors)

        # Check if memory usage exceeds threshold
        current_memory = self.get_tensor_memory()

        self.logger.info(f"Iteration {iteration} - Context: {context}")
        self.logger.info(f"Current GPU memory usage: {current_memory:.2f} MB")
        self.logger.info(f"Number of persistent tensors: {len(persistent_tensors)}")

        if current_memory > self.threshold_mb:
            self.logger.warning(f"Memory usage exceeds threshold ({self.threshold_mb} MB)")
            self.analyze_tensor_types()

        self.previous_tensors = current_tensors
        return list(persistent_tensors)

    def analyze_tensor_types(self):
        """Analyze types and shapes of tensors currently in memory."""
        type_count = defaultdict(int)
        shape_count = defaultdict(int)
        total_memory = 0

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    tensor = obj
                elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                    tensor = obj.data
                else:
                    continue

                type_count[str(tensor.dtype)] += 1
                shape_count[str(list(tensor.shape))] += 1
                total_memory += tensor.element_size() * tensor.nelement()
            except Exception:
                continue

        self.logger.info("\nTensor Analysis:")
        self.logger.info("Dtype distribution:")
        for dtype, count in type_count.items():
            self.logger.info(f"  {dtype}: {count}")

        self.logger.info("\nShape distribution:")
        for shape, count in shape_count.items():
            self.logger.info(f"  {shape}: {count}")

        self.logger.info(f"\nTotal tensor memory: {total_memory / 1024 / 1024:.2f} MB")