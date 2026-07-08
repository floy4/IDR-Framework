"""Modality bounds data structure for tracking token positions.

This module defines the data structure for tracking the boundaries of different
modalities (image, language, state) in the token sequence of Pi0/Pi05 models.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModalityBounds:
    """Records the position boundaries of each modality in the token sequence.
    
    In Pi0/Pi05 models, the token sequence is structured as:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │  Prefix (bidirectional attention)                               │
    │  ┌────────────────┐ ┌──────────────┐ ┌──────────────┐          │
    │  │ Image Tokens   │ │ Language     │ │ State Tokens │          │
    │  │ (SigLIP)       │ │ Tokens       │ │ (Pi05 only)  │          │
    │  │ ar_mask=False  │ │ ar_mask=False│ │ ar_mask=False│          │
    │  └────────────────┘ └──────────────┘ └──────────────┘          │
    │                                                                  │
    │  Suffix (causal attention)                                       │
    │  ┌────────────────────────────────────────────────────┐         │
    │  │ Action Tokens                                      │         │
    │  │ ar_mask=[True, False, False, ...] (causal)         │         │
    │  └────────────────────────────────────────────────────┘         │
    └─────────────────────────────────────────────────────────────────┘
    
    This dataclass tracks the (start, end) positions of each modality,
    enabling counterfactual analysis by modifying attention masks.
    
    Attributes:
        image_bounds: Dictionary mapping camera names to (start, end) positions.
                      E.g., {"base_0_rgb": (0, 100), "left_wrist_0_rgb": (100, 200)}
        language_bounds: Tuple of (start, end) for language tokens.
        state_bounds: Tuple of (start, end) for state tokens (Pi05 only).
                      For Pi0 (non-Pi05), this is (prefix_len, prefix_len) as state
                      is part of continuous action expert input.
        prefix_len: Total length of prefix tokens (image + language + state).
        suffix_start: Starting position of action tokens (same as prefix_len).
    
    Example:
        >>> bounds = ModalityBounds(
        ...     image_bounds={"base_0_rgb": (0, 100)},
        ...     language_bounds=(100, 150),
        ...     state_bounds=(150, 170),
        ...     prefix_len=170,
        ...     suffix_start=170,
        ... )
        >>> bounds.total_image_tokens
        100
        >>> bounds.get_modality_at_position(120)
        'language'
    """
    
    # Image tokens range for each camera
    image_bounds: dict[str, tuple[int, int]] = field(default_factory=dict)
    
    # Language tokens range (pure language part, excluding "Task:" prefix)
    language_bounds: tuple[int, int] = (0, 0)
    
    # State tokens range (discretized state for Pi05, empty for Pi0)
    state_bounds: tuple[int, int] = (0, 0)
    
    # Total prefix length
    prefix_len: int = 0
    
    # Suffix (action) start position
    suffix_start: int = 0
    
    @property
    def total_image_tokens(self) -> int:
        """Total number of image tokens across all cameras."""
        return sum(end - start for start, end in self.image_bounds.values())
    
    @property
    def total_language_tokens(self) -> int:
        """Total number of language tokens."""
        return self.language_bounds[1] - self.language_bounds[0]
    
    @property
    def total_state_tokens(self) -> int:
        """Total number of state tokens."""
        return self.state_bounds[1] - self.state_bounds[0]
    
    @property
    def has_state_tokens(self) -> bool:
        """Check if state tokens are present (Pi05 format)."""
        return self.state_bounds[0] < self.state_bounds[1]
    
    def get_modality_at_position(self, pos: int) -> str:
        """Determine which modality a token position belongs to.
        
        Args:
            pos: Token position in the sequence.
        
        Returns:
            Modality name: "image_{camera_name}", "language", "state", "action", or "unknown".
        """
        # Check image tokens
        for name, (start, end) in self.image_bounds.items():
            if start <= pos < end:
                return f"image_{name}"
        
        # Check language tokens
        if self.language_bounds[0] <= pos < self.language_bounds[1]:
            return "language"
        
        # Check state tokens
        if self.state_bounds[0] <= pos < self.state_bounds[1]:
            return "state"
        
        # Check action tokens
        if pos >= self.suffix_start:
            return "action"
        
        return "unknown"
    
    def get_positions_for_modality(self, modality: str) -> list[tuple[int, int]]:
        """Get all position ranges for a given modality.
        
        Args:
            modality: One of "image", "language", "state", "action", or "image_{name}".
        
        Returns:
            List of (start, end) tuples for the modality.
        """
        if modality == "image":
            return list(self.image_bounds.values())
        elif modality.startswith("image_"):
            camera_name = modality[6:]  # Remove "image_" prefix
            if camera_name in self.image_bounds:
                return [self.image_bounds[camera_name]]
            return []
        elif modality == "language":
            return [self.language_bounds]
        elif modality == "state":
            return [self.state_bounds]
        elif modality == "action":
            # Action tokens start at suffix_start and extend to the end
            return [(self.suffix_start, -1)]  # -1 indicates end of sequence
        else:
            return []
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "image_bounds": self.image_bounds,
            "language_bounds": self.language_bounds,
            "state_bounds": self.state_bounds,
            "prefix_len": self.prefix_len,
            "suffix_start": self.suffix_start,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ModalityBounds":
        """Create from dictionary."""
        return cls(
            image_bounds=data.get("image_bounds", {}),
            language_bounds=data.get("language_bounds", (0, 0)),
            state_bounds=data.get("state_bounds", (0, 0)),
            prefix_len=data.get("prefix_len", 0),
            suffix_start=data.get("suffix_start", 0),
        )
    
    def __repr__(self) -> str:
        return (
            f"ModalityBounds(\n"
            f"  image_bounds={self.image_bounds},\n"
            f"  language_bounds={self.language_bounds},\n"
            f"  state_bounds={self.state_bounds},\n"
            f"  prefix_len={self.prefix_len},\n"
            f"  suffix_start={self.suffix_start},\n"
            f"  total_image_tokens={self.total_image_tokens},\n"
            f"  total_language_tokens={self.total_language_tokens},\n"
            f"  total_state_tokens={self.total_state_tokens},\n"
            f")"
        )


def create_modality_bounds(
    image_token_counts: dict[str, int],
    language_token_count: int,
    state_token_count: int = 0,
    task_prefix_tokens: int = 0,  # "Task: " tokens
    state_prefix_tokens: int = 0,  # ", State: " tokens
    action_suffix_tokens: int = 0,  # ";\nAction: " tokens
) -> ModalityBounds:
    """Helper function to create ModalityBounds from token counts.
    
    This is useful when you know the number of tokens for each modality
    but need to compute the actual positions.
    
    Args:
        image_token_counts: Number of tokens for each camera.
        language_token_count: Number of language tokens (excluding prefixes).
        state_token_count: Number of state tokens (0 for Pi0).
        task_prefix_tokens: Number of "Task: " prefix tokens.
        state_prefix_tokens: Number of ", State: " prefix tokens.
        action_suffix_tokens: Number of ";\nAction: " suffix tokens.
    
    Returns:
        ModalityBounds with computed positions.
    """
    current_pos = 0
    
    # Image bounds
    image_bounds = {}
    for name, count in image_token_counts.items():
        image_bounds[name] = (current_pos, current_pos + count)
        current_pos += count
    
    image_offset = current_pos
    
    # Language bounds (including "Task: " prefix)
    language_start = image_offset + task_prefix_tokens
    language_end = language_start + language_token_count
    current_pos = language_end
    
    # State bounds (including ", State: " prefix)
    state_start = current_pos + state_prefix_tokens
    state_end = state_start + state_token_count
    current_pos = state_end
    
    # Prefix length (including action suffix tokens like ";\nAction: ")
    prefix_len = current_pos + action_suffix_tokens
    
    return ModalityBounds(
        image_bounds=image_bounds,
        language_bounds=(language_start, language_end),
        state_bounds=(state_start, state_end),
        prefix_len=prefix_len,
        suffix_start=prefix_len,
    )