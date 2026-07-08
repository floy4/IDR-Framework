# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

"""Modality bounds data structure for tracking positions in X-VLA sequence.

X-VLA Sequence Structure:
┌─────────────────────────────────────────────────────────────────┐
│  [action_tokens(with_proprio) | vlm_features | aux_visual | soft_prompts] │
│                                                                  │
│  Action Tokens 内部结构:                                        │
│  [action_emb | proprio_emb | time_emb] × T_act                  │
│  每个位置：[0:dim_action | dim_action:dim_action+dim_propio |   │
│           dim_action+dim_propio:dim_action+dim_propio+dim_time] │
│                                                                  │
│  CF Attention:                                                   │
│  - VLM/Aux: 可通过注意力掩码精确阻断（序列位置独立）            │
│  - Proprio: 使用输入置零阻断（嵌入在特征维度，非序列位置）      │
└─────────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModalityBounds:
    """Records the position boundaries of each modality in the X-VLA sequence.

    Sequence: [action_tokens(with_proprio) | vlm | aux_visual | soft_prompts]

    Action tokens contain: [action, proprio, time] embeddings concatenated.

    Attributes:
        action_bounds: Tuple of (start, end) for action tokens segment.
        vlm_bounds: Tuple of (start, end) for VLM features segment.
        aux_visual_bounds: Tuple of (start, end) for aux visual inputs segment.
        soft_prompt_bounds: Tuple of (start, end) for soft prompts segment.
        total_seq_len: Total sequence length.

        proprio_offset: Offset of proprio within each action token (feature dim).
        proprio_len: Length of proprio embedding in each action token (feature dim).
        num_actions: Number of action tokens.

    Note:
        Proprio is NOT a separate sequence position - it's embedded in action tokens.
        For CF blocking:
        - VLM/Aux: use attention mask (sequence-level blocking)
        - Proprio: use input zeroing (feature-level blocking)
    """

    # Main segment bounds (sequence position)
    action_bounds: tuple[int, int] = (0, 0)
    vlm_bounds: tuple[int, int] = (0, 0)
    aux_visual_bounds: tuple[int, int] = (0, 0)
    soft_prompt_bounds: tuple[int, int] = (0, 0)
    total_seq_len: int = 0

    # Proprio position within action tokens (feature dimension)
    proprio_offset: int = 0  # Start position in feature dim
    proprio_len: int = 0     # Length in feature dim
    num_actions: int = 0     # Number of action tokens

    @property
    def total_action_tokens(self) -> int:
        """Total number of action tokens."""
        return self.action_bounds[1] - self.action_bounds[0]

    @property
    def total_vlm_tokens(self) -> int:
        """Total number of VLM feature tokens."""
        return self.vlm_bounds[1] - self.vlm_bounds[0]

    @property
    def total_aux_visual_tokens(self) -> int:
        """Total number of aux visual tokens."""
        return self.aux_visual_bounds[1] - self.aux_visual_bounds[0]

    @property
    def total_soft_prompt_tokens(self) -> int:
        """Total number of soft prompt tokens."""
        return self.soft_prompt_bounds[1] - self.soft_prompt_bounds[0]

    @property
    def has_aux_visual(self) -> bool:
        """Check if aux visual inputs are present."""
        return self.aux_visual_bounds[0] < self.aux_visual_bounds[1]

    @property
    def has_soft_prompts(self) -> bool:
        """Check if soft prompts are present."""
        return self.soft_prompt_bounds[0] < self.soft_prompt_bounds[1]

    @property
    def has_proprio(self) -> bool:
        """Check if proprio is present in action tokens."""
        return self.proprio_len > 0

    def get_modality_at_position(self, pos: int) -> str:
        """Determine which modality a sequence position belongs to.

        Args:
            pos: Sequence position.

        Returns:
            Modality name: "action", "vlm", "aux_visual", "soft_prompt", or "unknown".
            Note: proprio is NOT a separate sequence position.
        """
        if self.action_bounds[0] <= pos < self.action_bounds[1]:
            return "action"
        if self.vlm_bounds[0] <= pos < self.vlm_bounds[1]:
            return "vlm"
        if self.aux_visual_bounds[0] <= pos < self.aux_visual_bounds[1]:
            return "aux_visual"
        if self.soft_prompt_bounds[0] <= pos < self.soft_prompt_bounds[1]:
            return "soft_prompt"
        return "unknown"

    def get_bounds_for_modality(self, modality: str) -> tuple[int, int]:
        """Get position bounds for a given modality.

        Args:
            modality: One of "action", "vlm", "aux_visual", "soft_prompt", "image", "proprio".
                      "image" is alias for both vlm + aux_visual (all visual).
                      "proprio" returns action_bounds (proprio is inside action tokens).

        Returns:
            (start, end) tuple for the modality.
        """
        if modality == "action":
            return self.action_bounds
        elif modality == "vlm":
            return self.vlm_bounds
        elif modality == "aux_visual":
            return self.aux_visual_bounds
        elif modality == "soft_prompt":
            return self.soft_prompt_bounds
        elif modality == "image":
            # All visual features (vlm + aux_visual)
            return (self.vlm_bounds[0], self.aux_visual_bounds[1])
        elif modality == "proprio":
            # Proprio is embedded in action tokens, not a separate segment
            # Return action bounds for sequence-level operations
            # Note: For CF blocking, use input zeroing, not attention mask
            return self.action_bounds
        else:
            return (0, 0)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "action_bounds": self.action_bounds,
            "vlm_bounds": self.vlm_bounds,
            "aux_visual_bounds": self.aux_visual_bounds,
            "soft_prompt_bounds": self.soft_prompt_bounds,
            "total_seq_len": self.total_seq_len,
            "proprio_offset": self.proprio_offset,
            "proprio_len": self.proprio_len,
            "num_actions": self.num_actions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModalityBounds":
        """Create from dictionary."""
        return cls(
            action_bounds=data.get("action_bounds", (0, 0)),
            vlm_bounds=data.get("vlm_bounds", (0, 0)),
            aux_visual_bounds=data.get("aux_visual_bounds", (0, 0)),
            soft_prompt_bounds=data.get("soft_prompt_bounds", (0, 0)),
            total_seq_len=data.get("total_seq_len", 0),
            proprio_offset=data.get("proprio_offset", 0),
            proprio_len=data.get("proprio_len", 0),
            num_actions=data.get("num_actions", 0),
        )

    def __repr__(self) -> str:
        return (
            f"ModalityBounds(\n"
            f"  action_bounds={self.action_bounds},\n"
            f"  vlm_bounds={self.vlm_bounds},\n"
            f"  aux_visual_bounds={self.aux_visual_bounds},\n"
            f"  soft_prompt_bounds={self.soft_prompt_bounds},\n"
            f"  total_seq_len={self.total_seq_len},\n"
            f"  proprio_offset={self.proprio_offset},\n"
            f"  proprio_len={self.proprio_len},\n"
            f"  num_actions={self.num_actions},\n"
            f"  total_action_tokens={self.total_action_tokens},\n"
            f"  total_vlm_tokens={self.total_vlm_tokens},\n"
            f"  total_aux_visual_tokens={self.total_aux_visual_tokens},\n"
            f")"
        )


def create_modality_bounds(
    num_actions: int,
    vlm_token_count: int,
    aux_visual_token_count: int = 0,
    soft_prompt_count: int = 0,
    proprio_offset: int = 0,
    proprio_len: int = 0,
) -> ModalityBounds:
    """Helper function to create ModalityBounds from token counts.

    Sequence: [action_tokens(with_proprio) | vlm | aux_visual | soft_prompts]

    Args:
        num_actions: Number of action tokens (T_act).
        vlm_token_count: Number of VLM feature tokens (T_vlm).
        aux_visual_token_count: Number of aux visual tokens (T_aux).
        soft_prompt_count: Number of soft prompt tokens (T_soft).
        proprio_offset: Offset of proprio in action token feature dim.
        proprio_len: Length of proprio embedding in feature dim.

    Returns:
        ModalityBounds with computed positions.

    Example:
        >>> bounds = create_modality_bounds(
        ...     num_actions=30,
        ...     vlm_token_count=128,
        ...     aux_visual_token_count=64,
        ...     soft_prompt_count=32,
        ...     proprio_offset=20,
        ...     proprio_len=20,
        ... )
        >>> bounds.vlm_bounds
        (30, 158)
    """
    current_pos = 0

    # Action tokens come first (proprio embedded inside)
    action_bounds = (current_pos, current_pos + num_actions)
    current_pos += num_actions

    # VLM features
    vlm_bounds = (current_pos, current_pos + vlm_token_count)
    current_pos += vlm_token_count

    # Aux visual inputs
    aux_visual_bounds = (current_pos, current_pos + aux_visual_token_count)
    current_pos += aux_visual_token_count

    # Soft prompts
    soft_prompt_bounds = (current_pos, current_pos + soft_prompt_count)
    current_pos += soft_prompt_count

    return ModalityBounds(
        action_bounds=action_bounds,
        vlm_bounds=vlm_bounds,
        aux_visual_bounds=aux_visual_bounds,
        soft_prompt_bounds=soft_prompt_bounds,
        total_seq_len=current_pos,
        proprio_offset=proprio_offset,
        proprio_len=proprio_len,
        num_actions=num_actions,
    )