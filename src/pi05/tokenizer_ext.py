"""Extended tokenizer with modality boundary tracking.

This module provides an extended version of PaligemmaTokenizer that
returns modality boundary information, without modifying the original file.
"""

import logging
from typing import Optional

import numpy as np

from openpi.models.tokenizer import PaligemmaTokenizer
from .modality_bounds import ModalityBounds


class ExtendedPaligemmaTokenizer(PaligemmaTokenizer):
    """Extended PaligemmaTokenizer with modality boundary tracking.
    
    This class inherits from the original PaligemmaTokenizer and adds
    functionality to track the boundaries of different modalities (image,
    language, state) in the token sequence.
    
    This is particularly useful for Pi05 format where state is tokenized
    as part of the discrete language input.
    
    Example:
        >>> tokenizer = ExtendedPaligemmaTokenizer(max_len=48)
        >>> state = np.array([0.1, -0.5, 0.3])  # Normalized state
        >>> tokens, mask, bounds = tokenizer.tokenize_with_bounds(
        ...     prompt="pick up the cup",
        ...     state=state,
        ...     image_token_counts={"base_0_rgb": 100},
        ... )
        >>> print(bounds.language_bounds)
        (104, 115)
        >>> print(bounds.state_bounds)
        (118, 121)
    """
    
    def tokenize_with_bounds(
        self,
        prompt: str,
        state: np.ndarray | None = None,
        image_token_counts: dict[str, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, ModalityBounds]:
        """Tokenize prompt and state, returning modality boundary information.
        
        This method extends the base tokenize() to also compute and return
        the position boundaries of each modality in the token sequence.
        
        Args:
            prompt: The language prompt describing the task.
            state: Robot state array (normalized to [-1, 1]). If provided,
                   uses Pi05 format where state is discretized and tokenized.
                   If None, uses Pi0 format where state is continuous input.
            image_token_counts: Dictionary mapping camera names to their
                               token counts. E.g., {"base_0_rgb": 100}.
                               Used to compute image token boundaries.
        
        Returns:
            Tuple of (tokens, mask, modality_bounds):
            - tokens: Tokenized sequence as numpy array
            - mask: Attention mask as numpy array
            - modality_bounds: ModalityBounds object with position info
        
        Note:
            Image tokens are typically embedded before text tokens, but
            image embedding happens in the model, not the tokenizer.
            The image_token_counts parameter is used to track the expected
            image token positions.
        """
        # Use the original tokenize method
        tokens, mask = self.tokenize(prompt, state)
        
        # Compute modality bounds
        if state is not None:
            # Pi05 format: state is part of discrete token sequence
            modality_bounds = self._compute_pi05_bounds(
                prompt, state, image_token_counts or {}
            )
        else:
            # Pi0 format: state is continuous input (not in token sequence)
            modality_bounds = self._compute_pi0_bounds(
                prompt, image_token_counts or {}
            )
        
        return tokens, mask, modality_bounds
    
    def _compute_pi05_bounds(
        self,
        prompt: str,
        state: np.ndarray,
        image_token_counts: dict[str, int],
    ) -> ModalityBounds:
        """Compute modality bounds for Pi05 format.
        
        Pi05 token sequence structure:
        [Image tokens] + "Task: {prompt}, State: {discretized_state};\\nAction: "
        
        Where discretized_state is a space-separated string of integers
        representing the binned state values.
        """
        # Compute image bounds
        image_bounds = {}
        current_pos = 0
        for name, count in image_token_counts.items():
            image_bounds[name] = (current_pos, current_pos + count)
            current_pos += count
        
        image_offset = current_pos
        
        # Tokenize different parts separately to get boundaries
        # Format: "Task: {cleaned_text}, State: {state_str};\\nAction: "
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        
        # Discretize state (same logic as in base tokenizer)
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))
        
        # Tokenize each part
        task_tokens = self._tokenizer.encode("Task: ", add_bos=True)
        language_tokens = self._tokenizer.encode(cleaned_text)
        state_prefix_tokens = self._tokenizer.encode(", State: ")
        state_tokens = self._tokenizer.encode(state_str)
        action_suffix_tokens = self._tokenizer.encode(";\nAction: ")
        
        # Compute boundaries
        language_start = image_offset + len(task_tokens)
        language_end = language_start + len(language_tokens)
        
        state_start = language_end + len(state_prefix_tokens)
        state_end = state_start + len(state_tokens)
        
        prefix_len = state_end + len(action_suffix_tokens)
        
        return ModalityBounds(
            image_bounds=image_bounds,
            language_bounds=(language_start, language_end),
            state_bounds=(state_start, state_end),
            prefix_len=prefix_len,
            suffix_start=prefix_len,
        )
    
    def _compute_pi0_bounds(
        self,
        prompt: str,
        image_token_counts: dict[str, int],
    ) -> ModalityBounds:
        """Compute modality bounds for Pi0 format.
        
        Pi0 token sequence structure:
        [Image tokens] + "{prompt}\\n"
        
        In Pi0, state is NOT part of the token sequence - it's part of
        the continuous action expert input. So state_bounds will be empty.
        """
        # Compute image bounds
        image_bounds = {}
        current_pos = 0
        for name, count in image_token_counts.items():
            image_bounds[name] = (current_pos, current_pos + count)
            current_pos += count
        
        image_offset = current_pos
        
        # Tokenize prompt (with BOS and newline as start-of-answer)
        # Format: "{cleaned_text}\\n"
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        prompt_tokens = self._tokenizer.encode(cleaned_text, add_bos=True)
        newline_tokens = self._tokenizer.encode("\n")
        
        # Compute boundaries
        # Note: In Pi0, the entire text is considered "language"
        language_start = image_offset
        language_end = image_offset + len(prompt_tokens)
        
        # State is not in token sequence for Pi0
        prefix_len = language_end + len(newline_tokens)
        
        return ModalityBounds(
            image_bounds=image_bounds,
            language_bounds=(language_start, language_end),
            state_bounds=(prefix_len, prefix_len),  # Empty state bounds
            prefix_len=prefix_len,
            suffix_start=prefix_len,
        )
    
    def compute_bounds_from_token_counts(
        self,
        prompt: str,
        state: np.ndarray | None = None,
        image_token_counts: dict[str, int] | None = None,
    ) -> ModalityBounds:
        """Compute modality bounds without tokenizing.
        
        This is a convenience method when you already know the token counts
        and just want to compute the boundaries.
        
        Args:
            prompt: The language prompt.
            state: Robot state array (for Pi05 format).
            image_token_counts: Token counts for each camera.
        
        Returns:
            ModalityBounds with computed positions.
        """
        image_token_counts = image_token_counts or {}
        
        # Compute image bounds
        image_bounds = {}
        current_pos = 0
        for name, count in image_token_counts.items():
            image_bounds[name] = (current_pos, current_pos + count)
            current_pos += count
        
        image_offset = current_pos
        
        if state is not None:
            # Pi05 format
            cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            
            task_tokens = self._tokenizer.encode("Task: ", add_bos=True)
            language_tokens = self._tokenizer.encode(cleaned_text)
            state_prefix_tokens = self._tokenizer.encode(", State: ")
            state_tokens = self._tokenizer.encode(state_str)
            action_suffix_tokens = self._tokenizer.encode(";\nAction: ")
            
            language_start = image_offset + len(task_tokens)
            language_end = language_start + len(language_tokens)
            state_start = language_end + len(state_prefix_tokens)
            state_end = state_start + len(state_tokens)
            prefix_len = state_end + len(action_suffix_tokens)
            
            return ModalityBounds(
                image_bounds=image_bounds,
                language_bounds=(language_start, language_end),
                state_bounds=(state_start, state_end),
                prefix_len=prefix_len,
                suffix_start=prefix_len,
            )
        else:
            # Pi0 format
            cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
            prompt_tokens = self._tokenizer.encode(cleaned_text, add_bos=True)
            newline_tokens = self._tokenizer.encode("\n")
            
            language_start = image_offset
            language_end = image_offset + len(prompt_tokens)
            prefix_len = language_end + len(newline_tokens)
            
            return ModalityBounds(
                image_bounds=image_bounds,
                language_bounds=(language_start, language_end),
                state_bounds=(prefix_len, prefix_len),
                prefix_len=prefix_len,
                suffix_start=prefix_len,
            )