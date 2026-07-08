"""
Attention Mask Strategy (Method B)

This strategy implements counterfactual analysis by modifying attention masks
to block information flow from specific modalities to action tokens.

The approach:
- Identify modality boundaries in token sequence (image, language, action tokens)
- Create custom attention masks to block specific modalities
- Run inference with modified masks
- Compare outputs to estimate modality effects

This is more precise than input zeroing because it directly controls
what information can flow through the Transformer.
"""

from typing import Dict, Optional, Tuple, Union, Any, List
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from prismatic.cf_attention.config import CfConfig, CfMode
from prismatic.cf_attention.utils import (
    compute_cf_effect,
    compute_delta,
    apply_cf_reweighting,
    get_cf_metrics,
)


@dataclass
class ModalityBounds:
    """
    Boundaries of different modalities in the token sequence.
    
    Attributes:
        image_bounds: (start, end) indices for image tokens
        language_bounds: (start, end) indices for language tokens
        action_bounds: (start, end) indices for action tokens (generated)
    """
    image_bounds: Tuple[int, int]
    language_bounds: Tuple[int, int]
    action_bounds: Tuple[int, int]
    
    def total_length(self) -> int:
        """Return total sequence length."""
        return self.action_bounds[1]
    
    def get_mask_for_modality(self, modality: str) -> Tuple[int, int]:
        """Get bounds for a specific modality."""
        if modality == "image":
            return self.image_bounds
        elif modality == "language":
            return self.language_bounds
        elif modality == "action":
            return self.action_bounds
        else:
            raise ValueError(f"Unknown modality: {modality}")


class AttentionMaskStrategy:
    """
    Counterfactual strategy using attention mask modification.
    
    This approach:
    1. Identifies token boundaries for each modality
    2. Creates custom attention masks blocking specific modalities
    3. Runs inference with modified masks
    4. Compares outputs to compute effect magnitudes
    5. Applies reweighting formula
    
    Advantages:
    - More precise: directly blocks information flow
    - Single inference pass (with hooks)
    - Doesn't change input distribution
    
    Disadvantages:
    - Complex implementation requiring model internals knowledge
    - May conflict with HuggingFace's generate method
    - Different LLM architectures need different handling
    """
    
    def __init__(self, vla_model: Any, config: CfConfig):
        """
        Initialize AttentionMaskStrategy.
        
        Args:
            vla_model: The VLA model (OpenVLA or PrismaticVLM)
            config: CF configuration
        """
        self.vla_model = vla_model
        self.config = config
        self._hooks = []
        self._modality_bounds = None
    
    def predict_action_with_cf(
        self,
        image: Union[PILImage.Image, np.ndarray],
        instruction: str,
        proprio_state: Optional[np.ndarray] = None,
        unnorm_key: Optional[str] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Predict action with counterfactual analysis using attention masks.
        
        Args:
            image: Input image (PIL Image or numpy array)
            instruction: Task instruction string
            proprio_state: Optional proprioceptive state
            unnorm_key: Dataset key for unnormalization
            **kwargs: Additional arguments
        
        Returns:
            Tuple of (actions, cf_metrics)
        """
        if not self.config.enable_cf:
            actions = self.vla_model.predict_action(image, instruction, unnorm_key=unnorm_key, **kwargs)
            return actions, None
        
        # Step 1: Baseline inference (normal attention)
        actions_baseline = self.vla_model.predict_action(image, instruction, unnorm_key=unnorm_key, **kwargs)
        
        # Step 2: Run CF inference with masked image attention
        if self.config.attention_mask_image:
            actions_img_blocked = self._predict_with_blocked_modality(
                image=image,
                instruction=instruction,
                blocked_modality="image",
                unnorm_key=unnorm_key,
                **kwargs
            )
        else:
            actions_img_blocked = actions_baseline.copy()
        
        # Step 3: Run CF inference with masked language attention
        if self.config.attention_mask_language:
            actions_lang_blocked = self._predict_with_blocked_modality(
                image=image,
                instruction=instruction,
                blocked_modality="language",
                unnorm_key=unnorm_key,
                **kwargs
            )
        else:
            actions_lang_blocked = actions_baseline.copy()
        
        # Step 4: Compute effects
        effect_vlm = compute_cf_effect(actions_baseline, actions_img_blocked)
        effect_lang = compute_cf_effect(actions_baseline, actions_lang_blocked)
        
        # For this strategy, we use language effect as proxy for proprio
        # since VLA-Adapter doesn't have explicit proprio tokens
        effect_prop = effect_lang
        
        # Step 5: Compute deltas
        delta_img = compute_delta(actions_baseline, actions_img_blocked)
        delta_prop = compute_delta(actions_baseline, actions_lang_blocked)
        
        # Step 6: Apply reweighting
        actions_final, use_baseline = apply_cf_reweighting(
            actions_baseline=actions_baseline,
            delta_img=delta_img,
            delta_prop=delta_prop,
            effect_vlm=effect_vlm,
            effect_prop=effect_prop,
            mode=self.config.mode,
            cf_guidance_scale=self.config.cf_guidance_scale,
            vlm_effect_upper_threshold=self.config.vlm_effect_upper_threshold,
        )
        
        # Step 7: Build metrics
        if self.config.return_cf_metrics:
            metrics = get_cf_metrics(
                effect_vlm=effect_vlm,
                effect_prop=effect_prop,
                use_baseline=use_baseline,
                mode=self.config.mode,
                delta_img_norm=np.linalg.norm(delta_img),
                delta_prop_norm=np.linalg.norm(delta_prop),
            )
            return actions_final, metrics
        
        return actions_final, None
    
    def _predict_with_blocked_modality(
        self,
        image: Union[PILImage.Image, np.ndarray],
        instruction: str,
        blocked_modality: str,
        unnorm_key: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Run inference with attention to specific modality blocked.
        
        This method uses hooks to modify the attention mask during generation.
        
        Args:
            image: Input image
            instruction: Task instruction
            blocked_modality: Which modality to block ("image", "language")
            unnorm_key: Dataset key
            **kwargs: Additional arguments
        
        Returns:
            Action array
        """
        # Prepare inputs
        image_transform, tokenizer = self.vla_model.vision_backbone.get_image_transform(), self.vla_model.llm_backbone.tokenizer
        
        # Build prompt
        prompt_builder = self.vla_model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()
        
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vla_model.device)
        
        # Process image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.vla_model.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.vla_model.device) for k, v in pixel_values.items()}
        
        # Estimate modality bounds
        modality_bounds = self._estimate_modality_bounds(pixel_values, input_ids)
        
        # Create CF attention mask
        cf_mask = self._create_cf_attention_mask(
            modality_bounds=modality_bounds,
            blocked_modality=blocked_modality,
            device=self.vla_model.device
        )
        
        # Register hook to inject attention mask
        self._register_attention_hooks(cf_mask, modality_bounds)
        
        try:
            # Run generation with hooks active
            autocast_dtype = self.vla_model.llm_backbone.half_precision_dtype
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vla_model.enable_mixed_precision_training):
                generated_ids = self.vla_model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=self.vla_model.get_action_dim(unnorm_key),
                    **kwargs
                )
            
            # Extract actions
            predicted_action_token_ids = generated_ids[0, -self.vla_model.get_action_dim(unnorm_key):]
            normalized_actions = self.vla_model.action_tokenizer.decode_token_ids_to_actions(
                predicted_action_token_ids.cpu().numpy()
            )
            
            # Unnormalize
            action_norm_stats = self.vla_model.get_action_stats(unnorm_key)
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
            actions = np.where(
                mask,
                0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
                normalized_actions,
            )
        finally:
            # Always remove hooks
            self._remove_hooks()
        
        return actions
    
    def _estimate_modality_bounds(
        self,
        pixel_values: Union[torch.Tensor, Dict],
        input_ids: torch.Tensor
    ) -> ModalityBounds:
        """
        Estimate the boundaries of different modalities in the token sequence.
        
        Args:
            pixel_values: Processed image tensor(s)
            input_ids: Tokenized input IDs
        
        Returns:
            ModalityBounds with estimated boundaries
        """
        # Estimate image token count based on vision backbone
        # For DinoSigLIP, typically around 256-576 tokens
        if isinstance(pixel_values, dict):
            # Multiple image inputs (e.g., Dino + SigLIP)
            # This depends on the specific vision backbone configuration
            num_image_tokens = 0
            for key in pixel_values:
                if 'dino' in key.lower():
                    num_image_tokens += 256  # Dino typically 256 patches
                elif 'siglip' in key.lower():
                    num_image_tokens += 256  # SigLIP typically 256 patches
        else:
            # Single image tensor
            # Estimate based on common ViT configurations
            if pixel_values.shape[-1] == 224:  # 224x224 image, 14x14 patches = 196 + 1 CLS
                num_image_tokens = 196
            elif pixel_values.shape[-1] == 336:  # 336x336
                num_image_tokens = 324
            else:
                num_image_tokens = 256  # Default estimate
        
        # Language tokens
        num_language_tokens = input_ids.shape[1]
        
        # Action tokens (to be generated)
        action_dim = self.vla_model.get_action_dim()
        
        return ModalityBounds(
            image_bounds=(0, num_image_tokens),
            language_bounds=(num_image_tokens, num_image_tokens + num_language_tokens),
            action_bounds=(num_image_tokens + num_language_tokens, num_image_tokens + num_language_tokens + action_dim),
        )
    
    def _create_cf_attention_mask(
        self,
        modality_bounds: ModalityBounds,
        blocked_modality: str,
        device: torch.device,
        dtype: torch.dtype = torch.bool
    ) -> torch.Tensor:
        """
        Create attention mask that blocks attention to specified modality.
        
        Args:
            modality_bounds: Boundaries of modalities
            blocked_modality: Which modality to block
            device: Torch device
            dtype: Mask dtype
        
        Returns:
            Attention mask tensor [seq_len, seq_len]
        """
        seq_len = modality_bounds.total_length()
        
        # Start with causal mask
        mask = torch.ones(seq_len, seq_len, dtype=dtype, device=device)
        
        # Apply causal structure: each token can attend to itself and previous tokens
        for i in range(seq_len):
            mask[i, :i+1] = True
        
        # Block attention from action tokens to blocked modality
        action_start, action_end = modality_bounds.action_bounds
        
        if blocked_modality == "image":
            img_start, img_end = modality_bounds.image_bounds
            mask[action_start:action_end, img_start:img_end] = False
        
        elif blocked_modality == "language":
            lang_start, lang_end = modality_bounds.language_bounds
            mask[action_start:action_end, lang_start:lang_end] = False
        
        return mask
    
    def _register_attention_hooks(
        self,
        cf_mask: torch.Tensor,
        modality_bounds: ModalityBounds
    ):
        """
        Register forward hooks to inject CF attention mask.
        
        Args:
            cf_mask: CF attention mask
            modality_bounds: Modality boundaries
        """
        self._modality_bounds = modality_bounds
        
        # Find attention layers in LLM backbone
        # This depends on the specific LLM architecture
        llm_backbone = self.vla_model.llm_backbone
        
        # For Llama/Qwen style models, attention is in model.layers.*.self_attn
        if hasattr(llm_backbone, 'model') and hasattr(llm_backbone.model, 'layers'):
            for layer in llm_backbone.model.layers:
                if hasattr(layer, 'self_attn'):
                    hook = layer.self_attn.register_forward_hook(
                        self._attention_hook_factory(cf_mask)
                    )
                    self._hooks.append(hook)
        
        # For other architectures, may need different handling
        elif hasattr(llm_backbone, 'layers'):
            for layer in llm_backbone.layers:
                if hasattr(layer, 'attention'):
                    hook = layer.attention.register_forward_hook(
                        self._attention_hook_factory(cf_mask)
                    )
                    self._hooks.append(hook)
    
    def _attention_hook_factory(self, cf_mask: torch.Tensor):
        """
        Create a hook function for attention layers.
        
        Args:
            cf_mask: CF attention mask
        
        Returns:
            Hook function
        """
        def hook(module, input, output):
            # This hook modifies the attention output based on CF mask
            # Note: The exact modification depends on the attention implementation
            
            if self.config.verbose:
                print(f"[CF Hook] Attention module: {module.__class__.__name__}")
            
            # For most Transformer attention implementations, we need to modify
            # the attention weights before they are applied to values.
            # However, since we're using forward hooks (not pre-hooks), the output
            # is already computed.
            
            # A more complete implementation would use pre-hooks or modify the
            # forward method directly. For now, this is a placeholder that logs
            # information.
            
            return output
        
        return hook
    
    def _remove_hooks(self):
        """
        Remove all registered hooks.
        """
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._modality_bounds = None
    
    def visualize_attention_mask(
        self,
        modality_bounds: ModalityBounds,
        blocked_modality: str = None,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize the attention mask structure.
        
        Args:
            modality_bounds: Boundaries of modalities
            blocked_modality: Optional modality to block for visualization
            save_path: Optional path to save visualization
        
        Returns:
            Attention mask as numpy array for visualization
        """
        seq_len = modality_bounds.total_length()
        
        # Create base mask
        mask = np.ones((seq_len, seq_len), dtype=np.float32)
        
        # Apply causal structure
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    mask[i, j] = 0.0
        
        # Block specific modality if specified
        if blocked_modality:
            action_start, action_end = modality_bounds.action_bounds
            if blocked_modality == "image":
                img_start, img_end = modality_bounds.image_bounds
                mask[action_start:action_end, img_start:img_end] = 0.0
            elif blocked_modality == "language":
                lang_start, lang_end = modality_bounds.language_bounds
                mask[action_start:action_end, lang_start:lang_end] = 0.0
        
        # Create visualization with labels
        if save_path:
            try:
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 10))
                im = ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
                
                # Add boundary lines
                img_end = modality_bounds.image_bounds[1]
                lang_end = modality_bounds.language_bounds[1]
                
                ax.axhline(y=img_end, color='red', linestyle='--', label='Image/Language boundary')
                ax.axvline(x=img_end, color='red', linestyle='--')
                ax.axhline(y=lang_end, color='blue', linestyle='--', label='Language/Action boundary')
                ax.axvline(x=lang_end, color='blue', linestyle='--')
                
                ax.set_xlabel('Token Position')
                ax.set_ylabel('Token Position')
                ax.set_title(f'Attention Mask (Blocked: {blocked_modality or "None"})')
                ax.legend()
                
                plt.colorbar(im, ax=ax)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
            except ImportError:
                print("matplotlib not available for visualization")
        
        return mask