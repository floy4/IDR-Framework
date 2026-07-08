"""
Output Reweighting Strategy (Method C)

This strategy implements counterfactual analysis using classifier-free guidance (CFG)
style reweighting of output probability distributions.

The approach:
- Get logits/probabilities for conditioned input (full image + instruction)
- Get logits/probabilities for unconditioned input (zeroed image)
- Reweight probabilities using CFG formula
- Sample actions from reweighted distribution

Reference: Classifier-Free Guidance technique in language models and diffusion models
"""

from typing import Dict, Optional, Tuple, Union, Any, List
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from prismatic.cf_attention.config import CfConfig, CfMode
from prismatic.cf_attention.utils import (
    compute_cf_effect,
    compute_delta,
    apply_cf_reweighting,
    create_zero_image,
    get_cf_metrics,
    apply_cfg_reweighting,
    logits_to_probs,
    compute_kl_divergence,
)


class OutputReweightingStrategy:
    """
    Counterfactual strategy using output probability reweighting.
    
    This approach:
    1. Runs forward pass to get logits with full conditioning
    2. Runs forward pass to get logits without image conditioning
    3. Applies CFG-style reweighting to logits
    4. Samples actions from reweighted distribution
    5. Computes effect magnitudes using KL divergence
    
    Advantages:
    - Mathematically grounded in CFG theory
    - Works well for discrete token generation
    - Can leverage existing HuggingFace logits outputs
    
    Disadvantages:
    - Requires access to intermediate logits (may need custom forward)
    - Multiple forward passes
    - Probability reweighting may produce anomalous distributions
    """
    
    def __init__(self, vla_model: Any, config: CfConfig):
        """
        Initialize OutputReweightingStrategy.
        
        Args:
            vla_model: The VLA model (OpenVLA or PrismaticVLM)
            config: CF configuration
        """
        self.vla_model = vla_model
        self.config = config
    
    def predict_action_with_cf(
        self,
        image: Union[PILImage.Image, np.ndarray],
        instruction: str,
        proprio_state: Optional[np.ndarray] = None,
        unnorm_key: Optional[str] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Predict action with counterfactual analysis using output reweighting.
        
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
        
        # Get action dimension
        action_dim = self.vla_model.get_action_dim(unnorm_key)
        
        # Step 1: Get logits for conditioned input
        logits_conditioned, actions_baseline = self._get_logits_and_actions(
            image=image,
            instruction=instruction,
            unnorm_key=unnorm_key,
            **kwargs
        )
        
        # Step 2: Get logits for unconditioned input (zero image)
        zero_image = create_zero_image(image)
        cf_instruction = "" if self.config.clear_prompt_for_image_cf else instruction
        logits_unconditioned, actions_img0 = self._get_logits_and_actions(
            image=zero_image,
            instruction=cf_instruction,
            unnorm_key=unnorm_key,
            **kwargs
        )
        
        # Step 3: Apply CFG reweighting
        logits_reweighted = apply_cfg_reweighting(
            logits_conditioned=logits_conditioned,
            logits_unconditioned=logits_unconditioned,
            cfg_scale=self.config.cfg_scale,
        )
        
        # Step 4: Sample actions from reweighted logits
        actions_cfg = self._sample_from_logits(logits_reweighted, unnorm_key)
        
        # Step 5: Compute effects using KL divergence or L2 norm
        effect_vlm = self._compute_effect_from_logits(
            logits_conditioned, logits_unconditioned, method="kl"
        )
        
        # For proprio effect, we need another comparison
        # Since VLA doesn't have explicit proprio, use baseline as proxy
        effect_prop = 0.0  # Placeholder
        
        # Step 6: Compute deltas in action space
        delta_img = compute_delta(actions_baseline, actions_img0)
        delta_prop = np.zeros_like(delta_img)  # Placeholder
        
        # Step 7: Combine CFG output with CF reweighting
        # Option 1: Use pure CFG output
        if self.config.mode == CfMode.BASE:
            actions_final = actions_baseline
        elif self.config.mode == CfMode.A:
            # Mode A with CFG: just use CFG output
            actions_final = actions_cfg
        else:
            # Other modes: combine CFG with CF formula
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
        
        use_baseline = effect_vlm > self.config.vlm_effect_upper_threshold
        
        # Step 8: Build metrics
        if self.config.return_cf_metrics:
            metrics = get_cf_metrics(
                effect_vlm=effect_vlm,
                effect_prop=effect_prop,
                use_baseline=use_baseline,
                mode=self.config.mode,
                delta_img_norm=np.linalg.norm(delta_img),
                delta_prop_norm=0.0,
            )
            metrics["cfg_scale"] = self.config.cfg_scale
            metrics["kl_divergence"] = effect_vlm
            return actions_final, metrics
        
        return actions_final, None
    
    def _get_logits_and_actions(
        self,
        image: Union[PILImage.Image, np.ndarray],
        instruction: str,
        unnorm_key: Optional[str] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Get logits and decoded actions for a given input.
        
        This method runs a custom forward pass to capture intermediate logits.
        
        Args:
            image: Input image
            instruction: Task instruction
            unnorm_key: Dataset key
            **kwargs: Additional arguments
        
        Returns:
            Tuple of (logits tensor, actions array)
        """
        # Prepare inputs
        image_transform, tokenizer = self.vla_model.vision_backbone.get_image_transform(), self.vla_model.llm_backbone.tokenizer
        
        # Build prompt
        prompt_builder = self.vla_model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()
        
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vla_model.device)
        
        # Handle special tokens for Llama tokenizer
        from transformers import LlamaTokenizerFast
        if isinstance(tokenizer, LlamaTokenizerFast):
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
        
        # Process image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.vla_model.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.vla_model.device) for k, v in pixel_values.items()}
        
        # Run forward pass to get logits
        action_dim = self.vla_model.get_action_dim(unnorm_key)
        
        # Use a custom generation that captures logits
        logits_list = []
        
        def logits_hook(module, input, output):
            # Capture logits from the output
            if isinstance(output, torch.Tensor):
                logits_list.append(output.detach())
        
        # Register hook on the LLM head
        llm_backbone = self.vla_model.llm_backbone
        if hasattr(llm_backbone, 'lm_head'):
            hook = llm_backbone.lm_head.register_forward_hook(logits_hook)
        elif hasattr(llm_backbone, 'model') and hasattr(llm_backbone.model, 'lm_head'):
            hook = llm_backbone.model.lm_head.register_forward_hook(logits_hook)
        else:
            # Try to find the output layer
            for name, module in llm_backbone.named_modules():
                if 'lm_head' in name or 'output' in name:
                    hook = module.register_forward_hook(logits_hook)
                    break
        
        try:
            # Run generation
            autocast_dtype = self.vla_model.llm_backbone.half_precision_dtype
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vla_model.enable_mixed_precision_training):
                generated_ids = self.vla_model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=action_dim,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **kwargs
                )
            
            # Extract logits from generation outputs
            if hasattr(generated_ids, 'scores'):
                # Stack scores (logits) for each generated token
                logits = torch.stack(generated_ids.scores, dim=1)  # [batch, seq, vocab]
            elif logits_list:
                # Use captured logits from hook
                logits = logits_list[-1]  # Last captured logits
            else:
                # Fallback: get logits from final hidden states
                with torch.no_grad():
                    outputs = self.vla_model.llm_backbone(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        output_hidden_states=True,
                    )
                    hidden_states = outputs.hidden_states[-1]
                    if hasattr(llm_backbone, 'lm_head'):
                        logits = llm_backbone.lm_head(hidden_states)
                    else:
                        # Approximate with generated sequences
                        logits = torch.zeros(1, action_dim, tokenizer.vocab_size, device=self.vla_model.device)
            
            # Decode actions
            if hasattr(generated_ids, 'sequences'):
                generated_ids = generated_ids.sequences
            predicted_action_token_ids = generated_ids[0, -action_dim:]
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
            # Remove hook
            hook.remove()
        
        return logits, actions
    
    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        unnorm_key: Optional[str] = None,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Sample actions from logits.
        
        Args:
            logits: Logits tensor [batch, seq, vocab] or [batch, vocab]
            unnorm_key: Dataset key for action dimension
            temperature: Sampling temperature
        
        Returns:
            Decoded actions array
        """
        action_dim = self.vla_model.get_action_dim(unnorm_key)
        
        # Ensure logits has correct shape
        if logits.dim() == 2:
            # [batch, vocab] - single token logits
            logits = logits.unsqueeze(1)
        
        # Get action token logits (last action_dim positions)
        if logits.shape[1] > action_dim:
            action_logits = logits[:, -action_dim:, :]
        else:
            action_logits = logits
        
        # Apply temperature and sample
        probs = F.softmax(action_logits / temperature, dim=-1)
        
        # Sample token IDs
        sampled_ids = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
        sampled_ids = sampled_ids.view(action_logits.shape[0], action_logits.shape[1])
        
        # Decode to actions
        normalized_actions = self.vla_model.action_tokenizer.decode_token_ids_to_actions(
            sampled_ids[0].cpu().numpy()
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
        
        return actions
    
    def _compute_effect_from_logits(
        self,
        logits_conditioned: torch.Tensor,
        logits_unconditioned: torch.Tensor,
        method: str = "kl"
    ) -> float:
        """
        Compute effect magnitude from logits distributions.
        
        Args:
            logits_conditioned: Logits with full conditioning
            logits_unconditioned: Logits without image conditioning
            method: Computation method ("kl", "l2")
        
        Returns:
            Scalar effect value
        """
        # Convert to probabilities
        probs_conditioned = logits_to_probs(logits_conditioned)
        probs_unconditioned = logits_to_probs(logits_unconditioned)
        
        # Flatten for comparison
        probs_cond_flat = probs_conditioned.view(-1).cpu().numpy()
        probs_uncond_flat = probs_unconditioned.view(-1).cpu().numpy()
        
        if method == "kl":
            # KL divergence from conditioned to unconditioned
            effect = compute_kl_divergence(probs_cond_flat, probs_uncond_flat)
        elif method == "l2":
            # L2 norm of probability difference
            effect = np.linalg.norm(probs_cond_flat - probs_uncond_flat)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return effect
    
    def run_cfg_analysis(
        self,
        image: Union[PILImage.Image, np.ndarray],
        instruction: str,
        cfg_scales: Optional[List[float]] = None,
        unnorm_key: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Run CFG analysis across different scale values.
        
        Useful for tuning the CFG scale parameter.
        
        Args:
            image: Input image
            instruction: Task instruction
            cfg_scales: List of CFG scales to test (default: [0.5, 1.0, 1.5, 2.0, 3.0])
            unnorm_key: Dataset key
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with analysis results for each scale
        """
        if cfg_scales is None:
            cfg_scales = [0.5, 1.0, 1.5, 2.0, 3.0]
        
        results = {}
        
        # Get base logits once
        logits_conditioned, actions_baseline = self._get_logits_and_actions(
            image=image,
            instruction=instruction,
            unnorm_key=unnorm_key,
            **kwargs
        )
        
        zero_image = create_zero_image(image)
        cf_instruction = "" if self.config.clear_prompt_for_image_cf else instruction
        logits_unconditioned, actions_img0 = self._get_logits_and_actions(
            image=zero_image,
            instruction=cf_instruction,
            unnorm_key=unnorm_key,
            **kwargs
        )
        
        # Compute base effect
        base_effect = self._compute_effect_from_logits(logits_conditioned, logits_unconditioned)
        
        original_scale = self.config.cfg_scale
        
        for scale in cfg_scales:
            self.config.cfg_scale = scale
            
            logits_reweighted = apply_cfg_reweighting(
                logits_conditioned=logits_conditioned,
                logits_unconditioned=logits_unconditioned,
                cfg_scale=scale,
            )
            
            actions = self._sample_from_logits(logits_reweighted, unnorm_key)
            
            # Compute effect at this scale
            effect = self._compute_effect_from_logits(logits_conditioned, logits_reweighted)
            
            results[scale] = {
                "actions": actions,
                "effect": effect,
                "delta_from_baseline": np.linalg.norm(actions - actions_baseline),
            }
        
        # Restore original scale
        self.config.cfg_scale = original_scale
        
        # Add baseline info
        results["baseline"] = {
            "actions": actions_baseline,
            "effect": base_effect,
        }
        results["unconditioned"] = {
            "actions": actions_img0,
        }
        
        return results
    
    def visualize_prob_distribution(
        self,
        logits: torch.Tensor,
        title: str = "Probability Distribution",
        save_path: Optional[str] = None,
        top_k: int = 20
    ) -> np.ndarray:
        """
        Visualize the probability distribution from logits.
        
        Args:
            logits: Logits tensor
            title: Plot title
            save_path: Optional path to save visualization
            top_k: Number of top tokens to show
        
        Returns:
            Probability array
        """
        probs = logits_to_probs(logits)
        
        # Get top-k probabilities
        probs_flat = probs.view(-1).cpu().numpy()
        top_indices = np.argsort(probs_flat)[-top_k:]
        top_probs = probs_flat[top_indices]
        
        if save_path:
            try:
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(range(top_k), top_probs)
                ax.set_xlabel('Token Index (sorted by probability)')
                ax.set_ylabel('Probability')
                ax.set_title(title)
                ax.set_ylim(0, 1)
                
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
            except ImportError:
                print("matplotlib not available for visualization")
        
        return probs_flat