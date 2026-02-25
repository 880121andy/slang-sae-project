#!/usr/bin/env python3
"""
Semantic Shift Slang Processing in LLMs - TOKEN-LEVEL SAE Feature Analysis

ACL 2025 Experiment - SAE Feature Analysis

This script analyzes how Semantic Shift slang (e.g., "cook", "slay", "vibe")
is processed across all 48 layers in Gemma-3-12B using Gemma Scope 2 SAEs.

KEY CHANGE: This version extracts features at the SPECIFIC TOKEN POSITION
where slang/literal words differ, NOT the last token of the entire sentence.

Hypothesis: Semantic shift slang requires inhibiting the dominant literal meaning,
so it should be processed primarily in deep layers (>66% depth).
"""

# =============================================================================
# 1. IMPORTS AND SETUP
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend if no display available
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    print("Warning: HF_TOKEN not found in .env file. Gemma models require authentication.")
    print("Create a .env file with: HF_TOKEN=your_huggingface_token")

if os.environ.get('DISPLAY') is None and os.name != 'nt':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm.auto import tqdm
from difflib import SequenceMatcher
import json
import gc
import ast
import re
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# =============================================================================
# 2. DATA STRUCTURES
# =============================================================================

@dataclass
class SlangPair:
    """A slang sentence and its literal equivalent"""
    slang_text: str
    literal_text: str


@dataclass
class SlangPairWithPosition:
    """A slang sentence and its literal equivalent with token positions"""
    slang_text: str
    literal_text: str
    slang_positions: List[Tuple[int, int]]  # List of (start, end) word positions
    literal_positions: List[Tuple[int, int]]
    slang_segments: str  # The actual slang words/phrases
    literal_segments: str  # The literal equivalents


# =============================================================================
# 3. DATASET LOADERS
# =============================================================================

def load_semantic_shift_pairs(df: pd.DataFrame) -> List[SlangPairWithPosition]:
    """Load semantic shift dataset with position tags"""
    pairs = []
    for _, row in df.iterrows():
        try:
            slang_pos = ast.literal_eval(row['slang_positions'])
            literal_pos = ast.literal_eval(row['literal_positions'])

            pairs.append(SlangPairWithPosition(
                slang_text=str(row['gen_z']),
                literal_text=str(row['normal']),
                slang_positions=slang_pos,
                literal_positions=literal_pos,
                slang_segments=str(row.get('slang_segments', '')),
                literal_segments=str(row.get('literal_segments', ''))
            ))
        except Exception as e:
            continue
    return pairs


def load_neologism_pairs(df: pd.DataFrame) -> List[SlangPairWithPosition]:
    """Load neologism dataset with position tags"""
    pairs = []
    for _, row in df.iterrows():
        try:
            slang_pos = ast.literal_eval(row['slang_positions'])
            literal_pos = ast.literal_eval(row['literal_positions'])

            pairs.append(SlangPairWithPosition(
                slang_text=str(row['gen_z']),
                literal_text=str(row['normal']),
                slang_positions=slang_pos,
                literal_positions=literal_pos,
                slang_segments=str(row.get('slang_word', row.get('term', ''))),
                literal_segments=str(row.get('literal_replacement', ''))
            ))
        except Exception as e:
            continue
    return pairs


def load_construction_pairs(df: pd.DataFrame) -> List[SlangPairWithPosition]:
    """Load constructions dataset with position tags"""
    pairs = []
    for _, row in df.iterrows():
        try:
            slang_pos = ast.literal_eval(row['slang_positions'])
            literal_pos = ast.literal_eval(row['literal_positions'])

            pairs.append(SlangPairWithPosition(
                slang_text=str(row['gen_z']),
                literal_text=str(row['normal']),
                slang_positions=slang_pos,
                literal_positions=literal_pos,
                slang_segments=str(row.get('slang_in_sentence', row.get('construction', ''))),
                literal_segments=str(row.get('literal_replacement', ''))
            ))
        except Exception as e:
            continue
    return pairs


def load_simple_pairs(df: pd.DataFrame) -> List[SlangPair]:
    """Load simple pairs without position tags"""
    pairs = [
        SlangPair(slang_text=row['gen_z'], literal_text=row['normal'])
        for _, row in df.iterrows()
    ]
    return pairs


# =============================================================================
# 4. MODEL LOADING
# =============================================================================

def load_model(model_id: str = "google/gemma-3-12b-it"):
    """Load the language model and tokenizer"""
    print(f"Loading {model_id} in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        token=HF_TOKEN,
        low_cpu_mem_usage=True,  # ADD THIS LINE
    )
    model.eval()

    # Get model config - Gemma 3 uses nested config
    if hasattr(model.config, 'text_config'):
        num_layers = model.config.text_config.num_hidden_layers
        hidden_size = model.config.text_config.hidden_size
    else:
        num_layers = len(model.model.language_model.layers)
        hidden_size = model.model.language_model.layers[0].self_attn.q_proj.in_features

    print(f"\nModel loaded!")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden size: {hidden_size}")
    if torch.cuda.is_available():
        print(f"  GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return model, tokenizer, num_layers, hidden_size


# =============================================================================
# 5. ACTIVATION CACHE
# =============================================================================

class ActivationCache:
    """Captures layer activations via forward hooks"""

    def __init__(self, model):
        self.model = model
        self.activations: Dict[int, torch.Tensor] = {}
        self.hooks: List = []

    def _hook(self, layer_idx: int):
        def fn(module, input, output):
            if isinstance(output, tuple):
                self.activations[layer_idx] = output[0].detach()
            else:
                self.activations[layer_idx] = output.detach()
        return fn

    def _get_layers(self):
        """Get transformer layers - Gemma 3 specific path"""
        return self.model.model.language_model.layers

    def register(self, layers: Optional[List[int]] = None):
        self.clear()
        model_layers = self._get_layers()
        layers = layers or list(range(len(model_layers)))

        for idx in layers:
            hook = model_layers[idx].register_forward_hook(self._hook(idx))
            self.hooks.append(hook)
        return self

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.activations = {}

    def get(self) -> Dict[int, torch.Tensor]:
        return self.activations


# =============================================================================
# 6. SAE LOADING UTILITIES
# =============================================================================

def load_sae(layer: int, width: str = "16k", l0: str = "small") -> Optional[SAE]:
    """
    Load SAE for a specific layer using sae_lens.

    Args:
        layer: Layer index (0 to NUM_LAYERS-1)
        width: '16k' or '262k'
        l0: 'small' (~10) or 'big' (~60)
    """
    try:
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release="gemma-scope-2-12b-it-res-all",
            sae_id=f"layer_{layer}_width_{width}_l0_{l0}",
        )
        return sae.to(device)
    except Exception as e:
        print(f"Could not load SAE for layer {layer}: {e}")
        return None


def load_all_saes(num_layers: int, width: str = "262k", l0: str = "small") -> Dict[int, SAE]:
    """Load SAEs for all layers."""
    saes = {}
    print(f"Loading SAEs for {num_layers} layers...")
    print(f"  Width: {width}, L0: {l0}")

    for layer in tqdm(range(num_layers), desc="Loading SAEs"):
        sae = load_sae(layer, width, l0)
        if sae is not None:
            saes[layer] = sae

    print(f"\nLoaded {len(saes)}/{num_layers} SAEs")
    if saes:
        sample_sae = saes[list(saes.keys())[0]]
        print(f"  d_in: {sample_sae.cfg.d_in}")
        print(f"  d_sae: {sample_sae.cfg.d_sae}")

    return saes


# =============================================================================
# 7. TOKEN POSITION MAPPING
# =============================================================================

def word_to_token_position(text: str, word_positions: List[Tuple[int, int]], tokenizer) -> List[int]:
    """
    Robustly maps word positions (word indices) to token indices using character offsets.
    Handles punctuation and spacing correctly.
    """
    words = text.split()
    word_spans = []
    current_char = 0
    for word in words:
        start = text.find(word, current_char)
        if start == -1:
            start = current_char
        end = start + len(word)
        word_spans.append((start, end))
        current_char = end

    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoding['offset_mapping']

    target_token_indices = set()

    for start_word_idx, end_word_idx in word_positions:
        if start_word_idx >= len(word_spans):
            continue

        valid_end_idx = min(end_word_idx, len(word_spans))
        if start_word_idx >= valid_end_idx:
            continue

        char_start = word_spans[start_word_idx][0]
        char_end = word_spans[valid_end_idx - 1][1]

        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == 0 and tok_end == 0:
                continue

            if (tok_start >= char_start and tok_start < char_end) or \
               (tok_end > char_start and tok_end <= char_end):
                target_token_indices.add(i)

    return sorted(list(target_token_indices))


def get_first_diff_token_position(text: str, word_positions: List[Tuple[int, int]], tokenizer) -> int:
    """
    Get the token position of the FIRST differing word.
    Falls back to middle of sentence if positions are invalid.
    """
    if not word_positions:
        tokens = tokenizer(text, return_tensors="pt")['input_ids']
        return tokens.shape[1] // 2

    token_indices = word_to_token_position(text, word_positions[:1], tokenizer)

    if token_indices:
        return min(token_indices)
    else:
        tokens = tokenizer(text, return_tensors="pt")['input_ids']
        return tokens.shape[1] // 2


# =============================================================================
# 8. SAE FEATURE EXTRACTION
# =============================================================================

def extract_sae_features(
    text: str,
    saes: Dict[int, SAE],
    model,
    tokenizer,
    pooling: str = 'last'
) -> Dict[int, torch.Tensor]:
    """
    Extract SAE sparse features for given text at all layers.

    Args:
        text: Input text
        saes: Dict of layer -> SAE
        model: The language model
        tokenizer: The tokenizer
        pooling: 'last', 'mean', or 'max'

    Returns:
        Dict mapping layer_idx -> sparse feature vector
    """
    layers = list(saes.keys())

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    cache = ActivationCache(model)
    cache.register(layers)

    with torch.no_grad():
        model(**inputs)

    result = {}
    for layer_idx, acts in cache.get().items():
        acts = acts.squeeze(0)

        if pooling == 'last':
            pooled = acts[-1]
        elif pooling == 'mean':
            pooled = acts.mean(dim=0)
        else:
            pooled = acts.max(dim=0)[0]

        sae = saes[layer_idx]
        features = sae.encode(pooled.unsqueeze(0).to(sae.W_enc.dtype))
        result[layer_idx] = features.squeeze(0).cpu()

    cache.clear()
    return result


# =============================================================================
# 9. TOKEN-LEVEL ANALYSIS (SINGLE LAYER)
# =============================================================================

def analyze_pairs_single_layer_token_level(
    pairs: List[SlangPairWithPosition],
    layer: int,
    model,
    tokenizer,
    width: str = "262k",
    l0: str = "big",
    max_pairs: Optional[int] = None,
    pooling: str = "first_diff"
) -> pd.DataFrame:
    """
    Analyze all pairs for a single layer using TOKEN-LEVEL positions.

    Args:
        pairs: List of SlangPairWithPosition
        layer: Layer to analyze
        model: The language model
        tokenizer: The tokenizer
        width: SAE width ("16k" or "262k")
        l0: SAE sparsity ("small" or "big")
        max_pairs: Maximum pairs to process
        pooling: How to pool activations:
            - "first_diff": Use first differing token position
            - "all_diff": Average over all differing token positions
            - "last": Use last token (original behavior)
    """
    if max_pairs:
        pairs = pairs[:max_pairs]

    print(f"\nLoading SAE for layer {layer}...")
    sae, _, _ = SAE.from_pretrained(
        release="gemma-scope-2-12b-it-res-all",
        sae_id=f"layer_{layer}_width_{width}_l0_{l0}",
    )
    sae = sae.to(device)
    print(f"  Loaded (d_sae={sae.cfg.d_sae})")

    metrics = {
        'l1_dist': [],
        'l2_dist': [],
        'cosine_dist': [],
        'slang_only_features': [],
        'literal_only_features': [],
        'shared_features': [],
        'slang_active': [],
        'literal_active': []
    }

    skipped = 0
    for pair in tqdm(pairs, desc=f"Layer {layer}"):
        try:
            # === SLANG SENTENCE ===
            slang_inputs = tokenizer(pair.slang_text, return_tensors="pt", truncation=True, max_length=512)
            slang_inputs = {k: v.to(device) for k, v in slang_inputs.items()}
            seq_len_slang = slang_inputs['input_ids'].shape[1]

            cache = ActivationCache(model)
            cache.register([layer])
            with torch.no_grad():
                model(**slang_inputs)
            slang_acts_full = cache.get()[layer].squeeze(0)
            cache.clear()

            if pooling == "first_diff":
                slang_pos = get_first_diff_token_position(pair.slang_text, pair.slang_positions, tokenizer)
                slang_pos = min(slang_pos, seq_len_slang - 1)
                slang_acts = slang_acts_full[slang_pos]
            elif pooling == "all_diff":
                token_indices = word_to_token_position(pair.slang_text, pair.slang_positions, tokenizer)
                if token_indices:
                    token_indices = [min(t, seq_len_slang - 1) for t in token_indices]
                    slang_acts = slang_acts_full[token_indices].mean(dim=0)
                else:
                    slang_acts = slang_acts_full[-1]
            else:
                slang_acts = slang_acts_full[-1]

            sf = sae.encode(slang_acts.unsqueeze(0).to(sae.W_enc.dtype)).squeeze(0).cpu()

            # === LITERAL SENTENCE ===
            literal_inputs = tokenizer(pair.literal_text, return_tensors="pt", truncation=True, max_length=512)
            literal_inputs = {k: v.to(device) for k, v in literal_inputs.items()}
            seq_len_literal = literal_inputs['input_ids'].shape[1]

            cache = ActivationCache(model)
            cache.register([layer])
            with torch.no_grad():
                model(**literal_inputs)
            literal_acts_full = cache.get()[layer].squeeze(0)
            cache.clear()

            if pooling == "first_diff":
                literal_pos = get_first_diff_token_position(pair.literal_text, pair.literal_positions, tokenizer)
                literal_pos = min(literal_pos, seq_len_literal - 1)
                literal_acts = literal_acts_full[literal_pos]
            elif pooling == "all_diff":
                token_indices = word_to_token_position(pair.literal_text, pair.literal_positions, tokenizer)
                if token_indices:
                    token_indices = [min(t, seq_len_literal - 1) for t in token_indices]
                    literal_acts = literal_acts_full[token_indices].mean(dim=0)
                else:
                    literal_acts = literal_acts_full[-1]
            else:
                literal_acts = literal_acts_full[-1]

            lf = sae.encode(literal_acts.unsqueeze(0).to(sae.W_enc.dtype)).squeeze(0).cpu()

            # === COMPUTE METRICS ===
            metrics['l1_dist'].append(torch.norm(sf - lf, p=1).item())
            metrics['l2_dist'].append(torch.norm(sf - lf, p=2).item())

            if sf.norm() > 0 and lf.norm() > 0:
                cos = 1 - F.cosine_similarity(sf.unsqueeze(0), lf.unsqueeze(0)).item()
            else:
                cos = 1.0
            metrics['cosine_dist'].append(cos)

            slang_active = sf > 0
            literal_active = lf > 0
            metrics['slang_only_features'].append((slang_active & ~literal_active).sum().item())
            metrics['literal_only_features'].append((~slang_active & literal_active).sum().item())
            metrics['shared_features'].append((slang_active & literal_active).sum().item())
            metrics['slang_active'].append(slang_active.sum().item())
            metrics['literal_active'].append(literal_active.sum().item())

        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"Error: {e}")
            continue

    del sae
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  Layer {layer} complete, SAE freed (skipped {skipped} pairs)")

    rows = []
    for metric_name, values in metrics.items():
        if len(values) > 0:
            rows.append({
                'layer': layer,
                'metric': metric_name,
                'mean': np.mean(values),
                'std': np.std(values),
                'n': len(values)
            })

    return pd.DataFrame(rows)


# =============================================================================
# 10. FULL LAYER ANALYSIS
# =============================================================================

def analyze_all_layers_sequential_token_level(
    pairs: List[SlangPairWithPosition],
    model,
    tokenizer,
    num_layers: int = 48,
    start_layer: int = 0,
    width: str = "16k",
    l0: str = "big",
    max_pairs: Optional[int] = None,
    pooling: str = "first_diff",
    checkpoint_dir: str = "."
) -> pd.DataFrame:
    """Process all layers one at a time using token-level positions."""

    all_results = []

    for layer in range(start_layer, num_layers):
        layer_df = analyze_pairs_single_layer_token_level(
            pairs, layer, model, tokenizer,
            width=width, l0=l0, max_pairs=max_pairs, pooling=pooling
        )
        all_results.append(layer_df)

        if (layer + 1) % 10 == 0:
            checkpoint_df = pd.concat(all_results, ignore_index=True)
            checkpoint_path = f'{checkpoint_dir}/checkpoint_token_level_layer_{layer+1}.csv'
            checkpoint_df.to_csv(checkpoint_path, index=False)
            print(f"  Checkpoint saved: {checkpoint_path}")

    return pd.concat(all_results, ignore_index=True)


def analyze_pairs_single_layer(
    pairs: List[SlangPair],
    layer: int,
    model,
    tokenizer,
    width: str = "262k",
    l0: str = "small",
    max_pairs: Optional[int] = None
) -> pd.DataFrame:
    """Analyze all pairs for a single layer (last token), then free SAE memory."""

    if max_pairs:
        pairs = pairs[:max_pairs]

    print(f"\nLoading SAE for layer {layer}...")
    sae, _, _ = SAE.from_pretrained(
        release="gemma-scope-2-12b-it-res-all",
        sae_id=f"layer_{layer}_width_{width}_l0_{l0}",
    )
    sae = sae.to(device)
    print(f"  Loaded (d_sae={sae.cfg.d_sae})")

    metrics = {
        'l1_dist': [],
        'l2_dist': [],
        'cosine_dist': [],
        'slang_only_features': [],
        'literal_only_features': [],
        'shared_features': [],
        'slang_active': [],
        'literal_active': []
    }

    for pair in tqdm(pairs, desc=f"Layer {layer}"):
        try:
            inputs = tokenizer(pair.slang_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            cache = ActivationCache(model)
            cache.register([layer])
            with torch.no_grad():
                model(**inputs)
            slang_acts = cache.get()[layer].squeeze(0)[-1]
            sf = sae.encode(slang_acts.unsqueeze(0).to(sae.W_enc.dtype)).squeeze(0).cpu()
            cache.clear()

            inputs = tokenizer(pair.literal_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            cache = ActivationCache(model)
            cache.register([layer])
            with torch.no_grad():
                model(**inputs)
            literal_acts = cache.get()[layer].squeeze(0)[-1]
            lf = sae.encode(literal_acts.unsqueeze(0).to(sae.W_enc.dtype)).squeeze(0).cpu()
            cache.clear()

            metrics['l1_dist'].append(torch.norm(sf - lf, p=1).item())
            metrics['l2_dist'].append(torch.norm(sf - lf, p=2).item())

            if sf.norm() > 0 and lf.norm() > 0:
                cos = 1 - F.cosine_similarity(sf.unsqueeze(0), lf.unsqueeze(0)).item()
            else:
                cos = 1.0
            metrics['cosine_dist'].append(cos)

            slang_active = sf > 0
            literal_active = lf > 0
            metrics['slang_only_features'].append((slang_active & ~literal_active).sum().item())
            metrics['literal_only_features'].append((~slang_active & literal_active).sum().item())
            metrics['shared_features'].append((slang_active & literal_active).sum().item())
            metrics['slang_active'].append(slang_active.sum().item())
            metrics['literal_active'].append(literal_active.sum().item())

        except Exception as e:
            print(f"Error: {e}")
            continue

    del sae
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  Layer {layer} complete, SAE freed")

    rows = []
    for metric_name, values in metrics.items():
        if len(values) > 0:
            rows.append({
                'layer': layer,
                'metric': metric_name,
                'mean': np.mean(values),
                'std': np.std(values),
                'n': len(values)
            })

    return pd.DataFrame(rows)


def analyze_all_layers_sequential(
    pairs: List[SlangPair],
    model,
    tokenizer,
    num_layers: int = 48,
    start_layer: int = 0,
    width: str = "262k",
    l0: str = "small",
    max_pairs: Optional[int] = None,
    checkpoint_dir: str = "."
) -> pd.DataFrame:
    """Process all layers one at a time to minimize memory."""

    all_results = []

    for layer in range(start_layer, num_layers):
        layer_df = analyze_pairs_single_layer(
            pairs, layer, model, tokenizer,
            width=width, l0=l0, max_pairs=max_pairs
        )
        all_results.append(layer_df)

        if (layer + 1) % 10 == 0:
            checkpoint_df = pd.concat(all_results, ignore_index=True)
            checkpoint_path = f'{checkpoint_dir}/checkpoint_layer_{layer+1}.csv'
            checkpoint_df.to_csv(checkpoint_path, index=False)
            print(f"  Checkpoint saved: {checkpoint_path}")

    return pd.concat(all_results, ignore_index=True)


# =============================================================================
# 11. VISUALIZATION
# =============================================================================

def plot_layer_differences(df: pd.DataFrame, metric: str = 'l2_dist', title_prefix: str = "", save_path: str = None):
    """Plot SAE feature differences across all layers"""

    metric_df = df[df['metric'] == metric].sort_values('layer')

    if len(metric_df) == 0:
        print(f"No data for metric: {metric}")
        return None

    num_layers = int(metric_df['layer'].max()) + 1

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.fill_between(
        metric_df['layer'],
        metric_df['mean'] - metric_df['std'],
        metric_df['mean'] + metric_df['std'],
        alpha=0.3, color='#e74c3c'
    )
    ax.plot(
        metric_df['layer'], metric_df['mean'],
        'o-', color='#e74c3c', linewidth=2, markersize=5
    )

    peak_idx = metric_df['mean'].idxmax()
    peak = metric_df.loc[peak_idx]
    ax.axvline(x=peak['layer'], color='gray', linestyle='--', alpha=0.7)
    ax.annotate(
        f"Peak: Layer {int(peak['layer'])} ({100*peak['layer']/(num_layers-1):.0f}% depth)",
        xy=(peak['layer'], peak['mean']),
        xytext=(15, 15), textcoords='offset points',
        fontsize=11, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='gray')
    )

    ax.axvspan(0, num_layers*0.33, alpha=0.08, color='blue', label='Early (0-33%)')
    ax.axvspan(num_layers*0.33, num_layers*0.66, alpha=0.08, color='green', label='Middle (33-66%)')
    ax.axvspan(num_layers*0.66, num_layers, alpha=0.08, color='red', label='Deep (66-100%)')

    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=14)
    ax.set_title(
        f'{title_prefix}TOKEN-LEVEL SAE Feature Differences\n'
        f'Comparing activations at SPECIFIC differing token positions',
        fontsize=16, fontweight='bold'
    )
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, num_layers)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    try:
        plt.show()
    except Exception:
        pass  # Ignore display errors in headless environments
    plt.close(fig)

    return peak


def plot_all_metrics(df: pd.DataFrame, save_path: str = None):
    """Plot multiple metrics side by side"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    metrics = ['l2_dist', 'cosine_dist', 'slang_only_features', 'literal_only_features']
    titles = ['L2 Distance', 'Cosine Distance', 'Slang-Only Features', 'Literal-Only Features']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    for ax, metric, title, color in zip(axes.flat, metrics, titles, colors):
        metric_df = df[df['metric'] == metric].sort_values('layer')
        if len(metric_df) == 0:
            ax.set_title(f"{title}\nNo data", fontsize=12)
            continue

        num_layers = int(metric_df['layer'].max()) + 1

        ax.fill_between(
            metric_df['layer'],
            metric_df['mean'] - metric_df['std'],
            metric_df['mean'] + metric_df['std'],
            alpha=0.3, color=color
        )
        ax.plot(
            metric_df['layer'], metric_df['mean'],
            'o-', color=color, linewidth=2, markersize=3
        )

        peak_idx = metric_df['mean'].idxmax()
        peak = metric_df.loc[peak_idx]
        ax.axvline(x=peak['layer'], color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f"{title}\nPeak: Layer {int(peak['layer'])} ({100*peak['layer']/(num_layers-1):.0f}%)",
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Layer')
        ax.grid(True, alpha=0.3)

    plt.suptitle('TOKEN-LEVEL SAE Feature Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    try:
        plt.show()
    except Exception:
        pass  # Ignore display errors in headless environments
    plt.close(fig)


def plot_feature_activation_comparison(df: pd.DataFrame, save_path: str = None):
    """Compare feature activation patterns between slang and literal"""

    fig, ax = plt.subplots(figsize=(16, 6))

    slang_df = df[df['metric'] == 'slang_active'].sort_values('layer')
    literal_df = df[df['metric'] == 'literal_active'].sort_values('layer')
    shared_df = df[df['metric'] == 'shared_features'].sort_values('layer')

    if len(slang_df) == 0:
        print("No activation data available")
        return

    ax.plot(slang_df['layer'], slang_df['mean'], 'o-', label='Slang Active', linewidth=2, markersize=4)
    ax.plot(literal_df['layer'], literal_df['mean'], 's-', label='Literal Active', linewidth=2, markersize=4)
    ax.plot(shared_df['layer'], shared_df['mean'], '^-', label='Shared', linewidth=2, markersize=4)

    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('Mean # Active Features', fontsize=14)
    ax.set_title('Feature Activation Patterns: Slang vs Literal\nacross Layers', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    try:
        plt.show()
    except Exception:
        pass  # Ignore display errors in headless environments
    plt.close(fig)


# =============================================================================
# 12. STATISTICAL ANALYSIS
# =============================================================================

def analyze_results(df: pd.DataFrame, metric: str = 'l2_dist'):
    """Statistical analysis of layer-wise differences"""

    metric_df = df[df['metric'] == metric].sort_values('layer')

    if len(metric_df) == 0:
        print(f"No data for metric: {metric}")
        return None

    num_layers = int(metric_df['layer'].max()) + 1

    early_end = int(num_layers * 0.33)
    middle_end = int(num_layers * 0.66)

    early = list(range(0, early_end))
    middle = list(range(early_end, middle_end))
    deep = list(range(middle_end, num_layers))

    print("=" * 70)
    print("TOKEN-LEVEL SAE FEATURE ANALYSIS RESULTS")
    print("=" * 70)
    print(f"\nModel: Gemma-3-12B-IT")
    print(f"SAE: Gemma Scope 2")
    print(f"Dataset: {int(metric_df['n'].iloc[0])} pairs")
    print(f"Layers analyzed: {num_layers}")
    print(f"\n*** Using TOKEN-LEVEL positions (not sentence-level) ***")

    peak_idx = metric_df['mean'].idxmax()
    peak = metric_df.loc[peak_idx]
    peak_layer = int(peak['layer'])
    peak_pct = peak_layer / (num_layers - 1) * 100

    print(f"\n{'-' * 70}")
    print(f"PEAK ANALYSIS")
    print(f"{'-' * 70}")
    print(f"Peak Layer: {peak_layer} / {num_layers-1} ({peak_pct:.1f}% depth)")
    print(f"Peak Value: {peak['mean']:.4f} +/- {peak['std']:.4f}")

    top5 = metric_df.nlargest(5, 'mean')
    print(f"\nTop 5 Layers: {top5['layer'].astype(int).tolist()}")

    early_df = metric_df[metric_df['layer'].isin(early)]
    middle_df = metric_df[metric_df['layer'].isin(middle)]
    deep_df = metric_df[metric_df['layer'].isin(deep)]

    early_mean = early_df['mean'].mean() if len(early_df) > 0 else np.nan
    middle_mean = middle_df['mean'].mean() if len(middle_df) > 0 else np.nan
    deep_mean = deep_df['mean'].mean() if len(deep_df) > 0 else np.nan

    print(f"\n{'-' * 70}")
    print(f"REGION ANALYSIS")
    print(f"{'-' * 70}")
    print(f"  Early  (layers 0-{early_end-1}):     {early_mean:.4f}")
    print(f"  Middle (layers {early_end}-{middle_end-1}):   {middle_mean:.4f}")
    print(f"  Deep   (layers {middle_end}-{num_layers-1}):   {deep_mean:.4f}")

    region_means = {'Early': early_mean, 'Middle': middle_mean, 'Deep': deep_mean}
    max_region = max(region_means, key=region_means.get)
    print(f"\n  -> Highest mean in: {max_region} layers")

    early_vals = early_df['mean'].values
    middle_vals = middle_df['mean'].values
    deep_vals = deep_df['mean'].values

    results_dict = {
        'peak_layer': int(peak_layer),
        'peak_pct': float(peak_pct),
        'peak_value': float(peak['mean']),
        'early_mean': float(early_mean) if not np.isnan(early_mean) else None,
        'middle_mean': float(middle_mean) if not np.isnan(middle_mean) else None,
        'deep_mean': float(deep_mean) if not np.isnan(deep_mean) else None,
    }

    if len(early_vals) > 1 and len(middle_vals) > 1 and len(deep_vals) > 1:
        f_stat, p_val = stats.f_oneway(early_vals, middle_vals, deep_vals)
        print(f"\n{'-' * 70}")
        print(f"STATISTICAL TESTS")
        print(f"{'-' * 70}")
        print(f"ANOVA (3 regions): F={f_stat:.2f}, p={p_val:.6f}")
        if p_val < 0.05:
            print("  -> Significant difference between regions!")

        results_dict['anova_f'] = float(f_stat)
        results_dict['anova_p'] = float(p_val)

        t_em, p_em = stats.ttest_ind(early_vals, middle_vals)
        t_md, p_md = stats.ttest_ind(middle_vals, deep_vals)
        t_ed, p_ed = stats.ttest_ind(early_vals, deep_vals)

        print(f"\nPairwise t-tests:")
        print(f"  Early vs Middle:  t={t_em:.2f}, p={p_em:.4f} {'*' if p_em < 0.05 else ''}")
        print(f"  Middle vs Deep:   t={t_md:.2f}, p={p_md:.4f} {'*' if p_md < 0.05 else ''}")
        print(f"  Early vs Deep:    t={t_ed:.2f}, p={p_ed:.4f} {'*' if p_ed < 0.05 else ''}")

    print(f"\n{'=' * 70}")
    print("HYPOTHESIS VALIDATION")
    print("=" * 70)
    print(f"\nHypothesis: Semantic shift processed in DEEP layers (>66% depth)")
    print(f"Result: Peak at {peak_pct:.1f}% depth (Layer {peak_layer})")

    if peak_pct > 66:
        verdict = "SUPPORTED"
        explanation = "Peak in deep layers as predicted!"
    elif peak_pct > 50:
        verdict = "PARTIAL"
        explanation = "Peak in middle-to-deep region"
    elif peak_pct > 33:
        verdict = "PARTIAL"
        explanation = "Peak in middle layers"
    else:
        verdict = "NOT SUPPORTED"
        explanation = "Peak in early layers"

    print(f"\n{verdict}: {explanation}")
    results_dict['verdict'] = verdict

    return results_dict


# =============================================================================
# 13. VERIFICATION UTILITIES
# =============================================================================

def quick_test_token_positions(pairs: List[SlangPairWithPosition], tokenizer, n: int = 5):
    """Test that token positions are being extracted correctly."""
    print("=" * 70)
    print("TOKEN POSITION VERIFICATION (MULTI-TOKEN CHECK)")
    print("=" * 70)

    for i, pair in enumerate(pairs[:n]):
        print(f"\n--- Example {i+1} ---")
        print(f"Slang: {pair.slang_text}")
        print(f"Literal: {pair.literal_text}")

        print(f"Slang segments: {pair.slang_segments}")

        slang_indices = word_to_token_position(pair.slang_text, pair.slang_positions, tokenizer)

        s_ids = tokenizer(pair.slang_text, return_tensors="pt")['input_ids'][0]
        valid_s_indices = [idx for idx in slang_indices if idx < len(s_ids)]
        slang_tokens = tokenizer.convert_ids_to_tokens(s_ids[valid_s_indices])

        print(f"Slang tokens:   {slang_tokens}")
        print(f"   (Indices: {slang_indices})")

        print(f"Literal segments: {pair.literal_segments}")

        literal_indices = word_to_token_position(pair.literal_text, pair.literal_positions, tokenizer)

        l_ids = tokenizer(pair.literal_text, return_tensors="pt")['input_ids'][0]
        valid_l_indices = [idx for idx in literal_indices if idx < len(l_ids)]
        literal_tokens = tokenizer.convert_ids_to_tokens(l_ids[valid_l_indices])

        print(f"Literal tokens: {literal_tokens}")
        print(f"   (Indices: {literal_indices})")


# =============================================================================
# 14. ACTIVATION STEERING
# =============================================================================

class ActivationSteering:
    """Steer model by adding direction vector at specific layer"""

    def __init__(self, model, tokenizer, saes: Dict[int, SAE]):
        self.model = model
        self.tokenizer = tokenizer
        self.saes = saes
        self.steering_vector: Optional[torch.Tensor] = None
        self.target_layer: Optional[int] = None
        self.strength: float = 1.0
        self.hook = None

    def compute_steering_vector(
        self,
        pairs: List[SlangPair],
        layer: int,
        n_samples: int = 50
    ):
        """Compute steering direction from SAE feature differences"""
        slang_feats = []
        literal_feats = []

        for pair in tqdm(pairs[:n_samples], desc="Computing steering vector"):
            sf = extract_sae_features(pair.slang_text, self.saes, self.model, self.tokenizer)[layer]
            lf = extract_sae_features(pair.literal_text, self.saes, self.model, self.tokenizer)[layer]
            slang_feats.append(sf)
            literal_feats.append(lf)

        mean_slang = torch.stack(slang_feats).mean(dim=0)
        mean_literal = torch.stack(literal_feats).mean(dim=0)
        feature_diff = mean_slang - mean_literal

        sae = self.saes[layer]
        self.steering_vector = sae.decode(feature_diff.unsqueeze(0).to(device)).squeeze(0)
        self.target_layer = layer

        print(f"Steering vector computed for layer {layer}")
        print(f"  Norm: {self.steering_vector.norm().item():.4f}")

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            h[:, -1, :] += self.strength * self.steering_vector.to(h.dtype)
            return (h,) + output[1:]
        else:
            output[:, -1, :] += self.strength * self.steering_vector.to(output.dtype)
            return output

    def generate(self, prompt: str, strength: float = 1.0, max_tokens: int = 50) -> str:
        """Generate with steering applied"""
        self.strength = strength

        layers = self.model.model.language_model.layers
        self.hook = layers[self.target_layer].register_forward_hook(self._hook_fn)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )

        self.hook.remove()
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# =============================================================================
# 15. MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""

    # Load model
    model, tokenizer, NUM_LAYERS, HIDDEN_SIZE = load_model()

    # Load dataset - choose one:
    # For semantic shift:
    df = pd.read_csv('genz_dataset_tagged.csv')
    pairs = load_semantic_shift_pairs(df)

    # For neologism:
    # df = pd.read_csv('neologism_tagged.csv')
    # pairs = load_neologism_pairs(df)

    # For constructions:
    # df = pd.read_csv('constructions_tagged.csv')
    # pairs = load_construction_pairs(df)

    print(f"Loaded {len(pairs)} pairs")

    # Verify positions
    quick_test_token_positions(pairs, tokenizer, n=5)

    # Run token-level analysis
    print("\nStarting sequential layer analysis...")
    print(f"Processing {len(pairs)} pairs across {NUM_LAYERS} layers\n")

    results_df = analyze_all_layers_sequential_token_level(
        pairs,
        model,
        tokenizer,
        num_layers=NUM_LAYERS,
        start_layer=20,
        width="16k",
        l0="big",
        max_pairs=None,
        pooling="all_diff"
    )

    print(f"\nAll layers complete!")
    print(f"Results shape: {results_df.shape}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f'sae_token_level_results_{timestamp}.csv', index=False)

    # Visualize
    peak = plot_layer_differences(results_df, 'l2_dist', save_path='sae_token_level_differences.png')
    plot_all_metrics(results_df, save_path='sae_token_level_all_metrics.png')
    plot_feature_activation_comparison(results_df, save_path='feature_activation_comparison.png')

    # Statistical analysis
    stats_results = analyze_results(results_df, 'l2_dist')

    # Save analysis
    with open(f'analysis_{timestamp}.json', 'w') as f:
        json.dump(stats_results, f, indent=2)

    print("\nSaved files:")
    print(f"  - sae_token_level_results_{timestamp}.csv")
    print(f"  - analysis_{timestamp}.json")
    print(f"  - sae_token_level_differences.png")
    print(f"  - sae_token_level_all_metrics.png")
    print(f"  - feature_activation_comparison.png")

    return results_df, stats_results


if __name__ == "__main__":
    main()
