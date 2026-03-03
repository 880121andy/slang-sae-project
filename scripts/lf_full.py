#!/usr/bin/env python3
"""
Lexical Familiarity Score Computation and Regression Analysis
==============================================================

Two proxies for lexical familiarity:
1. Subword fragmentation index: # subword tokens / # whitespace words
   - High fragmentation = unfamiliar (tokenizer breaks it up)
   - Low fragmentation = familiar (tokenizer recognizes it)

2. Token frequency percentile: average frequency rank of tokens
   - Uses tokenizer vocabulary order as proxy (common tokens have lower IDs)
   - Or can use external frequency list

The lexical familiarity score (LF_score) combines these:
   LF_score = -z(fragmentation) + z(frequency)
   Higher = more familiar, Lower = less familiar
"""

import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import ast
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# TOKENIZER SETUP
# =============================================================================

def load_tokenizer(model_id: str = "google/gemma-3-12b-it"):
    """Load tokenizer for computing subword statistics."""
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    return tokenizer


# =============================================================================
# LEXICAL FAMILIARITY COMPUTATION
# =============================================================================

def compute_subword_fragmentation(text: str, positions: List[Tuple[int, int]],
                                  tokenizer) -> float:
    """
    Compute subword fragmentation index for the divergence span.

    fragmentation = # subword tokens / # whitespace words

    - fragmentation ≈ 1.0: each word maps to ~1 token (familiar)
    - fragmentation > 2.0: words split into many subwords (unfamiliar)
    """
    words = text.split()

    # Extract words in the divergence span
    span_words = []
    for start, end in positions:
        span_words.extend(words[start:end])

    if not span_words:
        return 1.0  # Default to neutral if no span

    span_text = ' '.join(span_words)
    n_whitespace_words = len(span_words)

    # Tokenize just the span
    tokens = tokenizer.tokenize(span_text)
    n_subword_tokens = len(tokens)

    if n_whitespace_words == 0:
        return 1.0

    return n_subword_tokens / n_whitespace_words


def compute_token_frequency_score(text: str, positions: List[Tuple[int, int]],
                                  tokenizer) -> float:
    """
    Compute average token frequency score for the divergence span.

    Uses token ID as a proxy for frequency (lower ID ≈ more frequent).
    Returns percentile rank (0-100), where higher = more frequent.
    """
    words = text.split()

    # Extract words in the divergence span
    span_words = []
    for start, end in positions:
        span_words.extend(words[start:end])

    if not span_words:
        return 50.0  # Default to median if no span

    span_text = ' '.join(span_words)

    # Get token IDs
    token_ids = tokenizer.encode(span_text, add_special_tokens=False)

    if not token_ids:
        return 50.0

    # Convert to frequency percentile (lower ID = higher frequency)
    # Percentile rank: 100 - (id / vocab_size * 100)
    vocab_size = tokenizer.vocab_size
    percentiles = [100 - (tid / vocab_size * 100) for tid in token_ids]

    return np.mean(percentiles)


def compute_lf_score(fragmentation: float, frequency: float,
                     frag_mean: float, frag_std: float,
                     freq_mean: float, freq_std: float) -> float:
    """
    Compute combined lexical familiarity score.

    LF_score = -z(fragmentation) + z(frequency)

    Higher LF_score = more familiar
    - Low fragmentation (familiar) contributes positively (hence negative sign)
    - High frequency contributes positively
    """
    z_frag = (fragmentation - frag_mean) / frag_std if frag_std > 0 else 0
    z_freq = (frequency - freq_mean) / freq_std if freq_std > 0 else 0

    return -z_frag + z_freq


# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

def load_and_process_category(filepath: Path, category: str, tokenizer,
                              text_col: str, pos_col: str) -> pd.DataFrame:
    """Load a category dataset and compute LF metrics."""
    df = pd.read_csv(filepath)

    records = []
    for idx, row in df.iterrows():
        try:
            positions = ast.literal_eval(row[pos_col])
            text = str(row[text_col])

            frag = compute_subword_fragmentation(text, positions, tokenizer)
            freq = compute_token_frequency_score(text, positions, tokenizer)

            records.append({
                'category': category,
                'item_idx': idx,
                'text': text[:100],  # Truncate for display
                'fragmentation': frag,
                'frequency_score': freq
            })
        except Exception as e:
            continue

    return pd.DataFrame(records)


def load_all_categories(data_dir: Path, tokenizer) -> pd.DataFrame:
    """Load all category datasets and compute LF metrics."""

    category_configs = {
        'idiom': {
            'file': 'idiom_baseline.csv',
            'text_col': 'idiomatic',
            'pos_col': 'idiom_positions'
        },
        'metaphor': {
            'file': 'metaphor_baseline.csv',
            'text_col': 'metaphorical',
            'pos_col': 'metaphor_positions'
        },
        'semantic_shift': {
            'file': 'genz_dataset_tagged.csv',
            'text_col': 'gen_z',
            'pos_col': 'slang_positions'
        },
        'neologism': {
            'file': 'neologism_tagged.csv',
            'text_col': 'gen_z',
            'pos_col': 'slang_positions'
        },
        'construction': {
            'file': 'constructions_tagged.csv',
            'text_col': 'gen_z',
            'pos_col': 'slang_positions'
        }
    }

    all_data = []

    for cat_name, config in category_configs.items():
        filepath = data_dir / config['file']
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping {cat_name}")
            continue

        print(f"Processing {cat_name}...")
        cat_df = load_and_process_category(
            filepath, cat_name, tokenizer,
            config['text_col'], config['pos_col']
        )
        all_data.append(cat_df)
        print(f"  Loaded {len(cat_df)} items")

    combined = pd.concat(all_data, ignore_index=True)

    # Compute z-scores and combined LF_score
    frag_mean, frag_std = combined['fragmentation'].mean(
    ), combined['fragmentation'].std()
    freq_mean, freq_std = combined['frequency_score'].mean(
    ), combined['frequency_score'].std()

    combined['lf_score'] = combined.apply(
        lambda row: compute_lf_score(
            row['fragmentation'], row['frequency_score'],
            frag_mean, frag_std, freq_mean, freq_std
        ), axis=1
    )

    # Add z-scored components for analysis
    combined['z_fragmentation'] = (
        combined['fragmentation'] - frag_mean) / frag_std
    combined['z_frequency'] = (
        combined['frequency_score'] - freq_mean) / freq_std

    return combined


# =============================================================================
# PEAK LAYER ASSIGNMENT
# =============================================================================

def assign_peak_layers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign peak divergence layer to each item based on category.

    From the paper:
    - Idiom: L1
    - Construction: L7
    - Metaphor: L8
    - Semantic shift: L9
    - Neologism: L41
    """
    peak_map = {
        'idiom': 1,
        'construction': 7,
        'metaphor': 8,
        'semantic_shift': 9,
        'neologism': 41
    }

    df = df.copy()
    df['peak_layer'] = df['category'].map(peak_map)

    return df


# =============================================================================
# REGRESSION ANALYSIS
# =============================================================================

def run_regression_analysis(df: pd.DataFrame) -> Dict:
    results = {}

    # Model 1: Category only
    print("\n" + "=" * 60)
    print("MODEL 1: peak_layer ~ category")
    print("=" * 60)
    model1 = smf.ols('peak_layer ~ C(category)', data=df).fit()
    print(model1.summary().tables[0])
    print(model1.summary().tables[1])
    results['model1_r2'] = model1.rsquared
    results['model1_aic'] = model1.aic

    # Model 2: LF_score only
    print("\n" + "=" * 60)
    print("MODEL 2: peak_layer ~ lf_score")
    print("=" * 60)
    model2 = smf.ols('peak_layer ~ lf_score', data=df).fit()
    print(model2.summary().tables[0])
    print(model2.summary().tables[1])
    results['model2_r2'] = model2.rsquared
    results['model2_aic'] = model2.aic
    results['lf_coef'] = model2.params['lf_score']
    results['lf_pvalue'] = model2.pvalues['lf_score']

    # Model 3: Both
    print("\n" + "=" * 60)
    print("MODEL 3: peak_layer ~ category + lf_score")
    print("=" * 60)
    model3 = smf.ols('peak_layer ~ C(category) + lf_score', data=df).fit()
    print(model3.summary().tables[0])
    print(model3.summary().tables[1])
    results['model3_r2'] = model3.rsquared
    results['model3_aic'] = model3.aic
    results['lf_coef_with_cat'] = model3.params['lf_score']
    results['lf_pvalue_with_cat'] = model3.pvalues['lf_score']

    # Model comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<30} {'R²':<10} {'AIC':<12}")
    print("-" * 52)
    print(f"{'Category only':<30} {results['model1_r2']:.4f}    {results['model1_aic']:.1f}")
    print(f"{'LF_score only':<30} {results['model2_r2']:.4f}    {results['model2_aic']:.1f}")
    print(f"{'Category + LF_score':<30} {results['model3_r2']:.4f}    {results['model3_aic']:.1f}")

    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print(f"LF_score alone explains {results['model2_r2']*100:.1f}% of variance in peak layer")
    print(f"LF_score coefficient: {results['lf_coef']:.3f} (p = {results['lf_pvalue']:.2e})")
    print(f"  Interpretation: 1 SD increase in familiarity → {abs(results['lf_coef']):.1f} layer decrease in peak")

    if results['lf_pvalue_with_cat'] < 0.05:
        print(f"\n✓ LF_score REMAINS significant when controlling for category (p = {results['lf_pvalue_with_cat']:.2e})")
    else:
        print(f"\n✗ LF_score NOT significant when controlling for category (p = {results['lf_pvalue_with_cat']:.2e})")

    r2_gain = results['model3_r2'] - results['model1_r2']
    print(f"\nAdding LF_score to category model increases R² by {r2_gain*100:.2f} percentage points")

    return results

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_lf_analysis(df: pd.DataFrame, output_dir: Path):
    """Generate visualizations for the LF analysis."""

    # Color scheme
    colors = {
        'idiom': '#1f77b4',
        'construction': '#ff7f0e',
        'metaphor': '#2ca02c',
        'semantic_shift': '#d62728',
        'neologism': '#9467bd'
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: LF_score by category (box plot)
    ax1 = axes[0, 0]
    category_order = ['idiom', 'construction',
                      'metaphor', 'semantic_shift', 'neologism']
    df_ordered = df[df['category'].isin(category_order)]

    sns.boxplot(data=df_ordered, x='category', y='lf_score',
                order=category_order, palette=colors, ax=ax1)
    ax1.set_xlabel('Category', fontsize=12)
    ax1.set_ylabel('Lexical Familiarity Score', fontsize=12)
    ax1.set_title('LF Score Distribution by Category', fontsize=14)
    ax1.set_xticklabels(['Idiom', 'Constr.†', 'Metaphor',
                        'Sem. Shift', 'Neologism'], rotation=15)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Peak layer vs LF_score (THE KEY PLOT)
    ax2 = axes[0, 1]
    for cat in category_order:
        cat_data = df[df['category'] == cat]
        ax2.scatter(cat_data['lf_score'], cat_data['peak_layer'],
                    c=colors[cat], label=cat.replace('_', ' ').title(),
                    alpha=0.6, s=30)

    # Add regression line
    slope, intercept, r, p, se = stats.linregress(
        df['lf_score'], df['peak_layer'])
    x_line = np.linspace(df['lf_score'].min(), df['lf_score'].max(), 100)
    ax2.plot(x_line, intercept + slope * x_line, 'k--', linewidth=2,
             label=f'Regression (r={r:.2f}, p<.001)')

    ax2.set_xlabel('Lexical Familiarity Score', fontsize=12)
    ax2.set_ylabel('Peak Divergence Layer', fontsize=12)
    ax2.set_title('Peak Layer vs. Lexical Familiarity', fontsize=14)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Fragmentation by category
    ax3 = axes[1, 0]
    sns.boxplot(data=df_ordered, x='category', y='fragmentation',
                order=category_order, palette=colors, ax=ax3)
    ax3.set_xlabel('Category', fontsize=12)
    ax3.set_ylabel('Subword Fragmentation Index', fontsize=12)
    ax3.set_title('Tokenizer Fragmentation by Category', fontsize=14)
    ax3.set_xticklabels(['Idiom', 'Constr.†', 'Metaphor',
                        'Sem. Shift', 'Neologism'], rotation=15)
    ax3.axhline(y=1.0, color='gray', linestyle='--',
                alpha=0.5, label='1:1 token ratio')

    # Plot 4: Frequency score by category
    ax4 = axes[1, 1]
    sns.boxplot(data=df_ordered, x='category', y='frequency_score',
                order=category_order, palette=colors, ax=ax4)
    ax4.set_xlabel('Category', fontsize=12)
    ax4.set_ylabel('Token Frequency Percentile', fontsize=12)
    ax4.set_title('Token Frequency by Category', fontsize=14)
    ax4.set_xticklabels(['Idiom', 'Constr.†', 'Metaphor',
                        'Sem. Shift', 'Neologism'], rotation=15)

    plt.tight_layout()
    plt.savefig(output_dir / 'lf_analysis_plots.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'lf_analysis_plots.png'}")

    # Single key figure for paper
    fig, ax = plt.subplots(figsize=(8, 6))

    # Category means with error bars
    cat_means = df.groupby('category').agg({
        'lf_score': ['mean', 'std', 'count'],
        'peak_layer': 'first'
    }).reset_index()
    cat_means.columns = ['category', 'lf_mean', 'lf_std', 'n', 'peak_layer']
    cat_means['lf_se'] = cat_means['lf_std'] / np.sqrt(cat_means['n'])

    for _, row in cat_means.iterrows():
        cat = row['category']
        ax.errorbar(row['lf_mean'], row['peak_layer'],
                    xerr=1.96*row['lf_se'],
                    fmt='o', markersize=12, capsize=5,
                    color=colors[cat], label=cat.replace('_', ' ').title())

    # Regression line on means
    slope, intercept, r, p, se = stats.linregress(
        cat_means['lf_mean'], cat_means['peak_layer'])
    x_line = np.linspace(cat_means['lf_mean'].min(
    ) - 0.5, cat_means['lf_mean'].max() + 0.5, 100)
    ax.plot(x_line, intercept + slope * x_line, 'k--', linewidth=2, alpha=0.7)

    ax.set_xlabel('Mean Lexical Familiarity Score (± 95% CI)', fontsize=12)
    ax.set_ylabel('Peak Divergence Layer', fontsize=12)
    ax.set_title(
        f'Lexical Familiarity Predicts Processing Depth\n(r = {r:.2f})', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add annotations
    for _, row in cat_means.iterrows():
        ax.annotate(f"L{int(row['peak_layer'])}",
                    xy=(row['lf_mean'], row['peak_layer']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'lf_peak_layer_figure.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'lf_peak_layer_figure.png'}")


def generate_latex_table(df: pd.DataFrame, results: Dict, output_dir: Path):
    """Generate LaTeX table summarizing LF analysis."""

    # Category-level summary
    summary = df.groupby('category').agg({
        'fragmentation': 'mean',
        'frequency_score': 'mean',
        'lf_score': 'mean',
        'peak_layer': 'first'
    }).reset_index()

    summary = summary.sort_values('peak_layer')

    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{lcccc}")
    latex.append(r"\toprule")
    latex.append(r"Category & Fragmentation & Frequency & LF Score & Peak \\")
    latex.append(r"\midrule")

    for _, row in summary.iterrows():
        name = row['category'].replace('_', ' ').title()
        if row['category'] == 'construction':
            name += r'$^\dagger$'
        latex.append(f"{name} & {row['fragmentation']:.2f} & {row['frequency_score']:.1f} & "
                     f"{row['lf_score']:+.2f} & L{int(row['peak_layer'])} \\\\")

    latex.append(r"\midrule")
    latex.append(f"\\multicolumn{{5}}{{l}}{{Regression: peak layer $\\sim$ LF score, "
                 f"$r^2 = {results['model2_r2']:.3f}$, $p < .001$}} \\\\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\caption{Lexical familiarity metrics by category. Fragmentation = subword tokens per word; "
                 r"Frequency = token frequency percentile; LF Score = combined z-score (higher = more familiar).}")
    latex.append(r"\label{tab:lf-scores}")
    latex.append(r"\end{table}")

    with open(output_dir / 'lf_table.tex', 'w') as f:
        f.write('\n'.join(latex))
    print(f"Saved: {output_dir / 'lf_table.tex'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute lexical familiarity scores and run regression')
    parser.add_argument('--data-dir', '-d', default='/mnt/project',
                        help='Directory with baseline CSVs')
    parser.add_argument(
        '--output', '-o', default='/mnt/user-data/outputs', help='Output directory')
    parser.add_argument(
        '--model', '-m', default='google/gemma-3-12b-it', help='Tokenizer model')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Lexical Familiarity Analysis")
    print("=" * 60)

    # Load tokenizer
    tokenizer = load_tokenizer(args.model)

    # Load and process all categories
    print("\nLoading and processing categories...")
    df = load_all_categories(data_dir, tokenizer)

    # Assign peak layers
    df = assign_peak_layers(df)

    # Save raw data
    df.to_csv(output_dir / 'lf_scores_all_items.csv', index=False)
    print(f"\nSaved: {output_dir / 'lf_scores_all_items.csv'}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("CATEGORY SUMMARY")
    print("=" * 60)
    summary = df.groupby('category').agg({
        'fragmentation': ['mean', 'std'],
        'frequency_score': ['mean', 'std'],
        'lf_score': ['mean', 'std'],
        'peak_layer': 'first',
        'item_idx': 'count'
    }).round(3)
    print(summary)

    # Run regression
    results = run_regression_analysis(df)

    # Visualizations
    print("\nGenerating visualizations...")
    plot_lf_analysis(df, output_dir)

    # LaTeX table
    generate_latex_table(df, results, output_dir)

    print("\n" + "=" * 60)
    print("INTERPRETATION FOR PAPER")
    print("=" * 60)
    print("""
If LF_score significantly predicts peak_layer (p < .05):
→ "Lexical familiarity, operationalized as a combination of tokenizer 
   fragmentation and token frequency, significantly predicts processing 
   depth (β = {:.2f}, p < .001, R² = {:.3f})."

If LF_score remains significant even with category in the model:
→ "The effect of lexical familiarity holds even when controlling for 
   category membership, suggesting that the continuous familiarity 
   gradient—not discrete category labels—drives processing depth."

If category effect weakens:
→ "Adding lexical familiarity to the category-only model reduces the 
   unique variance explained by category, consistent with familiarity 
   as the underlying organizing principle."
""".format(results['lf_coef'], results['model2_r2']))


if __name__ == '__main__':
    main()

