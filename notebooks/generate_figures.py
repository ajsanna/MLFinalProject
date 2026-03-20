"""
Generate publication figures for the final report.
Outputs to notebooks/figures/.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'

OUT = "figures"

# ─────────────────────────────────────────────────────────────────
# Figure 1: NA Label Breakthrough (grouped bar chart)
# ─────────────────────────────────────────────────────────────────
na_labels = [
    "True_Correct:NA\n(n=2,961)",
    "True_Neither:NA\n(n=1,053)",
    "False_Neither:NA\n(n=1,308)",
    "False_Correct:NA\n(n=45)",
]
phase2 = [0.0000, 0.0000, 0.0000, 0.0000]
phase3 = [0.5669, 0.4525, 0.3292, 0.3074]

x = np.arange(len(na_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, phase2, width, label='Phase 2 (Zero-Shot)',
               color='#c0392b', alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + width/2, phase3, width, label='Phase 3 (Fine-Tuned)',
               color='#27ae60', alpha=0.85, edgecolor='white')

for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
            f'{h:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(na_labels, fontsize=10)
ax.set_ylabel('MAP@3')
ax.set_ylim(0, 0.75)
ax.set_title('MAP@3 on NA Labels: Zero-Shot vs. Fine-Tuned', fontweight='bold', pad=12)
ax.legend(framealpha=0.9)
ax.axhline(0, color='black', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/na_label_breakthrough.png", bbox_inches='tight')
plt.close()
print("Saved na_label_breakthrough.png")

# ─────────────────────────────────────────────────────────────────
# Figure 2: Training Curve
# ─────────────────────────────────────────────────────────────────
epochs  = [0, 1, 2]
map_vals = [0.2004, 0.6073, 0.5902]
labels_x = ['Zero-Shot\n(Baseline)', 'Epoch 1\n(Best)', 'Epoch 2']

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(epochs, map_vals, marker='o', markersize=9, linewidth=2.5,
        color='#2980b9', zorder=3)

for i, (x_val, y_val) in enumerate(zip(epochs, map_vals)):
    offset = 0.022 if i != 1 else 0.022
    ax.annotate(f'{y_val:.4f}',
                xy=(x_val, y_val),
                xytext=(x_val, y_val + offset),
                ha='center', fontsize=10, fontweight='bold', color='#2980b9')

# Highlight best point
ax.plot(1, 0.6073, marker='*', markersize=16, color='#f39c12', zorder=4)
ax.axhline(0.2004, color='#c0392b', linestyle='--', linewidth=1.2,
           label='Zero-shot baseline (0.2004)', alpha=0.7)

ax.set_xticks(epochs)
ax.set_xticklabels(labels_x, fontsize=10)
ax.set_ylabel('MAP@3 (Validation)')
ax.set_ylim(0.1, 0.72)
ax.set_title('Validation MAP@3 Across Training Phases', fontweight='bold', pad=12)
ax.legend(framealpha=0.9, fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/training_curve.png", bbox_inches='tight')
plt.close()
print("Saved training_curve.png")

# ─────────────────────────────────────────────────────────────────
# Figure 4: Per-Label MAP@3 Ranked Bar Chart
# ─────────────────────────────────────────────────────────────────
df = pd.read_csv("../models/run_20260307_184107/per_label_epoch1.csv")
df = df.sort_values("map", ascending=True).reset_index(drop=True)

def get_color(label):
    if label.startswith("True_Correct"):
        return '#27ae60'
    elif label.startswith("False_Misconception"):
        return '#e74c3c'
    elif label.startswith("True_Misconception"):
        return '#e67e22'
    elif label.startswith("False_Neither") or label.startswith("True_Neither"):
        return '#95a5a6'
    elif label.startswith("False_Correct"):
        return '#3498db'
    else:
        return '#bdc3c7'

colors = [get_color(l) for l in df["label"]]

fig, ax = plt.subplots(figsize=(9, 14))
bars = ax.barh(df["label"], df["map"], color=colors, edgecolor='white', linewidth=0.3)

ax.axvline(0.8, color='black', linestyle='--', linewidth=1.0, alpha=0.5, label='MAP@3 = 0.80')
ax.set_xlabel('MAP@3')
ax.set_title('Per-Label MAP@3 — Fine-Tuned Model (Epoch 1)', fontweight='bold', pad=12)
ax.set_xlim(0, 1.08)
ax.tick_params(axis='y', labelsize=7.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_patches = [
    mpatches.Patch(color='#27ae60', label='True_Correct'),
    mpatches.Patch(color='#e74c3c', label='False_Misconception'),
    mpatches.Patch(color='#e67e22', label='True_Misconception'),
    mpatches.Patch(color='#95a5a6', label='*_Neither'),
    mpatches.Patch(color='#3498db', label='False_Correct'),
]
ax.legend(handles=legend_patches, fontsize=9, loc='lower right', framealpha=0.9)
ax.axvline(0.8, color='black', linestyle='--', linewidth=1.0, alpha=0.5)

plt.tight_layout()
plt.savefig(f"{OUT}/per_label_map3.png", bbox_inches='tight')
plt.close()
print("Saved per_label_map3.png")

# ─────────────────────────────────────────────────────────────────
# Figure: Table 1 — Overall Model Comparison
# ─────────────────────────────────────────────────────────────────
models = ['Zero-Shot\n(Frozen MiniLM)', 'Fine-Tuned\n(MiniLM, Epoch 1)']
map3   = [0.2004, 0.6073]
top1   = [0.1619, None]   # only available for zero-shot
top3   = [0.2530, None]

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(models))
width = 0.25

b1 = ax.bar(x - width, map3,           width, label='MAP@3',         color='#2980b9', alpha=0.88, edgecolor='white')
b2 = ax.bar(x,         [0.1619, 0],    width, label='Top-1 Accuracy',color='#e67e22', alpha=0.88, edgecolor='white')
b3 = ax.bar(x + width, [0.2530, 0],    width, label='Top-3 Accuracy',color='#27ae60', alpha=0.88, edgecolor='white')

# Annotate MAP@3 bars
for bar, val in zip(b1, map3):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Annotate zero-shot top-1 and top-3
ax.text(b2[0].get_x() + b2[0].get_width()/2, b2[0].get_height() + 0.012,
        '16.19%', ha='center', va='bottom', fontsize=9)
ax.text(b3[0].get_x() + b3[0].get_width()/2, b3[0].get_height() + 0.012,
        '25.30%', ha='center', va='bottom', fontsize=9)

# Note for fine-tuned missing values
ax.text(b2[1].get_x() + b2[1].get_width()/2 + width/2, 0.025,
        'N/A', ha='center', va='bottom', fontsize=8, color='#7f8c8d', style='italic')

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel('Score')
ax.set_ylim(0, 0.75)
ax.set_title('Overall Model Performance: Zero-Shot vs. Fine-Tuned', fontweight='bold', pad=12)
ax.legend(framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/overall_comparison.png", bbox_inches='tight')
plt.close()
print("Saved overall_comparison.png")

print("\nAll figures saved to notebooks/figures/")
