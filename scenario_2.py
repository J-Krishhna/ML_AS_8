# =============================================================================
# Dimensionality Reduction Using PCA – Wine Dataset (sklearn built-in)
# =============================================================================

# ── 1. Import Libraries ──────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import warnings
warnings.filterwarnings("ignore")

# ── 2. Load Dataset ──────────────────────────────────────────────────────────
wine = load_wine()
X_raw = wine.data
y     = wine.target
feat_names  = wine.feature_names
class_names = wine.target_names

df = pd.DataFrame(X_raw, columns=feat_names)
df["class"] = y

print("=" * 65)
print("DATASET OVERVIEW – Wine Dataset (sklearn built-in)")
print("=" * 65)
print(f"Samples          : {X_raw.shape[0]}")
print(f"Features         : {X_raw.shape[1]}")
print(f"Classes          : {list(class_names)}  →  counts: {np.bincount(y).tolist()}")
print(f"Missing values   : {df.isnull().sum().sum()}")
print(f"\nFeature names:\n  {feat_names}")
print("\nDescriptive statistics:")
print(df.drop("class", axis=1).describe().round(3).to_string())

# ── 3. Standardise Features ───────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print("\n" + "=" * 65)
print("STANDARDISATION CHECK")
print("=" * 65)
print(f"  Mean (all features) ≈ {X_scaled.mean(axis=0).round(6).tolist()[:3]} ... (all ~0)")
print(f"  Std  (all features) ≈ {X_scaled.std(axis=0).round(6).tolist()[:3]} ... (all ~1)")
print(f"\n  Feature ranges BEFORE scaling:")
for i, f in enumerate(feat_names):
    print(f"    {f:<35} min={X_raw[:,i].min():.3f}  max={X_raw[:,i].max():.3f}")
print(f"\n  After scaling: every feature has mean=0, std=1")

# ── 4 & 5. Apply PCA – Compute All Principal Components ──────────────────────
pca_full = PCA(n_components=X_raw.shape[1], random_state=42)
pca_full.fit(X_scaled)

evr  = pca_full.explained_variance_ratio_          # each PC's share
cum  = np.cumsum(evr)                              # cumulative

print("\n" + "=" * 65)
print("PCA – ALL COMPONENTS: EXPLAINED VARIANCE")
print("=" * 65)
print(f"{'PC':>4}  {'Eigenvalue':>12}  {'Expl.Var(%)':>12}  {'Cumulative(%)':>14}")
print("-" * 48)
for i, (ev, e, c) in enumerate(zip(pca_full.explained_variance_, evr, cum), 1):
    flag = " ◄" if c <= 0.95 and (i == 1 or cum[i-2] < 0.95) else ""
    print(f"  PC{i:<2}  {ev:>12.4f}  {e*100:>12.4f}  {c*100:>13.4f}%{flag}")

# Optimal components
n_95 = np.argmax(cum >= 0.95) + 1
n_90 = np.argmax(cum >= 0.90) + 1
n_80 = np.argmax(cum >= 0.80) + 1
print(f"\n  Components to retain 80% variance : {n_80}")
print(f"  Components to retain 90% variance : {n_90}")
print(f"  Components to retain 95% variance : {n_95}")
print(f"\n  → Using PC1 + PC2 for 2-D visualisation  "
      f"(captures {cum[1]*100:.2f}% variance)")
print(f"  → Using PC1+PC2+PC3 for 3-D visualisation "
      f"(captures {cum[2]*100:.2f}% variance)")

# ── 6. Feature Loadings ───────────────────────────────────────────────────────
loadings = pd.DataFrame(
    pca_full.components_.T,
    index=feat_names,
    columns=[f"PC{i+1}" for i in range(X_raw.shape[1])]
)
print("\n" + "=" * 65)
print("FEATURE LOADINGS (contribution to each PC)")
print("=" * 65)
print(loadings[["PC1", "PC2", "PC3"]].round(4).to_string())
print("\nTop 3 features driving PC1 (|loading|):")
top_pc1 = loadings["PC1"].abs().sort_values(ascending=False).head(3)
for feat, val in top_pc1.items():
    print(f"  {feat:<35} loading={val:.4f}")
print("Top 3 features driving PC2 (|loading|):")
top_pc2 = loadings["PC2"].abs().sort_values(ascending=False).head(3)
for feat, val in top_pc2.items():
    print(f"  {feat:<35} loading={val:.4f}")

# ── 7. Reduce to 2D and 3D ────────────────────────────────────────────────────
pca_2d = PCA(n_components=2, random_state=42)
pca_3d = PCA(n_components=3, random_state=42)
X_2d   = pca_2d.fit_transform(X_scaled)
X_3d   = pca_3d.fit_transform(X_scaled)

print("\n" + "=" * 65)
print("DIMENSIONALITY REDUCTION SUMMARY")
print("=" * 65)
print(f"  Original shape       : {X_raw.shape}")
print(f"  Reduced to 2D shape  : {X_2d.shape}  "
      f"(variance retained: {pca_2d.explained_variance_ratio_.sum()*100:.2f}%)")
print(f"  Reduced to 3D shape  : {X_3d.shape}  "
      f"(variance retained: {pca_3d.explained_variance_ratio_.sum()*100:.2f}%)")

# Reconstruction error (2D)
X_reconstructed = pca_2d.inverse_transform(X_2d)
recon_error = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"  Reconstruction MSE (2D) : {recon_error:.6f}")

# Original vs reduced comparison (first 5 rows, first 4 features)
print("\n  Original data (first 5 rows, first 4 features, scaled):")
print(pd.DataFrame(X_scaled[:5, :4], columns=feat_names[:4]).round(4).to_string())
print("\n  After PCA → 2 components (first 5 rows):")
print(pd.DataFrame(X_2d[:5], columns=["PC1", "PC2"]).round(4).to_string())

# ── 8. Visualisations ─────────────────────────────────────────────────────────
colors  = ["#E63946", "#2A9D8F", "#E9C46A"]
markers = ["o", "s", "^"]
clabels = list(class_names)

fig = plt.figure(figsize=(18, 14))
fig.suptitle("PCA – Wine Dataset Dimensionality Reduction",
             fontsize=16, fontweight="bold")

# ─ Plot 1: Scree Plot ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(3, 3, 1)
pcs = range(1, X_raw.shape[1] + 1)
ax1.bar(pcs, evr * 100, color="steelblue", alpha=0.7, label="Individual")
ax1.plot(pcs, evr * 100, "ro-", markersize=6)
ax1.set_xlabel("Principal Component")
ax1.set_ylabel("Explained Variance (%)")
ax1.set_title("Scree Plot\n(Individual Variance per PC)", fontsize=11)
ax1.axhline(y=5, color="grey", linestyle=":", linewidth=1, label="5% line")
ax1.legend(fontsize=8)
ax1.set_xticks(list(pcs))
ax1.grid(axis="y", alpha=0.3)

# ─ Plot 2: Cumulative Variance ────────────────────────────────────────────────
ax2 = fig.add_subplot(3, 3, 2)
ax2.plot(pcs, cum * 100, "g^-", linewidth=2, markersize=7)
ax2.fill_between(pcs, cum * 100, alpha=0.15, color="green")
for thresh, col in [(0.80, "orange"), (0.90, "red"), (0.95, "purple")]:
    n = np.argmax(cum >= thresh) + 1
    ax2.axhline(thresh * 100, color=col, linestyle="--", linewidth=1.2,
                label=f"{int(thresh*100)}% @ PC{n}")
ax2.set_xlabel("Number of Components")
ax2.set_ylabel("Cumulative Variance (%)")
ax2.set_title("Cumulative Explained Variance", fontsize=11)
ax2.legend(fontsize=8)
ax2.set_xticks(list(pcs))
ax2.grid(True, alpha=0.3)



# ─ Plot 4: 2D Scatter – PC1 vs PC2 ───────────────────────────────────────────
ax4 = fig.add_subplot(3, 3, 4)
for i, (cls, col, mk) in enumerate(zip(clabels, colors, markers)):
    mask = y == i
    ax4.scatter(X_2d[mask, 0], X_2d[mask, 1],
                color=col, marker=mk, label=cls, s=60, alpha=0.85,
                edgecolors="white", linewidth=0.5)
ax4.set_xlabel(f"PC1 ({evr[0]*100:.2f}% variance)", fontsize=10)
ax4.set_ylabel(f"PC2 ({evr[1]*100:.2f}% variance)", fontsize=10)
ax4.set_title("2D PCA – PC1 vs PC2\n(Wine Classes)", fontsize=11)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# ─ Plot 5: 3D Scatter ─────────────────────────────────────────────────────────
ax5 = fig.add_subplot(3, 3, 5, projection="3d")
for i, (cls, col, mk) in enumerate(zip(clabels, colors, markers)):
    mask = y == i
    ax5.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
                color=col, marker=mk, label=cls, s=50, alpha=0.80)
ax5.set_xlabel(f"PC1", fontsize=9)
ax5.set_ylabel(f"PC2", fontsize=9)
ax5.set_zlabel(f"PC3", fontsize=9)
ax5.set_title(f"3D PCA  ({pca_3d.explained_variance_ratio_.sum()*100:.2f}% var.)",
              fontsize=11)
ax5.legend(fontsize=8)



plt.tight_layout()
plt.savefig("pca_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ Plot saved as 'pca_results.png'")

# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL EVALUATION METRICS SUMMARY")
print("=" * 65)
print(f"  Original dimensions      : {X_raw.shape[1]}")
print(f"  PC1 explained variance   : {evr[0]*100:.4f}%")
print(f"  PC2 explained variance   : {evr[1]*100:.4f}%")
print(f"  PC3 explained variance   : {evr[2]*100:.4f}%")
print(f"  PC1+PC2 cumulative       : {cum[1]*100:.4f}%")
print(f"  PC1+PC2+PC3 cumulative   : {cum[2]*100:.4f}%")
print(f"  Components for 80% var   : {n_80}")
print(f"  Components for 90% var   : {n_90}")
print(f"  Components for 95% var   : {n_95}")
print(f"  Reconstruction MSE (2D)  : {recon_error:.6f}")
print(f"  Top PC1 driver           : {top_pc1.index[0]}  (loading={top_pc1.values[0]:.4f})")
print(f"  Top PC2 driver           : {top_pc2.index[0]}  (loading={top_pc2.values[0]:.4f})")
print("=" * 65)
print("Done.")