# =============================================================================
# Association Rule Mining – Apriori Algorithm
# Dataset: Groceries_dataset.csv  (38,765 rows)
# =============================================================================

# ── 1. Import Libraries ──────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ── 2. Load Dataset ──────────────────────────────────────────────────────────
df = pd.read_csv("Groceries_dataset.csv")

print("=" * 65)
print("DATASET OVERVIEW")
print("=" * 65)
print(f"Shape            : {df.shape}")
print(f"Columns          : {df.columns.tolist()}")
print(f"Unique members   : {df['Member_number'].nunique()}")
print(f"Unique items     : {df['itemDescription'].nunique()}")
print(f"Missing values   : {df.isnull().sum().sum()}")
print(df.head(6))

# ── 3. Preprocessing – Build One-Hot Encoded Basket ──────────────────────────
# Group by Member_number (each member = one "transaction" across all visits)
# This yields avg 8.9 items/member, enabling meaningful multi-item rules
basket_raw = (
    df.groupby(["Member_number", "itemDescription"])["itemDescription"]
    .count()
    .unstack(fill_value=0)
)
basket = basket_raw.map(lambda x: True if x > 0 else False)

print("\n" + "=" * 65)
print("ONE-HOT ENCODED BASKET")
print("=" * 65)
print(f"Basket shape          : {basket.shape}  (members × items)")
items_per_member = basket.sum(axis=1)
print(f"Items per member      : mean={items_per_member.mean():.2f}, "
      f"min={items_per_member.min()}, max={items_per_member.max()}")

# Top 10 most frequent items
top_items = basket.sum().sort_values(ascending=False).head(10)
print("\nTop 10 items (member frequency):")
for item, count in top_items.items():
    print(f"  {item:<30} {count:>5}  ({count/len(basket)*100:.1f}%)")

# ── 4 & 5. Generate Frequent Itemsets at Multiple Support Thresholds ─────────
print("\n" + "=" * 65)
print("EFFECT OF SUPPORT THRESHOLD ON FREQUENT ITEMSETS")
print("=" * 65)

thresholds = [0.10, 0.07, 0.05, 0.03]
for thresh in thresholds:
    fi = apriori(basket, min_support=thresh, use_colnames=True)
    sizes = fi["itemsets"].apply(len).value_counts().sort_index()
    size_str = ", ".join([f"{k}-item:{v}" for k, v in sizes.items()])
    print(f"  min_support={thresh:.2f} → {len(fi):>4} itemsets  [{size_str}]")

# Use min_support=0.05 for full analysis (165 itemsets, good variety)
MIN_SUPPORT    = 0.05
MIN_CONFIDENCE = 0.30
MIN_LIFT       = 1.0

freq_items = apriori(basket, min_support=MIN_SUPPORT, use_colnames=True)
freq_items["length"] = freq_items["itemsets"].apply(len)

print(f"\n✅ Selected min_support={MIN_SUPPORT}  →  {len(freq_items)} frequent itemsets")
print(f"   1-itemsets: {(freq_items.length==1).sum()}")
print(f"   2-itemsets: {(freq_items.length==2).sum()}")
print(f"   3-itemsets: {(freq_items.length==3).sum()}")

print("\nTop 15 Frequent Itemsets (by support):")
top_fi = freq_items.sort_values("support", ascending=False).head(15)
for _, row in top_fi.iterrows():
    items_str = ", ".join(sorted(row["itemsets"]))
    print(f"  support={row['support']:.4f}  [{items_str}]")

# ── 6 & 7. Generate & Filter Association Rules ────────────────────────────────
rules = association_rules(freq_items, metric="confidence",
                          min_threshold=MIN_CONFIDENCE)
rules = rules[rules["lift"] >= MIN_LIFT].sort_values("lift", ascending=False)

print("\n" + "=" * 65)
print(f"ASSOCIATION RULES  (conf≥{MIN_CONFIDENCE}, lift≥{MIN_LIFT})")
print("=" * 65)
print(f"Total rules generated : {len(rules)}")

# Format for display
def fmt_rule(row):
    ant = ", ".join(sorted(row["antecedents"]))
    con = ", ".join(sorted(row["consequents"]))
    return (f"  [{ant}] → [{con}]  "
            f"supp={row['support']:.4f}  "
            f"conf={row['confidence']:.4f}  "
            f"lift={row['lift']:.4f}")

print("\nTop 15 Rules by Lift:")
for _, row in rules.head(15).iterrows():
    print(fmt_rule(row))

# Strong rules: high confidence
print("\nTop 10 Rules by Confidence:")
for _, row in rules.sort_values("confidence", ascending=False).head(10).iterrows():
    print(fmt_rule(row))

# Rules with lift > 1.3 (strong positive association)
strong = rules[rules["lift"] > 1.3]
print(f"\nRules with lift > 1.3  : {len(strong)}")
print(f"Rules with lift > 1.25 : {len(rules[rules['lift'] > 1.25])}")
print(f"Rules with conf > 0.5  : {len(rules[rules['confidence'] > 0.5])}")

# ── 8. Interpret Rules ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RULE INTERPRETATION (Top 5 by Lift)")
print("=" * 65)
for i, (_, row) in enumerate(rules.head(5).iterrows(), 1):
    ant = ", ".join(sorted(row["antecedents"]))
    con = ", ".join(sorted(row["consequents"]))
    print(f"\n  Rule {i}: [{ant}] → [{con}]")
    print(f"    Support    = {row['support']:.4f}  "
          f"({row['support']*100:.1f}% of members bought both)")
    print(f"    Confidence = {row['confidence']:.4f}  "
          f"({row['confidence']*100:.1f}% of [{ant}] buyers also bought [{con}])")
    print(f"    Lift       = {row['lift']:.4f}  "
          f"({'positive' if row['lift']>1 else 'negative'} association, "
          f"{(row['lift']-1)*100:.1f}% more likely than random)")

# ── 9. Visualizations ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Apriori – Market Basket Analysis: Groceries Dataset",
             fontsize=16, fontweight="bold")

# ─ Plot 1: Top 15 Frequent 1-Itemsets (bar chart) ───────────────────────────
ax1 = fig.add_subplot(3, 3, 1)
single = freq_items[freq_items.length == 1].copy()
single["item"] = single["itemsets"].apply(lambda x: list(x)[0])
single = single.sort_values("support", ascending=True).tail(15)
bars = ax1.barh(single["item"], single["support"] * 100,
                color=plt.cm.Blues(np.linspace(0.4, 0.9, len(single))))
ax1.set_xlabel("Support (%)")
ax1.set_title("Top 15 Frequent Items\n(1-itemsets)", fontsize=11)
ax1.axvline(MIN_SUPPORT * 100, color="red", linestyle="--",
            linewidth=1.2, label=f"min_supp={MIN_SUPPORT*100:.0f}%")
ax1.legend(fontsize=8)
ax1.grid(axis="x", alpha=0.3)

# ─ Plot 2: Top 15 Frequent 2-Itemsets ───────────────────────────────────────
ax2 = fig.add_subplot(3, 3, 2)
pairs = freq_items[freq_items.length == 2].copy()
pairs["label"] = pairs["itemsets"].apply(lambda x: "\n+ ".join(sorted(x)))
pairs = pairs.sort_values("support", ascending=True).tail(15)
ax2.barh(pairs["label"], pairs["support"] * 100,
         color=plt.cm.Greens(np.linspace(0.4, 0.9, len(pairs))))
ax2.set_xlabel("Support (%)")
ax2.set_title("Top 15 Frequent Pairs\n(2-itemsets)", fontsize=11)
ax2.grid(axis="x", alpha=0.3)
ax2.tick_params(axis="y", labelsize=7)


# ─ Plot 4: Support vs Confidence scatter ────────────────────────────────────
ax4 = fig.add_subplot(3, 3, 4)
sc = ax4.scatter(rules["support"] * 100, rules["confidence"] * 100,
                 c=rules["lift"], cmap="RdYlGn", s=60,
                 alpha=0.8, edgecolors="grey", linewidth=0.3,
                 vmin=1.0, vmax=rules["lift"].max())
plt.colorbar(sc, ax=ax4, label="Lift")
ax4.set_xlabel("Support (%)")
ax4.set_ylabel("Confidence (%)")
ax4.set_title("Support vs Confidence\n(color = Lift)", fontsize=11)
ax4.grid(True, alpha=0.3)

# ─ Plot 8: Network Graph of Top Association Rules ───────────────────────────
ax8 = fig.add_subplot(3, 3, (8, 9))
top_rules = rules.head(25)
G = nx.DiGraph()
for _, row in top_rules.iterrows():
    ant = ", ".join(sorted(row["antecedents"]))
    con = ", ".join(sorted(row["consequents"]))
    G.add_edge(ant, con, weight=row["lift"], confidence=row["confidence"])

pos = nx.spring_layout(G, seed=42, k=2.5)
edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
edge_norm = [(w - min(edge_weights)) / (max(edge_weights) - min(edge_weights) + 1e-9)
             for w in edge_weights]

# Color nodes by degree
node_colors = ["#FF6B6B" if G.in_degree(n) > G.out_degree(n) else "#4ECDC4"
               for n in G.nodes()]

nx.draw_networkx_nodes(G, pos, ax=ax8, node_color=node_colors,
                       node_size=1200, alpha=0.85)
nx.draw_networkx_labels(G, pos, ax=ax8, font_size=6.5, font_weight="bold")
nx.draw_networkx_edges(G, pos, ax=ax8,
                       edge_color=[plt.cm.Reds(0.4 + 0.6 * w) for w in edge_norm],
                       arrows=True, arrowsize=18,
                       width=[1.5 + 3 * w for w in edge_norm],
                       connectionstyle="arc3,rad=0.15")

# Legend
patch_ant  = mpatches.Patch(color="#4ECDC4", label="Antecedent-dominant")
patch_con  = mpatches.Patch(color="#FF6B6B", label="Consequent-dominant")
ax8.legend(handles=[patch_ant, patch_con], fontsize=8, loc="lower left")
ax8.set_title("Association Rule Network Graph\n(Top 25 rules, edge color/width ∝ Lift)",
              fontsize=11)
ax8.axis("off")

plt.tight_layout()
plt.savefig("apriori_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ Plot saved as 'apriori_results.png'")

# ── 10. Final Metrics Summary ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL EVALUATION METRICS SUMMARY")
print("=" * 65)
print(f"  Transactions (members)    : {len(basket)}")
print(f"  Unique items              : {basket.shape[1]}")
print(f"  Min Support Used          : {MIN_SUPPORT}  ({MIN_SUPPORT*100:.0f}%)")
print(f"  Frequent Itemsets Found   : {len(freq_items)}")
print(f"    → 1-itemsets            : {(freq_items.length==1).sum()}")
print(f"    → 2-itemsets            : {(freq_items.length==2).sum()}")
print(f"    → 3-itemsets            : {(freq_items.length==3).sum()}")
print(f"  Association Rules         : {len(rules)}")
print(f"  Lift  – mean={rules['lift'].mean():.4f}, "
      f"max={rules['lift'].max():.4f}, min={rules['lift'].min():.4f}")
print(f"  Conf  – mean={rules['confidence'].mean():.4f}, "
      f"max={rules['confidence'].max():.4f}")
print(f"  Supp  – mean={rules['support'].mean():.4f}, "
      f"max={rules['support'].max():.4f}")
print("=" * 65)
print("Done.")