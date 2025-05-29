import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load or recreate the synthetic dataset
np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=100)
sales = 50 + np.random.normal(0, 5, 100)
sales[50:] += 10  # Promotion effect starts at day 50

df = pd.DataFrame({"date": dates, "sales": sales})
df["promotion"] = 0
df.loc[50:, "promotion"] = 1

# Convert promotion to a categorical variable for better visualization
df["promotion_category"] = df["promotion"].map({0: "No Promotion", 1: "Promotion"})

# Create a figure with multiple subplots
plt.figure(figsize=(15, 10))

# 1. Bar chart comparing average sales
plt.subplot(2, 2, 1)
avg_sales = df.groupby("promotion_category")["sales"].mean()
avg_sales.plot(kind="bar", color=["#ff9999", "#66b3ff"])
plt.title("Average Sales by Promotion Status")
plt.ylabel("Average Sales")
plt.xticks(rotation=0)
for i, v in enumerate(avg_sales):
    plt.text(i, v + 0.5, f"{v:.2f}", ha="center")
plt.grid(axis="y", alpha=0.3)

# 2. Box plot to show distribution
plt.subplot(2, 2, 2)
sns.boxplot(x="promotion_category", y="sales", data=df, palette=["#ff9999", "#66b3ff"])
plt.title("Distribution of Sales by Promotion Status")
plt.ylabel("Sales")
plt.xlabel("")
plt.grid(axis="y", alpha=0.3)

# 3. Time series with clear before/after
plt.subplot(2, 1, 2)
plt.plot(df["date"], df["sales"], marker="o", markersize=4, linestyle="-", color="gray", alpha=0.6)
plt.axvline(x=df["date"][49], color="red", linestyle="--", label="Promotion Start")

# Color the points before and after promotion
no_promo = df[df["promotion"] == 0]
with_promo = df[df["promotion"] == 1]
plt.scatter(no_promo["date"], no_promo["sales"], color="#ff9999", label="No Promotion", s=50, zorder=2)
plt.scatter(with_promo["date"], with_promo["sales"], color="#66b3ff", label="Promotion", s=50, zorder=2)

# Add trend lines
no_promo_x = np.arange(len(no_promo))
with_promo_x = np.arange(len(with_promo))
no_promo_trend = np.poly1d(np.polyfit(no_promo_x, no_promo["sales"], 1))
with_promo_trend = np.poly1d(np.polyfit(with_promo_x, with_promo["sales"], 1))

plt.plot(no_promo["date"], no_promo_trend(no_promo_x), color="#ff9999", linestyle="-", linewidth=2)
plt.plot(with_promo["date"], with_promo_trend(with_promo_x), color="#66b3ff", linestyle="-", linewidth=2)

plt.title("Sales Over Time: Before and After Promotion")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(alpha=0.3)

# Add annotations
plt.annotate(f"Average: {no_promo['sales'].mean():.2f}",
             xy=(df["date"][25], no_promo["sales"].mean()),
             xytext=(df["date"][25], no_promo["sales"].mean() + 8),
             arrowprops=dict(arrowstyle="->", color="#ff9999"),
             color="#ff9999", fontweight="bold")

plt.annotate(f"Average: {with_promo['sales'].mean():.2f}",
             xy=(df["date"][75], with_promo["sales"].mean()),
             xytext=(df["date"][75], with_promo["sales"].mean() + 8),
             arrowprops=dict(arrowstyle="->", color="#66b3ff"),
             color="#66b3ff", fontweight="bold")

plt.annotate(f"Difference: +{with_promo['sales'].mean() - no_promo['sales'].mean():.2f}",
             xy=(df["date"][50], (with_promo["sales"].mean() + no_promo["sales"].mean())/2),
             xytext=(df["date"][35], (with_promo["sales"].mean() + no_promo["sales"].mean())/2 + 5),
             arrowprops=dict(arrowstyle="->", color="black"),
             color="black", fontweight="bold")

# Add summary statistics as text
stats_text = (
    f"Statistical Summary:\n"
    f"• Before Promotion: Mean = {no_promo['sales'].mean():.2f}, Std = {no_promo['sales'].std():.2f}\n"
    f"• After Promotion: Mean = {with_promo['sales'].mean():.2f}, Std = {with_promo['sales'].std():.2f}\n"
    f"• Difference: {with_promo['sales'].mean() - no_promo['sales'].mean():.2f} units\n"
    f"• The promotion increased sales by approximately {(with_promo['sales'].mean() - no_promo['sales'].mean())/no_promo['sales'].mean()*100:.1f}%"
)

plt.figtext(0.5, 0.01, stats_text, ha="center", fontsize=12, bbox={"facecolor":"#f0f0f0", "alpha":0.5, "pad":5})

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig("promotion_effect_visualization.png", dpi=300)
print("Visualization saved as 'promotion_effect_visualization.png'")

# Create additional before/after histogram visualization
plt.figure(figsize=(12, 6))
plt.hist(no_promo["sales"], bins=15, alpha=0.7, label="No Promotion", color="#ff9999")
plt.hist(with_promo["sales"], bins=15, alpha=0.7, label="Promotion", color="#66b3ff")
plt.axvline(no_promo["sales"].mean(), color="#ff9999", linestyle="dashed", linewidth=2)
plt.axvline(with_promo["sales"].mean(), color="#66b3ff", linestyle="dashed", linewidth=2)
plt.legend()
plt.title("Distribution of Sales: Before vs. After Promotion")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.savefig("sales_distribution_comparison.png", dpi=300)
print("Distribution comparison saved as 'sales_distribution_comparison.png'")

