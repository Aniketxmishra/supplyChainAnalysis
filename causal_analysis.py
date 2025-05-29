import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dowhy
from dowhy import CausalModel
import networkx as nx
from scipy import stats

print("Running Supply Chain Causal Analysis...")

# Load or recreate the synthetic dataset
np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=100)
sales = 50 + np.random.normal(0, 5, 100)
sales[50:] += 10  # Promotion effect starts at day 50

df = pd.DataFrame({"date": dates, "sales": sales})
df["promotion"] = 0
df.loc[50:, "promotion"] = 1

print("Data created successfully.")
print(f"Total observations: {len(df)}")
print(f"Observations with promotion: {df['promotion'].sum()}")
print(f"Observations without promotion: {len(df) - df['promotion'].sum()}")

# Visualize the data
plt.figure(figsize=(10,5))
plt.plot(df["date"], df["sales"], label='Sales')
plt.axvline(df["date"][50], color='red', linestyle='--', label='Promotion Starts')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.title("Sales Over Time with Promotion")
plt.savefig('sales_over_time.png')
plt.close()

print("Sales visualization saved as 'sales_over_time.png'")

# Visualize the causal graph using matplotlib
plt.figure(figsize=(6, 4))
G = nx.DiGraph()
G.add_edge('Promotion', 'Sales')
pos = {'Promotion': (0, 0), 'Sales': (1, 0)}
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', 
        font_size=12, font_weight='bold', arrowsize=20)
plt.title('Causal Graph: Promotion → Sales')
plt.savefig('causal_graph.png')
plt.close()

print("Causal graph visualization saved as 'causal_graph.png'")

# Perform basic statistical analysis to compare sales with and without promotion
sales_no_promotion = df[df['promotion'] == 0]['sales']
sales_with_promotion = df[df['promotion'] == 1]['sales']

# T-test to see if the difference is statistically significant
t_stat, p_value = stats.ttest_ind(sales_with_promotion, sales_no_promotion)

print("\n=== STATISTICAL ANALYSIS ===")
print(f"Average sales without promotion: {sales_no_promotion.mean():.4f}")
print(f"Average sales with promotion: {sales_with_promotion.mean():.4f}")
print(f"Difference: {sales_with_promotion.mean() - sales_no_promotion.mean():.4f}")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.8f}")
print(f"The difference is {'statistically significant' if p_value < 0.05 else 'not statistically significant'}")

# Create a causal model using DoWhy with a simpler graph format
model = CausalModel(
    data=df,
    treatment='promotion',
    outcome='sales',
    common_causes=None,  # No common causes in our simple model
    instruments=None,    # No instrumental variables in our simple model
    effect_modifiers=None  # No effect modifiers in our simple model
)

# Step 2: Identify the causal effect
identified_estimand = model.identify_effect()
print("Identified estimand:")
print(identified_estimand)

# Step 3: Estimate the causal effect using various methods
# Method 1: Backdoor adjustment (simple regression)
estimate = model.estimate_effect(identified_estimand,
                                method_name="backdoor.linear_regression")
print("\nCausal Estimate (Linear Regression):")
print(estimate)
print("Effect of promotion on sales:", estimate.value)

# Method 2: Propensity score matching
estimate_ps = model.estimate_effect(identified_estimand,
                                  method_name="backdoor.propensity_score_matching")
print("\nCausal Estimate (Propensity Score Matching):")
print(estimate_ps)
print("Effect of promotion on sales:", estimate_ps.value)

# Method 3: Double Machine Learning
try:
    estimate_dml = model.estimate_effect(identified_estimand,
                                      method_name="backdoor.econml.dml.DML")
    print("\nCausal Estimate (Double Machine Learning):")
    print(estimate_dml)
    print("Effect of promotion on sales:", estimate_dml.value)
except Exception as e:
    print("\nCould not run Double Machine Learning method:", str(e))

# Step 4: Refute the estimated effect
print("\n== REFUTATION TESTS ==")

# Test 1: Add a random common cause
refute_random = model.refute_estimate(identified_estimand, estimate,
                                   method_name="random_common_cause")
print("\nRefutation test with random common cause:")
print(refute_random)

# Test 2: Replace treatment with random
refute_placebo = model.refute_estimate(identified_estimand, estimate,
                                     method_name="placebo_treatment_refuter")
print("\nPlacebo Treatment test:")
print(refute_placebo)

# Test 3: Add random noise to the outcome
refute_noise = model.refute_estimate(identified_estimand, estimate,
                                 method_name="add_unobserved_common_cause",
                                 confounders_effect_on_treatment="binary_flip",
                                 confounders_effect_on_outcome="linear",
                                 effect_strength_on_treatment=0.1,
                                 effect_strength_on_outcome=0.1)
print("\nUnobserved common cause test:")
print(refute_noise)

# Test 4: Bootstrap
refute_bootstrap = model.refute_estimate(identified_estimand, estimate,
                                      method_name="bootstrap_refuter",
                                      num_simulations=100)
print("\nBootstrap refutation test:")
print(refute_bootstrap)

# Create a summary report of the findings
print("\n=== CAUSAL ANALYSIS SUMMARY ===")
print(f"Dataset: Synthetic sales data with promotion intervention at day 50")
print(f"Number of observations: {len(df)}")
print(f"Observations with promotion: {df['promotion'].sum()}")
print(f"Observations without promotion: {len(df) - df['promotion'].sum()}")

# Calculate average sales before and after promotion
avg_sales_before = df.loc[df['promotion'] == 0, 'sales'].mean()
avg_sales_after = df.loc[df['promotion'] == 1, 'sales'].mean()
naive_diff = avg_sales_after - avg_sales_before

print(f"\nAverage sales before promotion: {avg_sales_before:.4f}")
print(f"Average sales after promotion: {avg_sales_after:.4f}")
print(f"Naive difference: {naive_diff:.4f}")

# Using a regression model to estimate the causal effect
import statsmodels.api as sm

# Add a constant term for the intercept
X = sm.add_constant(df['promotion'])
model = sm.OLS(df['sales'], X).fit()

print("\n=== REGRESSION ANALYSIS ===")
print(model.summary().tables[1])  # Display regression results

print("\nConclusion:")
print(f"The promotional campaign has a statistically significant positive effect on sales.")
print(f"The estimated causal effect from regression is approximately {model.params[1]:.4f} units increase in sales.")
print(f"This matches our expectation of a 10-unit increase that was built into the synthetic data.")
print(f"The p-value of {model.pvalues[1]:.8f} indicates this effect is statistically significant.")

print("\nAnalysis complete. Visualizations saved as 'sales_over_time.png' and 'causal_graph.png'")

