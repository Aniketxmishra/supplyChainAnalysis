# Supply Chain Causal Analysis for Promotional Impact

## 🎯 Project Overview

This project demonstrates causal inference techniques for supply chain optimization, focusing on measuring the impact of promotional campaigns on sales. Using synthetic data with a known ground truth, we show how causal analysis can quantify the effect of interventions and provide actionable business insights.

## 🏆 Key Results Achieved

- **11.22 Units Promotional Effect** - Clear, statistically significant causal impact
- **23.0% Sales Increase** - Substantial boost from promotional activities
- **p < 0.00001** - Highly statistically significant findings
- **Robust Validation** - Multiple refutation tests confirm results

## 🔬 Technical Excellence

### Core Technologies
- **DoWhy** - Causal inference framework for rigorous effect estimation
- **statsmodels** - Statistical modeling for regression analysis
- **matplotlib/seaborn** - Comprehensive data visualization
- **scipy** - Statistical testing and validation
- **pandas/numpy** - Data manipulation and analysis

### Advanced Features
- Causal effect estimation with multiple methodologies
- Intervention analysis with statistical validation
- Comprehensive visual representation of causal effects
- Robust statistical testing and refutation

## 📊 Business Applications

### Supply Chain Optimization
- **Promotional Impact Analysis**: Quantified +11.22 units effect of marketing campaigns
- **ROI Calculation**: 23.0% increase in sales from promotions
- **Strategic Planning**: Data-driven recommendations for future campaigns

### Key Business Questions Answered
1. What is the causal effect of promotions on sales?
2. Is the promotional effect statistically significant?
3. How can we visualize and communicate the effect to stakeholders?
4. How robust is our causal analysis to potential confounders?

## 🚀 Quick Start

### Installation
```
pip install pandas numpy matplotlib seaborn scipy statsmodels dowhy networkx
```

### Run Analysis Scripts
```
python causal_analysis.py
python visualize_results.py
```

## 📁 Project Structure

```
supplyChainAnalysis/
├── causal_analysis.py        # Main causal analysis script
├── visualize_results.py      # Detailed visualization script
├── sales_over_time.png       # Time series visualization
├── causal_graph.png          # Causal diagram
├── promotion_effect_visualization.png  # Comprehensive visual analysis
├── sales_distribution_comparison.png   # Distribution comparison
└── README.md                 # This file
```

## 🔍 Methodology

### 1. Causal Model Construction
- Defined a clear causal graph: Promotion → Sales
- Identified causal estimand using DoWhy framework
- Applied rigorous causal assumptions

### 2. Effect Estimation
- Linear regression causal estimator: 11.22 units effect
- Verified with multiple estimation methods
- Statistical significance testing: p < 0.00001

### 3. Robustness Testing
- Random common cause refutation
- Placebo treatment testing
- Unobserved confounder sensitivity analysis
- Bootstrap resampling validation

## 📈 Key Findings

### Promotional Impact
- **Immediate Effect**: +11.22 units increase per promotion
- **Percentage Impact**: 23.0% sales increase from baseline
- **Statistical Validation**: Highly significant results (p < 0.00001)

### Visual Analysis
- Clear before/after comparison shows consistent effect
- Distribution shift confirms systematic impact
- Time series analysis reveals sustained improvement
- Box plots demonstrate reduced variance in promotional period

## 🛠️ Technical Implementation

### Causal Analysis Pipeline
```python
# Create a causal model using DoWhy
model = CausalModel(
    data=df,
    treatment='promotion',
    outcome='sales'
)

# Identify the causal effect
identified_estimand = model.identify_effect()

# Estimate the causal effect
estimate = model.estimate_effect(identified_estimand,
                                method_name="backdoor.linear_regression")

# Refute the estimate
refute_random = model.refute_estimate(identified_estimand, estimate,
                                   method_name="random_common_cause")
```

### Visualization Framework
```python
# Create comprehensive before/after visualization
plt.figure(figsize=(15, 10))

# Bar chart comparing averages
avg_sales = df.groupby("promotion_category")["sales"].mean()
avg_sales.plot(kind="bar", color=["#ff9999", "#66b3ff"])

# Time series with clear before/after
plt.plot(df["date"], df["sales"])
plt.axvline(x=df["date"][49], color="red", linestyle="--")

# Statistical annotations
stats_text = (
    f"• Before Promotion: Mean = {no_promo['sales'].mean():.2f}\n"
    f"• After Promotion: Mean = {with_promo['sales'].mean():.2f}\n"
    f"• Difference: {with_promo['sales'].mean() - no_promo['sales'].mean():.2f} units"
)
```

## 🧑‍💻 Author

**Aniket Mishra**
- GitHub: [@Aniketxmishra](https://github.com/Aniketxmishra)
- Email: aniketmishrarr@gmail.com
- Project: [Supply Chain Causal Analysis](https://github.com/Aniketxmishra/supplyChainAnalysis)

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Built using the DoWhy causal inference framework
- Inspired by real-world supply chain optimization challenges
- Demonstrates rigorous causal analysis methodology

---

*This project demonstrates the application of causal inference techniques to measure the impact of promotional campaigns on sales, providing actionable insights for supply chain optimization and marketing strategy.*

