# 💻 Laptop Price Prediction ML

Predictive model that estimates laptop prices based on technical specifications (RAM, CPU, GPU, storage, screen size, etc.), built with a business-oriented lens: not just "train a model," but understand *what actually drives price* and turn that into something usable.

## Objective

Build a regression model to estimate the price of a laptop from its specs, identify which features have the biggest impact on price, and provide a simple inference function that estimates the price of a new laptop given its configuration.

## Dataset

[Laptop Pricing Dataset](https://www.kaggle.com/datasets) — public dataset from Kaggle containing 238 laptops with features such as manufacturer, category, GPU, OS, CPU core, screen size, CPU frequency, RAM, SSD storage, weight, and price.

## Tools & Libraries

- Python (pandas, numpy)
- scikit-learn (Linear Regression, Random Forest)
- matplotlib / seaborn (visualization)
- joblib (model persistence)
- Kaggle Notebooks

## 🔍 Process

1. **Exploratory Data Analysis (EDA)** — distribution of prices, price by category/manufacturer, correlation between specs and price.
2. **Data cleaning** — null check, outlier inspection (kept, since high-end Workstation/Gaming laptops are legitimate, not data errors).
3. **Modeling** — trained and compared Linear Regression (baseline) vs Random Forest.
4. **Evaluation** — MAE, RMSE, R² translated into business terms (average dollar error).
5. **Inference** — a reusable function to estimate the price of any new laptop configuration.

## 📈 Results

| Model             | MAE ($) | RMSE ($) | R²    |
|-------------------|---------|----------|-------|
| Linear Regression | 331.77  | 412.09   | 0.263 |
| Random Forest     | 284.35  | 372.45   | 0.398 |

**Random Forest outperformed the linear baseline** on all three metrics, reducing average prediction error by ~$47 and explaining a higher share of price variance.

## 💡 Key Business Insights

- [Completa aquí con tu feature importance real, ej.: "RAM and CPU frequency were the strongest price drivers, followed by category (Workstation/Gaming command a premium)."]
- The model's ~$284 average error represents about X% of the average laptop price — usable as a first-pass price estimate, not a final pricing decision.
- Dataset size (238 rows) limits model robustness; a larger dataset and inclusion of `Manufacturer` as a feature would likely improve accuracy further.

## 🚀 How to Use

```python
predict_laptop_price(
    category=1,          # 1=Gaming, 2=Netbook, 3=Notebook, 4=Ultrabook, 5=Workstation
    gpu=3,                # 1=AMD, 2=Intel, 3=NVidia
    os_type=1,            # 1=Windows, 2=Linux
    cpu_core=7,            # 3=i3, 5=i5, 7=i7
    screen_size_cm=39.6,
    cpu_frequency=2.8,
    ram_gb=16,
    storage_gb_ssd=512,
    weight_kg=2.3
)
# -> Estimated price: $X
```
