
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, pearsonr
from pathlib import Path

# 1. Load and validate the dataset
file_path = Path("survey.csv")

if not file_path.exists():
    raise FileNotFoundError(
        "survey.csv not found. Keep it in the same folder as this program."
    )

df = pd.read_csv(file_path)

required_columns = [
    "Platform", "Purpose",
    "Daily_Hours", "Checks_Per_Day"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# 2. Clean the data
for col in ["Platform", "Purpose"]:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace("", pd.NA)

for col in ["Daily_Hours", "Checks_Per_Day"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_columns)

# Remove impossible values
df = df[
    df["Daily_Hours"].between(0, 24) &
    (df["Checks_Per_Day"] >= 0)
].copy()

if len(df) < 2:
    raise ValueError("At least two valid responses are required.")

print("=" * 45)
print("SOCIAL MEDIA USAGE ANALYSIS")
print("=" * 45)
print("Total valid responses:", len(df))

# 3. Descriptive statistics
print("\n1. DAILY USAGE STATISTICS")

hours = df["Daily_Hours"]

print(f"Mean: {hours.mean():.2f} hours")
print(f"Median: {hours.median():.2f} hours")
print(f"Standard deviation: {hours.std():.2f} hours")

# 4. Platform frequency and percentage
print("\n2. PLATFORM PREFERENCE")

platform_counts = df["Platform"].value_counts()
platform_percentages = (
    df["Platform"].value_counts(normalize=True) * 100
)

platform_summary = pd.DataFrame({
    "Students": platform_counts,
    "Percentage": platform_percentages.round(2)
})

print(platform_summary)

platform_summary.to_csv("platform_summary.csv")

# 5. Platform preference bar chart
plt.figure(figsize=(8, 5))

ax = platform_counts.plot(
    kind="bar", edgecolor="black"
)

plt.title("Social Media Platform Preferences")
plt.xlabel("Platform")
plt.ylabel("Number of Students")
plt.xticks(rotation=30, ha="right")

for bar in ax.patches:
    ax.annotate(
        f"{int(bar.get_height())}",
        (bar.get_x() + bar.get_width() / 2,
         bar.get_height()),
        ha="center", va="bottom"
    )

plt.tight_layout()
plt.savefig("platform_preferences.png", dpi=300)
plt.show()

# 6. Daily usage histogram
plt.figure(figsize=(8, 5))

plt.hist(hours, bins=5, edgecolor="black")

plt.title("Distribution of Daily Social Media Usage")
plt.xlabel("Hours per Day")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("daily_usage.png", dpi=300)
plt.show()

# 7. Cross-tabulation
print("\n3. PLATFORM VS PURPOSE")

cross_tab = pd.crosstab(
    df["Platform"], df["Purpose"]
)

print(cross_tab)
cross_tab.to_csv("platform_vs_purpose.csv")

# 8. Chi-square test
print("\n4. CHI-SQUARE TEST")

if cross_tab.shape[0] >= 2 and cross_tab.shape[1] >= 2:
    chi2, p_value, dof, expected = chi2_contingency(cross_tab)

    print(f"Chi-square statistic: {chi2:.3f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p_value:.4f}")

    if (expected < 5).any():
        print("Warning: Some expected counts are below 5.")
        print("Interpret the chi-square result cautiously.")
    else:
        print("All expected counts are at least 5.")

    if p_value < 0.05:
        print("A statistically significant association was found.")
    else:
        print("No statistically significant association was detected.")
else:
    print("Insufficient categories for the chi-square test.")

# 9. Correlation analysis
print("\n5. USAGE HOURS VS CHECKS PER DAY")

checks = df["Checks_Per_Day"]

if len(df) >= 3 and hours.nunique() > 1 and checks.nunique() > 1:
    r, p = pearsonr(hours, checks)

    print(f"Pearson correlation: {r:.3f}")
    print(f"P-value: {p:.4f}")

    if r > 0:
        print("The variables have a positive linear relationship.")
    elif r < 0:
        print("The variables have a negative linear relationship.")
    else:
        print("No linear correlation was observed.")

    if p < 0.05:
        print("The correlation is statistically significant.")
    else:
        print("The correlation is not statistically significant.")
else:
    print("Not enough variation or valid observations for correlation.")

print("\nAnalysis completed successfully.")
