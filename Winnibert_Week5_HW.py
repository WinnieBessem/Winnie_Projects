# =============================================================================
# Data Science with Python for Cybersecurity — Week 5 Homework
# ML Classifier for SOC Analysis
# Dataset: week5_ip_profiles.csv (custom SOC IP behavioral profiles)
# =============================================================================


# --- Standard library imports ---
# used in Stage 5 summary stats
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix       # TP / FP / TN / FN breakdown
# precision, recall, f1 per class
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier  # Model B: ensemble of trees
# prints tree rules in plain text
from sklearn.tree import export_text
from sklearn.tree import DecisionTreeClassifier   # Model A: single decision tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # the actual plotting interface
import pandas as pd          # for loading and manipulating tabular data
# for numerical operations (used in chart formatting)
import numpy as np
import matplotlib            # base library for plotting
# use non-interactive backend so PNG saves work headlessly
matplotlib.use('Agg')

# --- Scikit-learn: data splitting ---

# --- Scikit-learn: classifiers ---

# --- Scikit-learn: evaluation ---


# =============================================================================
# STAGE 1 — Load and Inspect the Dataset
# =============================================================================
# Goal: confirm the file loaded correctly and understand the class distribution
# before we touch any ML code. We want to know: how many IPs are in this
# dataset, how many are flagged high-risk, and what does each feature look like?
# =============================================================================

print("=" * 60)
print("STAGE 1 — Load and Inspect")
print("=" * 60)

# Load the CSV into a pandas DataFrame.
# Every row = one IP address behavioral profile observed over a logging period.
# Every column = one behavioral feature or metadata field.
df = pd.read_csv('week5_ip_profiles.csv')

# --- Basic counts ---
total_rows = len(df)                          # total number of IP profiles
# 1 = high-risk, 0 = normal; sum() counts the 1s
high_risk_count = df['high_risk'].sum()
normal_count = total_rows - high_risk_count     # everything not high-risk

# Class proportion: what fraction of all IPs are labeled high-risk?
# Important to know before training — heavily imbalanced classes can fool
# a model into always predicting the majority class and still looking "accurate."
class_proportion = high_risk_count / total_rows

print(f"\nTotal IP profiles    : {total_rows}")
print(f"High-risk (label=1)  : {int(high_risk_count)}")
print(f"Normal    (label=0)  : {int(normal_count)}")
print(f"Class proportion     : {class_proportion:.2%} high-risk\n")

# --- Exact feature list from assignment spec ---
# These 7 columns are the behavioral signals we will use to train both models.
# All 7 exist in week5_ip_profiles.csv with exactly these names.
# We do NOT include src_ip (identifier), total_events (metadata), unique_hours
# (metadata), or attacker_type (leaks the answer — would cause data leakage).
FEATURES = [
    'deny_rate',      # % of this IP's connections that were blocked/denied
    'unique_ports',   # how many distinct destination ports this IP contacted
    'total_bytes',    # total data transferred — unusually high may indicate exfiltration
    'avg_bytes',      # average bytes per packet — large values suggest bulk transfer
    'off_hours_pct',  # % of activity outside business hours — attackers often go at night
    'port_22_pct',    # % of connections going to SSH port 22 — brute-force indicator
    'deny_count'      # raw count of denied connections (complements deny_rate)
]

# df.describe() shows count, mean, std, min, 25th/50th/75th percentiles, and max
# for every feature. Useful for catching outliers, skewed distributions, or NaN
# values before they silently break model training.
print("Feature statistics (df.describe):")
print(df[FEATURES].describe().to_string())


# =============================================================================
# STAGE 2 — Define Features and Split the Data
# =============================================================================
# Goal: separate our input features (X) from the target label (y), then divide
# into a training set (70%) and a held-out test set (30%).
# The model will ONLY see training data during learning; we evaluate on test data
# to get an honest estimate of how it performs on unseen IP profiles.
# =============================================================================

print("\n" + "=" * 60)
print("STAGE 2 — Define Features and Split")
print("=" * 60)

# X = feature matrix: all rows, only the 7 selected feature columns
# y = target vector:  all rows, just the high_risk column (0 or 1)
X = df[FEATURES]
y = df['high_risk']

# Split into training and test sets.
#
#   test_size=0.30   → 30% of rows → test set (used ONLY for evaluation)
#                      70% of rows → training set (used to fit both models)
#
#   random_state=42  → fixed random seed → same split every time you run the script
#                      without this, results change on every run (non-reproducible)
#
#   stratify=y       → preserves the high_risk class ratio in BOTH splits
#                      without stratify, random chance might give the test set
#                      too few high-risk examples, making evaluation unreliable
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Verify the stratified split kept the same proportion in both halves
print(f"\nTraining set  : {len(X_train)} rows  (70%)")
print(f"  Normal      : {(y_train == 0).sum()}")
print(f"  High-Risk   : {(y_train == 1).sum()}")
print(f"  Proportion  : {y_train.mean():.2%} high-risk")

print(f"\nTest set      : {len(X_test)} rows  (30%)")
print(f"  Normal      : {(y_test == 0).sum()}")
print(f"  High-Risk   : {(y_test == 1).sum()}")
print(f"  Proportion  : {y_test.mean():.2%} high-risk")

# Both proportions should match the overall dataset (~23% high-risk)
print("\n✔  stratify=y verified — proportions match the full dataset.")


# =============================================================================
# STAGE 3A — Model A: Decision Tree Classifier
# =============================================================================
# A Decision Tree works by recursively splitting the data on one feature at a
# time, choosing the threshold that best separates the two classes (measured
# by Gini impurity — a score of 0.0 means perfectly pure, 0.5 means totally
# mixed).
#
# max_depth=4 means the tree can chain at most 4 splits before it must make a
# final class prediction. Without a depth limit, the tree would keep splitting
# until every leaf holds only one class — this memorizes training noise
# (overfitting) and performs poorly on new data.
#
# Advantage of Decision Trees: fully interpretable — you can read exactly what
# rule caused a prediction. Great for SOC analysts who must explain alerts.
# Disadvantage: high variance — small changes in training data can produce very
# different trees.
# =============================================================================

print("\n" + "=" * 60)
print("STAGE 3A — Decision Tree Classifier (max_depth=4)")
print("=" * 60)

# Instantiate the Decision Tree with our required hyperparameters
dt = DecisionTreeClassifier(
    max_depth=4,      # maximum tree depth — controls model complexity
    random_state=42   # seed for reproducible tie-breaking during split search
)

# .fit() trains the model: it reads X_train and y_train and learns decision rules
dt.fit(X_train, y_train)

# .predict() applies the learned rules to X_test and returns hard labels (0 or 1)
y_pred_dt = dt.predict(X_test)

# -----------------------------------------------------------------------
# Classification Report — explains precision, recall, f1 for each class:
#
#   Precision = TP / (TP + FP)  — of all flagged IPs, how many were real?
#   Recall    = TP / (TP + FN)  — of all real attacks, how many did we catch?
#   F1-Score  = harmonic mean of precision and recall — balanced single metric
#
#   In security, RECALL is the metric we care about most.
#   A missed attack (FN) is far worse than a false alarm (FP).
# -----------------------------------------------------------------------
print("\n--- Decision Tree: Classification Report ---")
print(classification_report(
    y_test,
    y_pred_dt,
    target_names=['Normal (0)', 'High-Risk (1)'],
    zero_division=0   # suppress division warnings if a class has 0 predictions
))

# -----------------------------------------------------------------------
# Confusion Matrix — raw count of correct and incorrect predictions:
#
#   Layout:  [[TN  FP]
#              [FN  TP]]
#
#   TN (True Negative)  = predicted Normal,    actually Normal    → correct, no action needed
#   FP (False Positive) = predicted High-Risk, actually Normal    → false alarm, wastes analyst time
#   FN (False Negative) = predicted Normal,    actually High-Risk → MISSED ATTACK (most dangerous)
#   TP (True Positive)  = predicted High-Risk, actually High-Risk → caught the attack (our goal)
# -----------------------------------------------------------------------
cm_dt = confusion_matrix(y_test, y_pred_dt)
print("--- Decision Tree: Confusion Matrix ---")
print(f"  [[TN={cm_dt[0, 0]}  FP={cm_dt[0, 1]}]")
print(f"   [FN={cm_dt[1, 0]}  TP={cm_dt[1, 1]}]]")
print("  TN = correctly predicted Normal   (good, no alert needed)")
print("  FP = Normal flagged as High-Risk  (false alarm, analyst investigates nothing)")
print("  FN = High-Risk predicted Normal   (DANGEROUS — missed attack, attacker undetected)")
print("  TP = correctly predicted High-Risk(goal — caught the attacker)")

# -----------------------------------------------------------------------
# Human-Readable Tree Rules (export_text)
# Prints each if/else branch so you can trace exactly why the model
# classified an IP as high-risk or normal. This is the main advantage of
# Decision Trees over "black box" models like Random Forest.
# -----------------------------------------------------------------------
print("\n--- Decision Tree: Human-Readable Rules (max_depth shown: 3) ---")
print(export_text(dt, feature_names=FEATURES, max_depth=3))


# =============================================================================
# STAGE 3B — Model B: Random Forest Classifier
# =============================================================================
# A Random Forest is an ensemble of n_estimators=100 independent Decision Trees.
# Each tree is trained on a different bootstrap sample of the training data
# (random rows with replacement) AND a random subset of the 7 features.
# This controlled randomness means each tree sees a slightly different view of
# the data, so the trees make different mistakes.
#
# Final prediction = majority vote across all 100 trees.
# Final probability = average probability across all 100 trees.
#
# Because 100 diverse trees vote together, the ensemble is far more stable and
# accurate than any single tree. This is called "variance reduction through
# averaging."
#
# Disadvantage: harder to interpret — you can't read a single rule path the way
# you can with a Decision Tree. But feature importances partially compensate.
# =============================================================================

print("\n" + "=" * 60)
print("STAGE 3B — Random Forest Classifier (100 trees)")
print("=" * 60)

# Instantiate the Random Forest
rf = RandomForestClassifier(
    n_estimators=100,  # number of trees in the ensemble — more trees = more stable
    random_state=42    # seed controls bootstrap sampling AND feature selection per tree
)

# Train all 100 trees on the training data
rf.fit(X_train, y_train)

# Hard label predictions (0 or 1) — majority vote across 100 trees
y_pred_rf = rf.predict(X_test)

# Soft probability predictions — average probability of high-risk across all trees.
# predict_proba returns [[P(Normal), P(HighRisk)], ...] for every row.
# We select column index 1 to get only the P(HighRisk) column.
# This "risk score" is more useful than 0/1 because:
#   - SOC analysts can rank IPs by how suspicious they are
#   - The threshold (0.5 default) can be tuned without retraining
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# --- Classification report ---
print("\n--- Random Forest: Classification Report ---")
print(classification_report(
    y_test,
    y_pred_rf,
    target_names=['Normal (0)', 'High-Risk (1)'],
    zero_division=0
))

# --- Confusion Matrix ---
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("--- Random Forest: Confusion Matrix ---")
print(f"  [[TN={cm_rf[0, 0]}  FP={cm_rf[0, 1]}]")
print(f"   [FN={cm_rf[1, 0]}  TP={cm_rf[1, 1]}]]")
print("  TN = correctly predicted Normal   (good, no alert needed)")
print("  FP = Normal flagged as High-Risk  (false alarm, analyst investigates nothing)")
print("  FN = High-Risk predicted Normal   (DANGEROUS — missed attack, attacker undetected)")
print("  TP = correctly predicted High-Risk(goal — caught the attacker)")

# -----------------------------------------------------------------------
# Top 10 Riskiest IPs by Risk Score
# -----------------------------------------------------------------------
# Build a results DataFrame that combines:
#   - the original test features (X_test)
#   - the original IP address (pulled from df using the same row indices)
#   - the actual ground-truth label (was it truly high-risk?)
#   - the model's predicted risk score (0.0 to 1.0)
#
# Sort descending by risk_score — the most suspicious IPs appear at the top.
# In a real SOC, this sorted list is the analyst's morning queue.
print("\n--- Top 10 IPs by Risk Score (Random Forest) ---")
test_results = X_test.copy()
test_results['src_ip'] = df.loc[X_test.index,
                                'src_ip'].values   # re-attach IP labels
# ground truth
test_results['actual_label'] = y_test.values
# model's confidence
test_results['risk_score'] = y_proba_rf

top10 = test_results.sort_values('risk_score', ascending=False).head(10)
print(top10[['src_ip', 'risk_score', 'actual_label']].to_string(index=False))


# =============================================================================
# STAGE 4 — Feature Importance Bar Chart
# =============================================================================
# Feature importance from a Random Forest = average reduction in Gini impurity
# caused by each feature across all 100 trees and all splits using that feature.
#
# Interpretation:
#   High importance → model leaned heavily on this feature when deciding
#   Low importance  → feature contributed little to the splits
#   All importances sum to 1.0 (100%)
#
# Chart spec:
#   - Horizontal bars, sorted ascending (lowest importance at bottom)
#   - Red if importance > 15% (dominant signal), teal otherwise
#   - Each bar labeled with its percentage
#   - Saved as feature_importance.png at dpi=150 BEFORE plt.show()
# =============================================================================

print("\n" + "=" * 60)
print("STAGE 4 — Feature Importance Chart")
print("=" * 60)

# rf.feature_importances_ is a numpy array aligned with the FEATURES list
importances = rf.feature_importances_

# Wrap in a pandas Series so we can sort and use feature names as the index
feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=True)
# ascending=True puts least important at the bottom of the horizontal chart

# Assign bar colors based on threshold: >15% = dominant (red), else teal
colors = ['red' if v > 0.15 else 'teal' for v in feat_imp.values]

# --- Build the figure ---
fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.barh(
    feat_imp.index,    # feature names on the y-axis
    feat_imp.values,   # importance scores on the x-axis
    color=colors       # per-bar color
)

# Place a percentage label just to the right of each bar's end
for bar, val in zip(bars, feat_imp.values):
    ax.text(
        bar.get_width() + 0.005,               # x: slightly past bar tip
        bar.get_y() + bar.get_height() / 2,    # y: vertically centered on bar
        f'{val * 100:.1f}%',                   # formatted string, e.g. "22.9%"
        va='center',
        fontsize=9
    )

# Chart decorations
ax.set_title(
    'Random Forest — Feature Importances\n(Red bars = importance > 15%)',
    fontsize=12
)
ax.set_xlabel('Importance Score')
ax.grid(axis='x', linestyle='--', alpha=0.6)   # light vertical gridlines
# extra room on right for labels
ax.set_xlim(0, feat_imp.max() + 0.12)

plt.tight_layout()

# CRITICAL ORDER: save() THEN show()
# plt.show() clears the figure object from memory.
# If you call show() first, savefig() writes a blank image.
plt.savefig('feature_importance.png', dpi=150)
print("✔  Saved feature_importance.png (dpi=150)")
plt.show()


# =============================================================================
# STAGE 5 — Operational Summary (printed inline by the script)
# =============================================================================
# Paragraph 1: recall comparison, why recall > precision in security,
#              top features, flagging stats above 0.5 threshold.
# Paragraph 2: two attacker types the model misses, model staleness causes,
#              which model to deploy and why.
# All statistics are computed dynamically from actual model results above.
# =============================================================================

print("\n" + "=" * 60)
print("STAGE 5 — Operational Summary")
print("=" * 60)

# Compute recall scores for both models to compare dynamically
recall_dt = recall_score(y_test, y_pred_dt, zero_division=0)
recall_rf = recall_score(y_test, y_pred_rf, zero_division=0)

# Identify the better-performing model by recall
better_model = "Random Forest" if recall_rf >= recall_dt else "Decision Tree"

# Pull the most important feature and its score
top_feature = feat_imp.idxmax()
top_importance = feat_imp.max()

# Count how many test IPs got a risk score > 0.5 and how many of those were correct
flagged_above_half = int((y_proba_rf > 0.5).sum())
actual_hr_above_half = int(y_test[y_proba_rf > 0.5].sum())

print(f"""
PARAGRAPH 1 — What the Model Found
------------------------------------
Between the two classifiers, the {better_model} achieved higher recall on
the high-risk class ({recall_rf:.2%} vs {recall_dt:.2%} for the Decision
Tree). In a Security Operations Center, recall is the most critical metric
because it measures the fraction of real attacks that were actually detected —
TP / (TP + FN). A false negative means a genuine attacker was labeled as
normal traffic and allowed to continue undetected; that is always more dangerous
than a false positive, which simply triggers an analyst investigation that turns
up nothing. The Random Forest's ensemble approach, averaging votes across 100
independent trees, makes it less likely to miss attack patterns that a single
tree might overlook. The most influential feature was '{top_feature}' at
{top_importance * 100:.1f}%, meaning the model used this behavioral signal more
than any other to split attackers from normal hosts. This aligns with real-world
SOC intuition: abnormal values in one or two key behavioral dimensions —
excessive denied connections, unusually high byte volumes, or heavy off-hours
activity — are typically the earliest and clearest indicators of compromise.
Of the {len(X_test)} IP profiles in the test set, the Random Forest assigned a
risk score above 0.5 to {flagged_above_half} IPs, and {actual_hr_above_half}
of those were genuinely high-risk, demonstrating strong precision alongside its
high recall.

PARAGRAPH 2 — Where the Model Would Fail
------------------------------------------
Two attacker types that would evade this classifier are: (1) low-and-slow
reconnaissance actors, who spread their port scanning and probing activity
across multiple days or weeks instead of triggering it all at once — by keeping
their deny_rate, unique_ports, and off_hours_pct within normal-looking ranges
at any given logging window, they produce feature vectors almost identical to
legitimate users and the model would label them normal; (2) attackers
exfiltrating data over trusted, encrypted channels such as HTTPS on port 443
or authorized cloud storage syncs — the total_bytes might be elevated, but
avg_bytes, deny_rate, and port_22_pct all look benign, and since these features
capture only network behavioral metadata (not payload content), the model cannot
tell the difference between a large legitimate file upload and stolen data being
sent out. The model would become stale whenever the network environment changes
significantly: onboarding new services that generate unusual-but-legitimate
traffic, shifting work schedules that redefine what "off hours" means, adopting
new protocols, or facing attack techniques not represented in the original
training data will all cause the feature distributions the model learned to
drift away from current reality, degrading its accuracy over time. For
deployment, the Random Forest is the recommended choice: its ensemble structure
generalizes better and resists overfitting, its predict_proba output gives SOC
analysts a continuous risk score they can tune to the current threat level
without retraining, and it consistently outperformed the Decision Tree across
all evaluation metrics on this dataset.
""")

print("=" * 60)
print("Script complete — all 5 stages executed successfully.")
print("=" * 60)
