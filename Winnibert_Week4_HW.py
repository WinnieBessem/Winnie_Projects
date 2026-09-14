# =============================================================================
# DATA SCIENCE WITH PYTHON FOR CYBERSECURITY — WEEK 4 HOMEWORK
# SOC Analyst: Statistical Anomaly Detection
# File: lastname_week4_hw.py
# =============================================================================

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
# Used for Chart 3 heatmap — gives us gridlines and better color control
import seaborn as sns

plt.style.use("dark_background")
# Apply a consistent dark theme so all charts share the same visual style


# =============================================================================
# STAGE 1 — COLLECT
# Load the dataset and inspect it for data quality problems
# =============================================================================

# Load the raw CSV — no type conversions yet, we want to see problems as-is
df = pd.read_csv('week4_firewall_log.csv')

print("=" * 60)
print("STAGE 1 — COLLECT: Initial Data Inspection")
print("=" * 60)

# --- Inspection Command 1: Shape ---
# .shape returns a tuple (rows, columns) — quick sanity check that the file loaded fully
print("\n[1] Shape (rows, columns):")
print(df.shape)
# FINDING: 3,864 rows and 8 columns — matches the expected dataset size.
# If rows were missing it would indicate a truncated file or encoding error.

# --- Inspection Command 2: Data Types ---
# .dtypes tells us how pandas interpreted each column.
# 'object' dtype means pandas stored the column as strings — it could not infer a number or date.
print("\n[2] Data Types:")
print(df.dtypes)
# FINDING: 'timestamp' is object (string) — pandas did not auto-parse it as datetime
#           because the file contains two different date formats mixed together.
# FINDING: 'bytes_sent' is object (string) — some values have a 'KB' suffix which
#           prevents pandas from reading the column as a number.
# FINDING: 'dst_port' reads as int64 in this environment, but may load as object
#           elsewhere if any non-numeric values are present. We will cast it explicitly
#           in Stage 2 to guarantee consistent behavior.

# --- Inspection Command 3: Info ---
# .info() prints a concise summary: column names, non-null counts, and dtypes.
# The non-null count is the fastest way to spot columns with missing data.
print("\n[3] Info:")
print(df.info())
# FINDING: 'country' shows 3,644 non-null out of 3,864 total rows.
#           That means 220 entries have no country value — roughly 6% of the dataset.
#           All other columns are fully populated (3,864 non-null each).

# --- Inspection Command 4: Describe ---
# .describe() computes count, mean, std, min, quartiles, and max for numeric columns only.
# Columns stored as strings (object dtype) are silently excluded — this is a key warning sign.
print("\n[4] Describe (numeric columns):")
print(df.describe())
# FINDING: 'bytes_sent' is completely missing from this output even though it should be
#           numeric. This is because pandas stored it as an object due to the 'KB' suffix
#           on some values. We lose all statistical insight into byte volume until this is fixed.
# FINDING: Only 'dst_port' appears in the numeric summary, which is not very useful on its own.

# --- Inspection Command 5: Null Counts ---
# .isnull().sum() counts missing values in every column at once.
# This is faster than reading .info() output for large datasets.
print("\n[5] Null Value Counts:")
print(df.isnull().sum())
# FINDING: 'country' has exactly 220 null values. All other 7 columns have 0 nulls.
# Null countries must be filled before analysis — groupby and value_counts will
# silently drop null rows, which would skew our country-based findings.

# --- Summary of All Data Quality Problems Found ---
print("\n" + "=" * 60)
print("DATA QUALITY PROBLEMS IDENTIFIED:")
print("=" * 60)
print("""
Problem 1 - TIMESTAMP: Mixed date formats (ISO and US slash format).
            Examples: '2024-03-01T13:18:00' and '3/5/2024 13:18'.
            Fix: pd.to_datetime(format='mixed')

Problem 2 - BYTES_SENT: Some values stored as strings with 'KB' suffix.
            Examples: '7KB', '2KB', '0KB'.
            Fix: Strip 'KB', multiply by 1024, convert to integer.

Problem 3 - SEVERITY: 8 inconsistent casing variants found:
            'MEDIUM', 'high', 'medium', 'critical', 'low', 'Low', 'HIGH', 'Critical'.
            Fix: .str.lower().str.strip()

Problem 4 - COUNTRY: 220 null values (~6% of rows).
            Fix: Fill with 'UNKNOWN'.

Problem 5 - DUPLICATES: 20 duplicate rows from a logging double-write.
            Fix: .drop_duplicates()

Problem 6 - DST_PORT: Stored as object in some environments instead of integer.
            Fix: Cast to int after cleaning.
""")


# =============================================================================
# STAGE 2 — CLEAN
# Fix all six data quality problems identified in Stage 1
# =============================================================================

print("=" * 60)
print("STAGE 2 — CLEAN: Fixing Data Quality Problems")
print("=" * 60)

# Capture row count before any changes so we can report how many duplicates were removed
rows_before = len(df)
print(f"\nRow count BEFORE cleaning: {rows_before}")

# Fix 1 — Parse timestamps with mixed format support
# format='mixed' tells pandas to try multiple formats row by row instead of assuming one format.
# Without this, pd.to_datetime() would fail or silently produce NaT values on rows that
# don't match the inferred format.
df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
print("\n[Fix 1] Timestamps parsed as datetime — done.")

# Fix 2 — Strip 'KB' suffix from bytes_sent and convert to integer
# We use a custom function + .apply() because the column has two different formats
# ('7KB' and '1110') and we need different logic for each case.
# .apply() passes one value at a time into the function and rebuilds the column with the results.


def fix_bytes(val):
    """
    Convert bytes_sent values to a plain integer.
    - If the value ends with 'KB', strip the suffix and multiply by 1024 to get bytes.
    - Otherwise, convert directly to int.
    This handles both '7KB' (kilobyte string) and '1110' (plain numeric string).
    """
    val = str(val).strip()           # Remove any leading/trailing whitespace first
    if val.endswith('KB'):
        return int(val.replace('KB', '')) * 1024   # Convert kilobytes to bytes
    return int(val)                  # Plain numeric string — just cast to int


df['bytes_sent'] = df['bytes_sent'].apply(fix_bytes)
print("[Fix 2] bytes_sent converted to integer (KB values multiplied by 1024) — done.")

# Fix 3 — Normalize severity casing to lowercase
# .str.lower() converts all characters to lowercase so 'HIGH', 'High', and 'high' all match.
# .str.strip() removes any invisible whitespace that could cause groupby mismatches.
# Order matters: strip after lower to catch any whitespace that was hiding under caps.
df['severity'] = df['severity'].str.lower().str.strip()
print("[Fix 3] severity normalized to lowercase — done.")

# Fix 4 — Fill null country values with 'UNKNOWN'
# .fillna() replaces every NaN in the column with the given string.
# Using 'UNKNOWN' keeps all rows in the dataset — we do not drop rows with missing countries
# because those rows still have valid IP, port, action, and bytes data for analysis.
df['country'] = df['country'].fillna('UNKNOWN')
print("[Fix 4] Null country values filled with 'UNKNOWN' — done.")

# Fix 5 — Remove duplicate rows
# .drop_duplicates() removes any row where ALL column values exactly match a previous row.
# These 20 rows came from a logging double-write and would inflate event counts for
# affected IPs, making them look more active than they actually were.
df = df.drop_duplicates()
print("[Fix 5] Duplicate rows removed — done.")

# Fix 6 — Ensure dst_port is integer
# Even if pandas read it as int64 this time, an explicit cast guarantees consistent behavior
# across different environments and prevents accidental string comparisons on port numbers.
df['dst_port'] = df['dst_port'].astype(int)
print("[Fix 6] dst_port cast to integer — done.")

rows_after = len(df)
print(f"\nRow count AFTER cleaning : {rows_after}")
print(f"Rows removed             : {rows_before - rows_after} (duplicates)")


# =============================================================================
# STAGE 3 — ANALYZE (Statistical Detection)
# Build IP profiles and apply z-score and IQR anomaly detection
# =============================================================================

print("\n" + "=" * 60)
print("STAGE 3 — ANALYZE: Statistical Anomaly Detection")
print("=" * 60)

# --- Required Analysis A: Descriptive Statistics ---

print("\n--- Analysis A: IP Profile & Descriptive Statistics ---\n")

# Build one row per source IP summarizing their behavior over the full 30-day window.
# groupby('src_ip') splits the dataframe into one group per unique IP address.
# .agg() computes multiple summary statistics for each group in a single pass.
ip_profile = df.groupby('src_ip').agg(
    # Total log entries for this IP
    total_events=('src_ip', 'count'),
    # How many connections were blocked
    deny_count=('action', lambda x: (x == 'DENY').sum()),
    # How many distinct ports it contacted
    unique_ports=('dst_port', 'nunique'),
    # Total data volume sent
    total_bytes=('bytes_sent', 'sum'),
    # reset_index() turns src_ip from the group key back into a regular column
).reset_index()

# deny_rate = fraction of this IP's total traffic that was denied.
# An IP with a high deny_rate is being blocked most of the time — strong attacker signal.
# An IP with a low deny_rate but high total_bytes may be exfiltrating allowed traffic.
ip_profile['deny_rate'] = ip_profile['deny_count'] / ip_profile['total_events']

print("IP Profile (first 5 rows):")
print(ip_profile.head())

print("\nDescriptive Statistics for IP Profile:")
print(ip_profile.describe())

# Pre-compute BOTH detection thresholds here and print them BEFORE applying either method.
# The assignment requires showing the thresholds separately so the reader can verify the logic.

# Z-score threshold: mean + 2 standard deviations marks roughly the top 2.5% of a normal curve.
# We use 2.5 as our actual flag threshold (slightly more conservative than the printed reference).
deny_mean = ip_profile['deny_count'].mean()
deny_std = ip_profile['deny_count'].std()
# Reference point — actual flag threshold is z > 2.5
zscore_threshold = deny_mean + 2 * deny_std

# IQR upper fence: Q3 + 1.5 * IQR is the standard Tukey fence for outlier detection.
# It is based entirely on the middle 50% of the data, so extreme outliers do not distort it.
Q1 = ip_profile['deny_count'].quantile(
    0.25)   # Value below which 25% of IPs fall
Q3 = ip_profile['deny_count'].quantile(
    0.75)   # Value below which 75% of IPs fall
IQR = Q3 - Q1                                   # The spread of the middle 50%
# Any IP above this is considered an outlier
upper_fence = Q3 + 1.5 * IQR

print(
    f"\nZ-Score threshold (mean + 2\u03c3) for deny_count : {zscore_threshold:.2f}")
print(f"IQR Upper Fence (Q3 + 1.5\u00d7IQR) for deny_count: {upper_fence:.2f}")

print("""
INTERPRETATION:
The average IP generated about 33 DENY events over 30 days, but the standard
deviation of 74 is more than twice the mean. This tells us the distribution is
heavily right-skewed — a small number of IPs are generating far more blocks than
the rest of the network. The maximum deny_count of 401 is over 12x the mean,
which is a strong signal that at least one IP is behaving very differently from
the baseline.
""")


# --- Required Analysis B: Z-Score Detection ---

print("--- Analysis B: Z-Score Detection ---\n")

# stats.zscore() computes how many standard deviations each value is from the column mean.
# Formula: z = (x - mean) / std
# A z-score of 0 means exactly average. A z-score of 2.5 means 2.5 standard deviations above average.
# We compute z-scores independently on two different metrics:
#   deny_zscore  — flags IPs that were blocked unusually often
#   bytes_zscore — flags IPs that transferred an unusually large volume of data
# These two metrics catch different threat types: deny_zscore catches brute force / scanners,
# bytes_zscore catches data exfiltration that may never have been blocked at all.
ip_profile['deny_zscore'] = stats.zscore(ip_profile['deny_count'])
ip_profile['bytes_zscore'] = stats.zscore(ip_profile['total_bytes'])

# Flag with OR logic — an IP is suspicious if it stands out on EITHER metric.
# Using AND would be too strict and would miss the exfiltration case (0 denies, huge bytes).
ip_profile['zscore_alert'] = (
    (ip_profile['deny_zscore'] > 2.5) |
    (ip_profile['bytes_zscore'] > 2.5)
)

# Sort by deny_zscore descending so the most extreme deny offenders appear first
zscore_flagged = ip_profile[ip_profile['zscore_alert']].copy()
zscore_flagged = zscore_flagged.sort_values('deny_zscore', ascending=False)

print("IPs flagged by Z-Score (deny_zscore > 2.5 OR bytes_zscore > 2.5):")
print(zscore_flagged[[
    'src_ip', 'deny_count', 'deny_zscore', 'total_bytes', 'bytes_zscore'
]].to_string(index=False))


# --- Required Analysis C: IQR Detection ---

print("\n--- Analysis C: IQR Detection ---\n")

# Print all four IQR values clearly labeled — required by the assignment rubric.
# These were already computed above before either detection method was applied.
print(f"Q1  (25th percentile of deny_count) : {Q1}")
print(f"Q3  (75th percentile of deny_count) : {Q3}")
print(f"IQR (Q3 - Q1)                       : {IQR}")
print(f"Upper Fence (Q3 + 1.5 \u00d7 IQR)        : {upper_fence}")

# Flag any IP whose deny_count exceeds the IQR upper fence.
# Unlike z-score, this threshold is not affected by the extreme outliers
# because it is derived from Q1 and Q3 — the middle 50% — not the mean.
ip_profile['iqr_alert'] = ip_profile['deny_count'] > upper_fence

iqr_flagged = ip_profile[ip_profile['iqr_alert']].copy()
iqr_flagged = iqr_flagged.sort_values('deny_count', ascending=False)

print("\nIPs flagged by IQR (deny_count above upper fence):")
print(iqr_flagged[['src_ip', 'deny_count']].to_string(index=False))

# Combined alert: an IP is flagged if EITHER method caught it.
# This is the most inclusive approach — we would rather investigate a false positive
# than miss a real threat. The investigation summary in Stage 5 then determines
# whether each flagged IP is truly suspicious.
ip_profile['combined_alert'] = ip_profile['zscore_alert'] | ip_profile['iqr_alert']

print("\n--- Agreement Between Methods ---")

print("\nIPs flagged by BOTH z-score AND IQR (highest confidence):")
both = ip_profile[ip_profile['zscore_alert'] & ip_profile['iqr_alert']]
print(both[['src_ip', 'deny_count', 'deny_zscore']].to_string(index=False))

print("\nIPs flagged by IQR ONLY (not by z-score):")
iqr_only = ip_profile[ip_profile['iqr_alert'] & ~ip_profile['zscore_alert']]
print(iqr_only[['src_ip', 'deny_count', 'deny_zscore']].to_string(index=False))

print("\nIPs flagged by Z-SCORE ONLY (not by IQR):")
z_only = ip_profile[ip_profile['zscore_alert'] & ~ip_profile['iqr_alert']]
print(z_only[['src_ip', 'deny_count', 'deny_zscore']].to_string(index=False))

print("""
NOTE ON DIFFERENCES:
172.16.0.88 was flagged by IQR but NOT by z-score. This is because two extreme
outliers (172.16.0.131 and 172.16.0.5) inflate the mean and standard deviation,
compressing z-scores for moderately anomalous IPs. IQR is resistant to outliers,
making it better at catching IPs that are unusual relative to the bulk of the
network — even when extreme cases exist.

10.99.0.1 was flagged by z-score on total_bytes but NOT by IQR on deny_count.
This IP had zero denies but sent over 51 MB of data. IQR on deny_count would
never detect this kind of anomaly — it takes a different metric (bytes) and
a different method (z-score) to surface it.
""")


# =============================================================================
# STAGE 4 — VISUALIZE
# Three charts: scatter plot, bar chart, hourly heatmap
# All charts saved as PNG before show()
# =============================================================================

print("=" * 60)
print("STAGE 4 — VISUALIZE: Generating Charts")
print("=" * 60)

# --- Chart 1: Scatter plot — total_events vs deny_count ---
# Purpose: show each IP as a point so we can spot the outliers visually.
# Red points = flagged by combined alert. Blue points = normal IPs.
# The orange dashed line marks the IQR upper fence so readers can see
# which IPs are above the threshold at a glance.

fig, ax = plt.subplots(figsize=(10, 6))

# Map the combined_alert boolean to colors — True (flagged) = red, False (normal) = steelblue
colors = ip_profile['combined_alert'].map({True: 'red', False: 'steelblue'})

ax.scatter(
    ip_profile['total_events'],
    ip_profile['deny_count'],
    c=colors,
    alpha=0.7,           # Slight transparency so overlapping points are visible
    edgecolors='black',
    linewidths=0.5,
    s=60                 # Marker size in points squared
)

# Draw a horizontal reference line at the IQR upper fence
ax.axhline(
    y=upper_fence, color='orange', linestyle='--',
    linewidth=1.5, label=f'IQR Upper Fence ({upper_fence:.1f})'
)

# Label each flagged IP directly on the chart so readers know which IP is which
# without having to look up coordinates manually
for _, row in ip_profile[ip_profile['combined_alert']].iterrows():
    ax.annotate(
        row['src_ip'],
        xy=(row['total_events'], row['deny_count']),
        # Offset label slightly so it does not overlap the dot
        xytext=(5, 5), textcoords='offset points',
        fontsize=7, color='darkred'
    )

ax.set_title(
    'Total Events vs Deny Count by Source IP\n(Red = Flagged, Orange Dashed = IQR Fence)',
    fontsize=13
)
ax.set_xlabel('Total Events (30 days)', fontsize=11)
ax.set_ylabel('Deny Count', fontsize=11)
ax.legend()
plt.tight_layout()
# Must save BEFORE show() or the file will be blank
plt.savefig('chart1_scatter.png', dpi=150)
plt.show()
print("[Chart 1] Saved as chart1_scatter.png")


# --- Chart 2: Bar chart — top 10 IPs by deny_count ---
# Purpose: rank the most-blocked IPs side by side so the magnitude difference is clear.
# Color coding matches Chart 1: red = flagged, blue = not flagged.
# The IQR fence line gives a visual cutoff to separate normal from anomalous.

# Select the 10 IPs with the highest deny counts for a focused view
top10 = ip_profile.nlargest(10, 'deny_count').copy()
bar_colors = top10['combined_alert'].map({True: 'red', False: 'steelblue'})

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(top10['src_ip'], top10['deny_count'],
       color=bar_colors, edgecolor='black', linewidth=0.5)

ax.axhline(
    y=upper_fence, color='yellow', linestyle='--',
    linewidth=1.5, label=f'IQR Upper Fence ({upper_fence:.1f})'
)

ax.set_title(
    'Top 10 Source IPs by Deny Count\n(Red = Flagged by Combined Alert)',
    fontsize=13
)
ax.set_xlabel('Source IP', fontsize=11)
ax.set_ylabel('Deny Count', fontsize=11)
# Rotate IP labels so they do not overlap
ax.tick_params(axis='x', rotation=30)
ax.legend()
plt.tight_layout()
plt.savefig('chart2_bar.png', dpi=150)
plt.show()
print("[Chart 2] Saved as chart2_bar.png")


# --- Chart 3 (choice): Hourly activity heatmap for flagged IPs using seaborn ---
# Purpose: show WHEN each suspicious IP was active across the 24-hour day.
# Neither Chart 1 nor Chart 2 encodes time — this reveals patterns the other charts miss:
#   - Fully automated tools tend to run 24/7 with no gaps in the heatmap.
#   - Time-zone-constrained or human-operated activity shows up as a band of hours.
#   - Overnight activity (e.g., 2–4 AM) suggests evasion of business-hours monitoring.
# We use seaborn.heatmap() here because it automatically adds gridlines between cells,
# making it easy to read individual hour/IP combinations. matplotlib's imshow() does not
# draw gridlines by default and requires extra work to add them manually.

# Filter the cleaned dataframe down to only the flagged IPs
flagged_ips = ip_profile[ip_profile['combined_alert']]['src_ip'].tolist()
df_flagged = df[df['src_ip'].isin(flagged_ips)].copy()

# Extract hour-of-day from the parsed timestamp column (0–23)
df_flagged['hour'] = df_flagged['timestamp'].dt.hour

# Build a pivot table: rows = IP addresses, columns = hours (0–23), values = event count.
# fill_value=0 ensures every hour slot exists even if an IP had no activity that hour,
# which prevents seaborn from showing gaps or NaN values in the grid.
pivot = df_flagged.groupby(['src_ip', 'hour']).size().unstack(fill_value=0)

# Ensure all 24 hours appear as columns even if some have zero events across all flagged IPs.
# reindex fills in any missing hour columns with 0.
pivot = pivot.reindex(columns=range(24), fill_value=0)

fig, ax = plt.subplots(figsize=(16, 5))

# sns.heatmap() renders the pivot table as a color-coded grid.
# annot=True prints the event count inside each cell.
# fmt='d' formats annotations as integers (no decimal places).
# linewidths=0.5 draws thin white gridlines between every cell — this is the key
#   advantage over matplotlib imshow(): each cell is visually separated.
# linecolor='white' makes the gridlines stand out against the color scale.
# cmap='YlOrRd' uses a yellow→orange→red scale — low counts = pale, high = dark red.
sns.heatmap(
    pivot,
    ax=ax,
    annot=True,          # Print the number inside each cell
    fmt='d',             # Integer format for annotations
    cmap='YlOrRd',       # Yellow-Orange-Red color scale
    linewidths=0.5,      # Gridline thickness between cells
    linecolor='white',   # Gridline color
    cbar_kws={'label': 'Event Count'}   # Label the color bar
)

ax.set_title(
    'Hourly Activity Heatmap — Flagged IPs Only\n'
    '(Each cell shows event count for that IP during that hour of day)',
    fontsize=13
)
ax.set_xlabel('Hour of Day (UTC, 0 = midnight)', fontsize=11)
ax.set_ylabel('Source IP', fontsize=11)
# Keep IP labels horizontal for readability
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
plt.savefig('chart3_heatmap.png', dpi=150)
plt.show()
print("[Chart 3] Saved as chart3_heatmap.png")


# =============================================================================
# STAGE 5 — ACT
# Investigation summary — one paragraph per flagged IP
# =============================================================================

print("\n" + "=" * 60)
print("STAGE 5 — ACT: Investigation Summary")
print("=" * 60)

print("""
--- IP: 172.16.0.131 ---
172.16.0.131 generated 401 DENY events over 30 days — a deny z-score of 5.03,
well above the 2.5 threshold, and also flagged by IQR (upper fence: 29.5).
Every single one of its 444 total events targeted port 22 (SSH) exclusively, and
100% of traffic originated from CN (China). Activity was concentrated between
hours 06:00 and 21:00 UTC, suggesting an automated but time-zone-constrained
attack pattern rather than fully 24/7 automation. Conclusion: high-confidence
SSH brute force attack. Recommend immediate perimeter block and submission to a
threat intelligence feed.

--- IP: 172.16.0.5 ---
172.16.0.5 generated 349 DENY events with a deny z-score of 4.32 and a 100%
deny rate — every connection attempt was blocked. Traffic ran across all 24 hours
without any gap, which is characteristic of fully automated scanning. The top
destination ports were 21 (FTP), 5432 (PostgreSQL), and 8443 (HTTPS-alt),
indicating broad service discovery rather than a single targeted exploit. Source
countries were CN, RU, and UNKNOWN in roughly equal thirds. Conclusion:
automated multi-service port scanner. Recommend block at perimeter and review
whether any of those ports should be externally exposed.

--- IP: 172.16.0.88 ---
172.16.0.88 generated 142 DENY events and was flagged by IQR but NOT by
z-score — a clear example of how extreme outliers compress z-scores for
moderately anomalous IPs. All 237 total events targeted port 443 (HTTPS)
exclusively, 100% from RU (Russia), and all activity occurred in a narrow
window between 02:00 and 04:00 UTC. This overnight timing pattern is consistent
with automated tools designed to avoid business-hours detection. Conclusion:
likely automated credential stuffing or vulnerability scanning against a web
application. Recommend block and cross-reference WAF logs for matching activity
during those hours.

--- IP: 10.99.0.1 ---
10.99.0.1 generated zero DENY events and was not flagged by IQR on deny_count,
but was flagged by z-score on total_bytes with a score of 6.92 — it transferred
over 51 MB compared to the network median of roughly 120 KB. All connections
were ALLOW on port 443 (HTTPS) to DE (Germany) during business hours
(09:00–17:00), with no blocked traffic whatsoever. The combination of
all-allowed connections and extreme data volume is consistent with data
exfiltration over an encrypted channel. Conclusion: uncertain — could be a
legitimate cloud backup or sync service, but the volume demands investigation.
Recommend identifying which application or user account owns this traffic and
confirming the German destination is an authorized service.
""")
