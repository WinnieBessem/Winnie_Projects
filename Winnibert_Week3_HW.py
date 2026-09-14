"""
Week 10 Homework — Security Visualization Challenge
Scenario 1: Network Anomaly Visualizer

This script simulates a real SOC workflow across five pipeline stages:
  Stage 1 — Collect  : Generate a synthetic 30-day firewall log dataset
  Stage 2 — Clean    : Normalize and validate the raw data
  Stage 3 — Analyze  : Build summary tables and compute the alert threshold
  Stage 4 — Visualize: Produce 4 individual charts and a multi-panel dashboard
  Stage 5 — Act      : Print specific security findings backed by the charts
"""

# ── Library imports ──────────────────────────────────────────────────────────
import numpy as np                        # numerical operations and seeding
import pandas as pd                       # DataFrame creation and manipulation
import matplotlib.pyplot as plt           # core plotting engine
import matplotlib.gridspec as gridspec    # multi-panel dashboard layout
# statistical chart library (heatmap, countplot)
import seaborn as sns
from matplotlib.patches import Patch      # used to build custom legend entries
# date arithmetic for timestamp generation
from datetime import datetime, timedelta
import random                             # random sampling for synthetic data

# Apply a consistent dark theme so all charts share the same visual style
plt.style.use("dark_background")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — COLLECT
# Generate a synthetic 30-day firewall log that meets all scenario requirements.
# Seeds are fixed so the dataset is reproducible every time the script runs.
# ─────────────────────────────────────────────────────────────────────────────

# Fix random seeds so results are identical on every run
random.seed(42)
np.random.seed(42)

# Dataset size and time window constants
NUM_ROWS = 600
START_DATE = datetime(2024, 3, 1)   # first day of the 30-day logging period

# Two separate IP pools keep attackers and normal traffic clearly separated.
# HIGH_RISK_IPS will be weighted toward DENY actions to simulate repeat attackers.
# NORMAL_IPS represent ordinary internal hosts with mixed allow/deny traffic.
HIGH_RISK_IPS = [f"192.168.{i}.1" for i in range(
    1, 6)]           # 5 attacker IPs
NORMAL_IPS = [f"10.0.{i}.{j}" for i in range(1, 5)             # 20 internal IPs
              for j in range(1, 6)]

# Common destination ports that real attackers probe (SSH, HTTP, RDP, DNS, etc.)
PORTS = [22, 80, 443, 3389, 8080, 53, 445, 3306, 25, 110]
ACTIONS = ["ALLOW", "DENY"]
SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# Build the raw event list row by row before converting to a DataFrame
rows = []
for _ in range(NUM_ROWS):

    # Pick a random day within the 30-day window
    day_offset = random.randint(0, 29)
    base_dt = START_DATE + timedelta(days=day_offset)

    # Off-hours gate: ~30% of events fall before 8 am or after 6 pm.
    # This ensures the heatmap in Chart 3 shows a visible off-hours pattern.
    if random.random() < 0.30:
        # Off-hours: midnight to 7 am, or 6 pm to 11 pm
        hour = random.choice(list(range(0, 8)) + list(range(18, 24)))
    else:
        # Business hours: 8 am to 5 pm
        hour = random.randint(8, 17)

    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    ts = base_dt.replace(hour=hour, minute=minute, second=second)

    # High-risk IP branch: 35% chance of selecting an attacker IP.
    # Attacker events have an 85% chance of being DENY and lean toward HIGH/CRITICAL severity.
    if random.random() < 0.35:
        src_ip = random.choice(HIGH_RISK_IPS)
        action = "DENY" if random.random() < 0.85 else "ALLOW"
        sev = random.choice(["HIGH", "CRITICAL", "HIGH"]
                            )   # doubles HIGH probability
    else:
        # Normal traffic: random IP, random action, any severity
        src_ip = random.choice(NORMAL_IPS)
        action = random.choice(ACTIONS)
        sev = random.choice(SEVERITIES)

    rows.append({
        "timestamp":  ts,
        "src_ip":     src_ip,
        "dst_port":   random.choice(PORTS),
        "action":     action,
        # realistic packet size range in bytes
        "bytes_sent": random.randint(40, 65000),
        "severity":   sev,
    })

# Inject a concentrated attack spike on two specific days.
# This creates the threshold breaches that Chart 1 (time-series) needs to show.
# All spike events are off-hours and come from high-risk IPs to simulate a real campaign.
for spike_day in [10, 22]:
    spike_dt = START_DATE + timedelta(days=spike_day)
    for _ in range(40):
        # Concentrate the spike in early morning or late night hours
        hr = random.choice([1, 2, 3, 22, 23])
        ts = spike_dt.replace(hour=hr, minute=random.randint(0, 59))
        rows.append({
            "timestamp":  ts,
            "src_ip":     random.choice(HIGH_RISK_IPS),
            "dst_port":   random.choice(PORTS),
            "action":     "DENY",                              # all spike events are DENY
            "bytes_sent": random.randint(40, 65000),
            "severity":   random.choice(["HIGH", "CRITICAL"]),
        })

# Convert the list of dicts to a DataFrame and sort chronologically
df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

# ── Inspection toolkit ───────────────────────────────────────────────────────
# Print a full profile of the raw dataset before any cleaning is applied.
print("=== STAGE 1: COLLECT ===")
print(f"Shape: {df.shape}")           # total rows and columns

print("\nColumn info (dtypes and null counts):")
df.info()

print(f"\nUnique values per column:\n{df.nunique()}")

# value_counts() on categorical columns reveals distribution imbalances early
print(f"\nSeverity breakdown:\n{df['severity'].value_counts()}")
print(f"\nAction breakdown:\n{df['action'].value_counts()}")

# describe() on numeric columns catches any out-of-range values before cleaning
print(
    f"\nBytes sent — descriptive statistics:\n{df[['bytes_sent']].describe()}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — CLEAN
# Validate and normalize the raw data using 6 techniques.
# Always operate on df_clean (a full copy) so the original df is preserved
# for comparison or re-processing without re-running Stage 1.
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== STAGE 2: CLEAN ===")
print(f"Rows before cleaning: {len(df)}")

# Technique 0: Copy — work on df_clean exclusively from this point forward.
# A boolean-filtered slice like df[df['action']=='DENY'] is still a view,
# not a true copy, so .copy() on the full DataFrame is the safe pattern.
df_clean = df.copy()

# Technique 1: Ensure the timestamp column is a proper datetime dtype.
# This is defensive — the column is already datetime from Stage 1 —
# but pd.to_datetime() catches any edge cases if the data came from a CSV.
df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])

# Technique 2: Drop exact duplicate rows.
# Duplicate firewall events can happen from log collector restarts or double-writes.
df_clean.drop_duplicates(inplace=True)

# Technique 3: Drop rows where any critical column is null.
# A firewall event with no timestamp, IP, action, or severity is not analyzable.
df_clean.dropna(subset=["timestamp", "src_ip",
                "action", "severity"], inplace=True)

# Technique 4: Normalize string columns to uppercase and strip whitespace.
# This prevents groupby from treating "deny", "Deny", and "DENY" as separate values.
df_clean["action"] = df_clean["action"].str.upper().str.strip()
df_clean["severity"] = df_clean["severity"].str.upper().str.strip()

# Technique 5: Remove rows with physically impossible byte values.
# Negative bytes are a data error; anything over 1 MB is outside realistic firewall traffic.
df_clean = df_clean[
    (df_clean["bytes_sent"] >= 0) &
    (df_clean["bytes_sent"] <= 1_000_000)
]

# Technique 6: Derive time-component columns from the timestamp.
# These columns make Stage 3 groupby operations and the Stage 4 heatmap much simpler.
# calendar date (YYYY-MM-DD)
df_clean["date"] = df_clean["timestamp"].dt.date
df_clean["hour"] = df_clean["timestamp"].dt.hour        # 0–23 integer hour
df_clean["day_of_week"] = df_clean["timestamp"].dt.day_name()  # e.g. "Monday"

print(f"Rows after cleaning : {len(df_clean)}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — ANALYZE
# Build two summary tables, compute a derived metric, and calculate the
# statistical alert threshold used in Chart 1.
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== STAGE 3: ANALYZE ===")

# ── Summary table 1: daily DENY volume ──────────────────────────────────────
# Filter to DENY events only, then count how many occurred on each calendar date.
# This table feeds directly into Chart 1 (time-series) and the threshold calculation.
daily_deny = (
    df_clean[df_clean["action"] == "DENY"]
    .groupby("date")
    # named aggregation keeps the column label clean
    .agg(deny_count=("action", "count"))
    .reset_index()
    # sort chronologically for correct line chart order
    .sort_values("date")
)

# ── Summary table 2: per-IP DENY count ──────────────────────────────────────
# Count DENY events grouped by source IP, sorted descending for the bar chart.
ip_deny = (
    df_clean[df_clean["action"] == "DENY"]
    .groupby("src_ip")
    .agg(deny_count=("action", "count"))
    .reset_index()
    .sort_values("deny_count", ascending=False)
)

# ── Derived metric: deny_rate ────────────────────────────────────────────────
# Count total events per IP (ALLOW + DENY), then merge with the DENY-only table
# to compute the ratio: deny_rate = DENY events / total events for that IP.
# A high deny_rate (close to 1.0) means almost all traffic from that IP was blocked.
ip_totals = df_clean.groupby("src_ip").size().reset_index(name="total_events")
ip_deny = ip_deny.merge(ip_totals, on="src_ip", how="left")
ip_deny["deny_rate"] = ip_deny["deny_count"] / ip_deny["total_events"]

# Flag any IP with more than 25 DENY events as high-risk.
# This threshold is scaled to the synthetic dataset; on real logs use a higher cutoff.
# The high_risk column drives the color coding in Chart 2.
ip_deny["high_risk"] = ip_deny["deny_count"] > 25

print("Top 10 source IPs by DENY count:")
print(ip_deny.head(10).to_string(index=False))

# ── Alert threshold: μ + 2σ ──────────────────────────────────────────────────
# Under a normal distribution, ~95% of values fall within 2 standard deviations
# of the mean. Any day above μ + 2σ is in the top ~2.5% of expected activity,
# which is a statistically meaningful definition of "abnormally high."
mean_deny = daily_deny["deny_count"].mean()
std_deny = daily_deny["deny_count"].std()
threshold = mean_deny + 2 * std_deny

# 3-day rolling average smooths out day-to-day noise in Chart 1.
# min_periods=1 prevents NaN on the first two days where a full 3-day window
# is not yet available.
rolling_avg = daily_deny["deny_count"].rolling(window=3, min_periods=1).mean()

print(f"\nDaily DENY — mean: {mean_deny:.1f},  σ: {std_deny:.1f},  "
      f"alert threshold (μ+2σ): {threshold:.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — VISUALIZE
# Every chart must be saved as PNG with plt.savefig() BEFORE plt.show().
# Calling show() first clears the figure buffer, resulting in a blank saved file.
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== STAGE 4: VISUALIZE ===")

# ── Shared colour palette ────────────────────────────────────────────────────
# Centralising colors here means changing one line updates all charts at once.
COLOR_LINE = "#00C8FF"   # cyan — primary data line
COLOR_ROLLING = "#FFB700"   # amber — rolling average line
COLOR_THRESHOLD = "#FF4C4C"   # red   — alert threshold line
COLOR_SHADE = "#FF4C4C"   # red   — threshold breach shading
COLOR_SAFE = "#4CAF50"   # green — normal/safe IP bars
COLOR_RISK = "#FF4C4C"   # red   — high-risk IP bars


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1: Daily DENY Volume — Time-Series Line Chart
# Answers: "Were there any days with abnormally high DENY activity in March 2024?"
# Required elements: threshold line (axhline), rolling average, alert-period shading
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))

# Use integer positions on the x-axis so matplotlib spaces points evenly.
# Real dates are applied as tick labels below.
date_nums = range(len(daily_deny))

# Primary line: raw daily DENY count with dot markers for each data point
ax.plot(date_nums, daily_deny["deny_count"],
        color=COLOR_LINE, linewidth=1.8, marker="o", markersize=4,
        label="Daily DENY Count")

# Dashed overlay: 3-day rolling average to show the underlying trend
ax.plot(date_nums, rolling_avg,
        color=COLOR_ROLLING, linewidth=2.2, linestyle="--",
        label="3-Day Rolling Average")

# Horizontal alert threshold line — any day above this line warrants SOC investigation
ax.axhline(y=threshold, color=COLOR_THRESHOLD, linewidth=1.5, linestyle=":",
           label=f"Alert Threshold (μ+2σ = {threshold:.0f})")

# Boolean mask: True on days where the daily count exceeded the threshold
above_threshold = daily_deny["deny_count"] > threshold

# Red shading fills the area between the actual count and the threshold line
# only on breach days, making the alert periods immediately visible
ax.fill_between(date_nums, daily_deny["deny_count"], threshold,
                where=above_threshold, alpha=0.25, color=COLOR_SHADE,
                label="Alert Period")

# Apply real date strings as x-axis tick labels every 5 days to avoid overcrowding
date_labels = [str(d) for d in daily_deny["date"]]
ax.set_xticks(range(0, len(date_nums), 5))
ax.set_xticklabels([date_labels[i] for i in range(0, len(date_labels), 5)],
                   rotation=30, ha="right", fontsize=8)

ax.set_title("Daily DENY Event Volume — March 2024 (30-Day Firewall Log)",
             fontsize=13, pad=12)
ax.set_xlabel("Date", fontsize=10)
ax.set_ylabel("DENY Event Count", fontsize=10)
ax.legend(fontsize=9)
# horizontal grid aids value estimation
ax.grid(axis="y", linestyle="--", alpha=0.3)

# savefig MUST come before show() — show() clears the figure
plt.tight_layout()
plt.savefig("chart1_timeseries_deny.png", dpi=150)
plt.show()
print("Saved: chart1_timeseries_deny.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2: Top 10 Source IPs by DENY Count — Horizontal Bar Chart
# Answers: "Which source IPs are generating the most blocked traffic?"
# Required elements: horizontal orientation, descending sort, color by risk, bar labels
# ─────────────────────────────────────────────────────────────────────────────

# Sort ascending so the longest bar ends up at the top of a horizontal bar chart
top10 = ip_deny.head(10).sort_values("deny_count", ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))

# Map high_risk flag to color: red for high-risk IPs, green for normal ones
bar_colors = [COLOR_RISK if hr else COLOR_SAFE for hr in top10["high_risk"]]

# barh() draws horizontal bars; IP labels on the y-axis are easier to read than rotated x labels
bars = ax.barh(top10["src_ip"], top10["deny_count"],
               color=bar_colors, edgecolor="none")

# Place the exact count at the right end of each bar so values are readable at a glance
ax.bar_label(bars, fmt="%d", padding=4, fontsize=9, color="white")

# Build a manual legend since the color coding doesn't come from a hue column
legend_elements = [
    Patch(facecolor=COLOR_RISK, label="High-Risk IP (>25 DENY events)"),
    Patch(facecolor=COLOR_SAFE, label="Normal IP"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

ax.set_title("Top 10 Source IPs by DENY Count — March 2024",
             fontsize=13, pad=12)
ax.set_xlabel("DENY Event Count", fontsize=10)
ax.set_ylabel("Source IP", fontsize=10)
# vertical grid lines aid bar length comparison
ax.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("chart2_top10_ips.png", dpi=150)
plt.show()
print("Saved: chart2_top10_ips.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3: Seaborn Heatmap — Hour-of-Day × Day-of-Week DENY Count
# Answers: "Is there a time-of-day or day-of-week pattern in DENY activity?"
# Required elements: annot=True, fmt='d', linewidths=0.5, explicit vmax, colorbar label
# ─────────────────────────────────────────────────────────────────────────────

# Explicit day order ensures columns appear Monday–Sunday, not alphabetically
DOW_ORDER = ["Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Saturday", "Sunday"]

# Filter to DENY events only before building the pivot table
deny_events = df_clean[df_clean["action"] == "DENY"].copy()

# Build a 2D pivot: rows = hours (0–23), columns = days of week.
# Each cell value is the total number of DENY events at that hour/day combination.
# fillna(0) replaces missing combinations with zero so the heatmap has no blank cells.
heatmap_data = (
    deny_events
    .groupby(["hour", "day_of_week"])
    .size()
    .reset_index(name="count")
    .pivot(index="hour", columns="day_of_week", values="count")
    .reindex(columns=DOW_ORDER)    # enforce Monday-first column order
    .fillna(0)
    .astype(int)                   # convert float NaN-fills back to integers
)

fig, ax = plt.subplots(figsize=(11, 8))

sns.heatmap(
    heatmap_data,
    ax=ax,
    annot=True,                           # display the integer count inside every cell
    # format annotations as plain integers (not scientific)
    fmt="d",
    linewidths=0.5,                       # thin borders between cells improve readability
    # anchor the top of the color scale to the actual max
    vmax=heatmap_data.values.max(),
    # yellow-orange-red: light = low, dark = high activity
    cmap="YlOrRd",
    cbar_kws={"label": "DENY Event Count"},
)

# Draw a cyan rectangle border around every off-hours row (before 8 am or after 6 pm).
# These borders make the off-hours anomaly visible without altering the color scale.
off_hours = list(range(0, 8)) + list(range(18, 24))
for hr in off_hours:
    if hr in heatmap_data.index:
        # Convert the hour value to a row index position within the pivot table
        row_idx = list(heatmap_data.index).index(hr)
        ax.add_patch(plt.Rectangle(
            (0, row_idx),              # bottom-left corner of the rectangle
            # width spans all 7 day columns, height = 1 row
            len(DOW_ORDER), 1,
            fill=False,
            edgecolor="cyan",
            lw=1.2
        ))

ax.set_title(
    "DENY Events by Hour-of-Day × Day-of-Week — March 2024\n"
    "(Cyan borders = off-hours: before 8 am or after 6 pm)",
    fontsize=12, pad=12
)
ax.set_xlabel("Day of Week", fontsize=10)
ax.set_ylabel("Hour of Day (24-hour)", fontsize=10)

plt.tight_layout()
plt.savefig("chart3_heatmap_hour_dow.png", dpi=150)
plt.show()
print("Saved: chart3_heatmap_hour_dow.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4: Event Severity Distribution — Seaborn Countplot
# Answers: "How are firewall events distributed across severity levels?"
# Required elements: Seaborn chart, labeled axes, bar value annotations
# ─────────────────────────────────────────────────────────────────────────────

# Ordered severity levels from least to most severe for a logical left-to-right layout
SEV_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# Semantic color mapping: green = safe, yellow = caution, orange = warning, red = critical
SEV_COLORS = {
    "LOW":      "#4CAF50",
    "MEDIUM":   "#FFB700",
    "HIGH":     "#FF7043",
    "CRITICAL": "#FF1744",
}

fig, ax = plt.subplots(figsize=(8, 5))

# hue="severity" + legend=False applies the palette without adding a redundant legend
sns.countplot(
    data=df_clean,
    x="severity",
    hue="severity",
    order=SEV_ORDER,
    palette=SEV_COLORS,
    legend=False,
    ax=ax,
)

# Loop through every bar patch and place its count just above the top edge
for bar in ax.patches:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,   # center the label horizontally
        height + 1,                            # position just above the bar top
        f"{int(height)}",
        ha="center", va="bottom", fontsize=10, color="white"
    )

ax.set_title("Event Severity Distribution — All Firewall Events, March 2024",
             fontsize=13, pad=12)
ax.set_xlabel("Severity Level", fontsize=10)
ax.set_ylabel("Event Count", fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("chart4_severity_countplot.png", dpi=150)
plt.show()
print("Saved: chart4_severity_countplot.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 5: Multi-Panel Dashboard (2 × 2 GridSpec)
# Combines all four charts into one leadership-ready briefing figure.
# GridSpec gives precise control over panel positions and spacing.
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 14))

# Master title for the entire dashboard figure
fig.suptitle(
    "Network Security Dashboard — March 2024 Firewall Log Analysis",
    fontsize=16, fontweight="bold", y=0.98
)

# GridSpec(2, 2): 2 rows, 2 columns.
# hspace controls vertical space between rows; wspace controls horizontal space between columns.
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel A (top-left): Time-Series ──────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])

# Replicate Chart 1 inside the panel — same data, smaller labels to fit the tighter space
ax_a.plot(date_nums, daily_deny["deny_count"],
          color=COLOR_LINE, linewidth=1.8, marker="o", markersize=3,
          label="Daily DENY")
ax_a.plot(date_nums, rolling_avg,
          color=COLOR_ROLLING, linewidth=2, linestyle="--",
          label="3-Day Avg")
ax_a.axhline(y=threshold, color=COLOR_THRESHOLD, linewidth=1.5,
             linestyle=":", label=f"Threshold ({threshold:.0f})")
ax_a.fill_between(date_nums, daily_deny["deny_count"], threshold,
                  where=above_threshold, alpha=0.22, color=COLOR_SHADE)

# Show a tick every 7 days so the x-axis isn't too crowded inside the panel
ax_a.set_xticks(range(0, len(date_nums), 7))
ax_a.set_xticklabels(
    [date_labels[i] for i in range(0, len(date_labels), 7)],
    rotation=25, ha="right", fontsize=7
)
ax_a.set_title("A. Daily DENY Volume with Alert Threshold", fontsize=10)
ax_a.set_xlabel("Date", fontsize=8)
ax_a.set_ylabel("DENY Count", fontsize=8)
ax_a.legend(fontsize=7)
ax_a.grid(axis="y", linestyle="--", alpha=0.3)

# ── Panel B (top-right): Top-10 IP Bar Chart ─────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

# Reuse top10 and bar_colors computed for Chart 2
bars_b = ax_b.barh(top10["src_ip"], top10["deny_count"],
                   color=bar_colors, edgecolor="none")
ax_b.bar_label(bars_b, fmt="%d", padding=3, fontsize=7, color="white")
ax_b.set_title("B. Top 10 Source IPs by DENY Count", fontsize=10)
ax_b.set_xlabel("DENY Count", fontsize=8)
ax_b.set_ylabel("Source IP", fontsize=8)
ax_b.tick_params(labelsize=7)
ax_b.legend(handles=legend_elements, fontsize=7, loc="lower right")
ax_b.grid(axis="x", linestyle="--", alpha=0.3)

# ── Panel C (bottom-left): Heatmap ───────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])

# Pass ax=ax_c so Seaborn draws into this specific panel instead of a new figure
sns.heatmap(
    heatmap_data,
    ax=ax_c,
    annot=True,
    fmt="d",
    linewidths=0.4,
    # smaller annotation font to fit inside the panel
    annot_kws={"size": 7},
    vmax=heatmap_data.values.max(),
    cmap="YlOrRd",
    cbar_kws={"label": "DENY Count", "shrink": 0.8},
)
ax_c.set_title("C. DENY Events: Hour-of-Day × Day-of-Week", fontsize=10)
ax_c.set_xlabel("Day of Week", fontsize=8)
ax_c.set_ylabel("Hour of Day", fontsize=8)
ax_c.tick_params(labelsize=7)

# ── Panel D (bottom-right): Severity Countplot ───────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])

# Pass ax=ax_d so Seaborn draws into this panel instead of a new figure
sns.countplot(
    data=df_clean,
    x="severity",
    hue="severity",
    order=SEV_ORDER,
    palette=SEV_COLORS,
    legend=False,
    ax=ax_d,
)

# Add count labels above each bar, scaled down for the smaller panel size
for bar in ax_d.patches:
    height = bar.get_height()
    ax_d.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.5,
        f"{int(height)}",
        ha="center", va="bottom", fontsize=8, color="white"
    )

ax_d.set_title("D. Event Severity Distribution", fontsize=10)
ax_d.set_xlabel("Severity", fontsize=8)
ax_d.set_ylabel("Count", fontsize=8)
ax_d.tick_params(labelsize=8)
ax_d.grid(axis="y", linestyle="--", alpha=0.3)

# bbox_inches="tight" ensures the suptitle is not clipped when saving
plt.savefig("chart5_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: chart5_dashboard.png")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — ACT
# Print at least 3 specific, chart-backed security findings.
# Each finding references a chart by name, and at least one finding (Finding 3)
# describes a pattern that is only visible in the chart, not in any summary table.
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== STAGE 5: ACT — Security Findings ===")

# ── Finding 1 — sourced from Chart 1 (time-series) ───────────────────────────
# Identify every day where the daily DENY count exceeded the μ + 2σ threshold
breach_days = daily_deny[daily_deny["deny_count"] > threshold]

print(f"\nFinding 1 (Chart 1 — Time-Series):")
print(f"  {len(breach_days)} day(s) exceeded the alert threshold of {threshold:.0f} DENY events.")

if not breach_days.empty:
    # Pull the specific date and count of the worst breach for a concrete recommendation
    peak_day = breach_days.loc[breach_days["deny_count"].idxmax(), "date"]
    peak_count = breach_days["deny_count"].max()
    print(f"  Peak activity: {peak_day} with {peak_count} DENY events.")
    print(
        f"  Recommendation: Investigate traffic sources on {peak_day} for coordinated attack activity.")

# ── Finding 2 — sourced from Chart 2 (top-10 IP bar chart) ───────────────────
# Filter the IP table to only those flagged as high-risk
high_risk_ips = ip_deny[ip_deny["high_risk"] == True]

print(f"\nFinding 2 (Chart 2 — Top-10 IPs Bar Chart):")
print(f"  {len(high_risk_ips)} source IP(s) exceeded the high-risk DENY threshold.")

if not high_risk_ips.empty:
    # Report the single worst offender with its deny_rate for executive context
    worst_ip = high_risk_ips.iloc[0]["src_ip"]
    worst_count = high_risk_ips.iloc[0]["deny_count"]
    worst_rate = high_risk_ips.iloc[0]["deny_rate"]
    print(
        f"  Top offender: {worst_ip} — {worst_count} DENY events ({worst_rate:.0%} deny rate).")
    print(
        f"  Recommendation: Immediately block {worst_ip} at the perimeter firewall.")

# ── Finding 3 — sourced from Chart 3 (heatmap) — chart-only pattern ──────────
# Calculate the percentage of DENY events that fell outside business hours.
# This hour-level concentration is invisible in the daily_deny summary table,
# making it a true chart-only finding.
off_hours_deny = deny_events[deny_events["hour"].isin(off_hours)]
off_hours_pct = len(off_hours_deny) / len(deny_events) * 100

# Find the single hour with the highest off-hours DENY count
peak_off_hour = (
    deny_events[deny_events["hour"].isin(off_hours)]
    .groupby("hour")
    .size()
    .idxmax()
)

print(f"\nFinding 3 (Chart 3 — Heatmap, pattern only visible across chart cells):")
print(f"  {off_hours_pct:.1f}% of all DENY events occurred outside business hours.")
print(
    f"  Hour {peak_off_hour:02d}:00 has the highest off-hours DENY concentration —")
print(f"  this hour-level pattern is not visible in any summary table.")
print(f"  Recommendation: Configure automated alerting for DENY spikes during off-hours windows.")

print("\nAnalysis complete. All charts saved as PNG files.")
