# =============================================================
# Insider Threat Detector - Badge Access Log Analysis
# Cyber Data Scenario Challenge - Week 1 Homework
# =============================================================

# pandas lets us create and work with tables of data (DataFrames)
import pandas as pd
# numpy gives us tools to generate random numbers and arrays
import numpy as np
# matplotlib is used to create charts and visualizations
import matplotlib.pyplot as plt


# =============================================================
# STAGE 1 - COLLECT
# Generate a fake 200-row badge access log dataset.
# We simulate this data because we don't have a real security system,
# but the structure mirrors real physical access logs.
# =============================================================

# Setting a seed means the random data stays the same every run,
# which makes testing and debugging much easier
np.random.seed(42)

num_records = 200
# Creates ['EMP001', 'EMP002', ..., 'EMP010'] for employee IDs
employee_ids = [f"EMP{str(i).zfill(3)}" for i in range(1, 11)]
zones = ["Server Room", "Executive Floor",
         "Data Center", "HR Office", "Main Lobby"]

# Generate random timestamps spread across 30 days including nights and weekends.
# We convert days to seconds so we can pick a random second within that range.
start_date = pd.Timestamp("2024-01-01")
random_seconds = np.random.randint(0, 30 * 24 * 60 * 60, size=num_records)
timestamps = [start_date + pd.Timedelta(seconds=int(s))
              for s in random_seconds]

# Build the DataFrame by combining all randomly generated columns.
# p=[0.75, 0.25] means 75% of swipes are granted and 25% denied - realistic ratio
df = pd.DataFrame({
    "employee_id":   np.random.choice(employee_ids, size=num_records),
    "zone":          np.random.choice(zones, size=num_records),
    "timestamp":     timestamps,
    "access_result": np.random.choice(["granted", "denied"], size=num_records, p=[0.75, 0.25])
})

# 5 required inspection commands - confirms the data loaded correctly
print("Shape:",            df.shape)           # (rows, columns)
print("\nData Types:\n",   df.dtypes)           # confirms column types
print("\nFirst 5 Rows:\n", df.head())           # visual preview of the data
print("\nSummary:\n",      df.describe(include="all"))  # stats per column
print("\nMissing:\n",      df.isnull().sum())   # counts any empty cells


# =============================================================
# STAGE 2 - CLEAN
# Fix data types, casing, duplicates, and missing values.
# Skipping this step would cause incorrect results in analysis
# because Python treats "Granted" and "granted" as different values.
# =============================================================

# Convert timestamp to datetime so we can extract hour and date later.
# errors="coerce" turns any unreadable timestamps into NaT instead of crashing.
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Lowercase both text columns so values like "Server Room" and "server room"
# are treated as the same thing during filtering and grouping
df["zone"] = df["zone"].str.lower()
df["access_result"] = df["access_result"].str.lower()

# Save the row count before dropping so we can report how many were removed
rows_before = len(df)
# drop_duplicates() removes any row that is an exact copy of another row
df.drop_duplicates(inplace=True)
print(
    f"\nDuplicates removed: {rows_before - len(df)} | Rows remaining: {len(df)}")

# Use placeholder values for missing text so we don't lose the record entirely.
# A badge swipe with no employee ID is still a security event worth keeping.
df["employee_id"] = df["employee_id"].fillna("UNKNOWN")
df["zone"] = df["zone"].fillna("unknown zone")
df["access_result"] = df["access_result"].fillna("unknown")
# Drop rows with no timestamp since we can't do time-based analysis without it
df.dropna(subset=["timestamp"], inplace=True)

print("\nMissing values after cleaning:\n",
      df.isnull().sum())  # should all be 0
print("\nCleaned preview:\n", df.head())


# =============================================================
# STAGE 3 - ANALYZE
# Answer 3 security questions about suspicious behavior.
# Each question prints a clearly labeled finding.
# =============================================================

# Pull the hour (0-23) and date out of the timestamp into separate columns
# so we can filter by time of day and group by day easily
df["hour"] = df["timestamp"].dt.hour
df["date"] = df["timestamp"].dt.date

# --- Question 1: Who accessed restricted zones after hours? ---
# After hours = before 7am or after 7pm (hours 0-6 and 19-23)
# Restricted zones are the two most sensitive areas in the building
restricted_zones = ["server room", "data center"]
# Boolean mask - True for every row that falls outside business hours
after_hours = (df["hour"] < 7) | (df["hour"] >= 19)
# Apply both filters at once: must be after hours AND in a restricted zone
q1 = df[after_hours & df["zone"].isin(restricted_zones)]

print("\n" + "="*55)
print("FINDING 1: After-hours restricted zone access")
print("="*55)
print(f"Total swipes: {len(q1)}")  # how many times did this happen?
# which employees were involved?
print("Employees:    ", q1["employee_id"].unique())
print(q1[["employee_id", "zone", "timestamp", "hour",
      # show all details of each swipe
          "access_result"]].to_string(index=False))

# --- Question 2: Who made the most after-hours attempts? ---
# groupby counts how many swipes each employee made after hours,
# then we sort descending so the most suspicious employee appears first
after_hours_counts = (
    df[after_hours]
    .groupby("employee_id")       # group all after-hours rows by employee
    .size()                        # count swipes per employee
    .sort_values(ascending=False)  # highest count at the top
    .reset_index(name="after_hours_swipes")  # rename count column
)

print("\n" + "="*55)
print("FINDING 2: Employees ranked by after-hours attempts")
print("="*55)
print(after_hours_counts.to_string(index=False))

# --- Question 3: Who is denied access to zones they normally enter? ---
# First get all unique employee+zone pairs where access was granted
# This is a red flag - it could mean a badge was revoked,
# someone is testing doors they shouldn't be at,
# or an employee's access level was downgraded after an incident.

granted = df[df["access_result"] == "granted"][[
    "employee_id", "zone"]].drop_duplicates()
# Then get all unique employee+zone pairs where access was denied
denied = df[df["access_result"] == "denied"][[
    "employee_id", "zone"]].drop_duplicates()
# Inner join keeps only rows that appear in BOTH tables -
# meaning the employee has been both granted AND denied in the same zone

# Now merge the two sets to find employees who appear in BOTH.
# An employee showing up in both means they were granted access to a zone
# at some point, but were also denied access to that SAME zone at another point.
suspicious_denials = pd.merge(
    granted, denied, on=["employee_id", "zone"], how="inner")

print("\n" + "="*55)
print("FINDING 3: Denied access to normally-accessible zones")
print("="*55)
# Check if any suspicious denials were found before printing.
# If the DataFrame is empty, we print a different message so the output is clear.
if len(suspicious_denials) == 0:
    print("No suspicious denial patterns found.")
else:
    print(f"Total flagged pairs: {len(suspicious_denials)}")
    print(suspicious_denials.to_string(index=False))


# =============================================================
# STAGE 4 - VISUALIZE
# Charts make suspicious patterns immediately visible.
# A spike or outlier in a chart is much faster to spot
# than reading through rows of printed numbers.
# =============================================================

# --- Chart 1: Bar chart - top employees by after-hours swipes ---
# Shows WHO is most suspicious at a glance
top10 = after_hours_counts.head(10)  # take top 10 most suspicious employees
fig, ax = plt.subplots(figsize=(10, 6))  # create a 10x6 inch figure
# Red bars signal danger - taller bar = more suspicious
ax.bar(top10["employee_id"], top10["after_hours_swipes"],
       color="crimson", label="After-Hours Swipes")
ax.set_title("Top Employees by After-Hours Access Attempts", fontsize=14)
ax.set_xlabel("Employee ID", fontsize=12)
ax.set_ylabel("Number of After-Hours Swipes", fontsize=12)
plt.xticks(rotation=45, ha="right")  # rotate labels so they don't overlap
ax.legend(loc="upper right")
# light grid makes bar heights easier to read
ax.yaxis.grid(True, color="gray", alpha=0.7)
plt.tight_layout()   # prevents labels from being cut off
# save before show() or file may be blank
plt.savefig("chart1_top_afterhours.png", dpi=150)
plt.show()
print("Chart 1 saved.")

# --- Chart 2: Line chart - swipe activity by hour of day ---
# Shows WHEN suspicious activity peaks - a 3am spike is a major red flag
hourly_counts = (
    df[after_hours]
    .groupby("hour")               # group after-hours rows by hour of day
    .size()                         # count swipes per hour
    .reset_index(name="swipe_count")
)

fig, ax = plt.subplots(figsize=(10, 6))
# marker="o" puts a dot at each data point so individual hours are visible
ax.plot(hourly_counts["hour"], hourly_counts["swipe_count"],
        marker="o", color="darkorange", linewidth=2, label="Swipes Per Hour")
# Shaded band draws attention to the most suspicious time window (midnight-5am)
ax.axvspan(0, 5, alpha=0.1, color="red", label="Deep Night (12am-5am)")
ax.set_title("After-Hours Badge Swipe Activity by Hour of Day", fontsize=14)
ax.set_xlabel("Hour of Day (24-Hour Format)", fontsize=12)
ax.set_ylabel("Number of Swipes", fontsize=12)
# Manually set x ticks to only show after-hours hours (avoids decimals like 19.5)
ax.set_xticks(list(range(0, 7)) + list(range(19, 24)))
ax.legend(loc="upper right")
ax.grid(True, color="black", alpha=0.5)
plt.tight_layout()
plt.savefig("chart2_hourly_activity.png", dpi=150)
plt.show()
print("Chart 2 saved.")


# =============================================================
# STAGE 5 - ACT
# Automatically flag suspicious behavior and produce a report.
# This turns the script from a data viewer into a security tool
# that takes action based on what the data reveals.
# =============================================================

# --- Alert 1: Flag employees exceeding after-hours threshold ---
# Storing the threshold as a variable means we can change it in one place
print("\n" + "="*55)
print("ALERT 1: After-Hours Threshold Check (limit = 5)")
print("="*55)
threshold = 5
# Filter after_hours_counts to only employees who exceeded our limit
flagged_employees = after_hours_counts[after_hours_counts["after_hours_swipes"] > threshold]

if len(flagged_employees) == 0:
    print(f"No employees exceeded {threshold} after-hours swipes.")
else:
    # Loop through each flagged employee and print a personalized alert
    for _, row in flagged_employees.iterrows():
        print(
            f"  ALERT: {row['employee_id']} --- {row['after_hours_swipes']} after-hours swipes (limit: {threshold})")

# --- Alert 2: Flag deep night access to critical zones ---
# Even a single granted entry to Server Room or Data Center at 2am is serious
print("\n" + "="*55)
print("ALERT 2: Deep Night Critical Zone Access (12am - 5am)")
print("="*55)
# Filter for rows matching ALL three conditions at once
deep_night_flags = df[
    # must be a critical zone
    df["zone"].isin(["server room", "data center"]) &
    # must be between 12am-5am
    df["hour"].isin(range(0, 5)) &
    (df["access_result"] == "granted")                  # must have been let in
]

if len(deep_night_flags) == 0:
    print("No critical zone access detected between 12am and 5am.")
else:
    print(
        f"WARNING: {len(deep_night_flags)} granted entries in critical zones during deep night:\n")
    print(deep_night_flags[["employee_id", "zone",
          "timestamp", "hour"]].to_string(index=False))

# --- Final Summary Report ---
# Combines all findings into one readable summary for a manager or security team
print("\n" + "="*55)
print("FINAL SUMMARY REPORT")
print("="*55)
print(f"Total records analyzed:           {len(df)}")
# nunique = count of distinct values
print(f"Total unique employees:           {df['employee_id'].nunique()}")
print(f"Total zones monitored:            {df['zone'].nunique()}")
print(f"Total after-hours swipes:         {len(df[after_hours])}")
print(f"Employees exceeding threshold:    {len(flagged_employees)}")
print(f"Deep night critical zone entries: {len(deep_night_flags)}")
# iloc[0] grabs the first row (highest count) since after_hours_counts is sorted descending
print(f"Top suspect:                      {after_hours_counts.iloc[0]['employee_id']} "
      f"({after_hours_counts.iloc[0]['after_hours_swipes']} after-hours swipes)")
print("\nRecommended Actions:")
print("  1. Review CCTV footage for deep night critical zone entries")
print("  2. Interview employees who exceeded the after-hours threshold")
print("  3. Audit badge permissions for employees flagged in Alert 2")
print("  4. Cross-reference with network login logs")
print("  5. Escalate to HR if access matches disciplinary records")
print("\n" + "="*55)
print("END OF REPORT")
print("="*55)
