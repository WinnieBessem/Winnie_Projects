# =============================================================
# Insider Threat Detector - Badge Access Log Analysis
# Cyber Data Scenario Challenge - Week 2 Homework
# Upgraded from Week 1: added merge, 300 rows, 6 cleaning steps
# =============================================================

# pandas: our main tool for working with tabular data (DataFrames)
import pandas as pd
# numpy: used to generate random numbers and arrays for fake data
import numpy as np
# matplotlib: used to draw charts and save them as image files
import matplotlib.pyplot as plt


# =============================================================
# STAGE 1 - COLLECT
# Goal: Build a realistic 300-row badge access log dataset and
# a second employee profile table we will merge in Stage 5.
# We generate the data with code because we don't have a real
# badge system - but the column structure mirrors real logs.
# =============================================================

# np.random.seed(42) locks the random number generator so the
# script produces the SAME fake data every time it runs.
# Without this, every run gives different numbers, making
# testing and debugging much harder.
np.random.seed(42)

# Total number of badge swipe records to generate this week.
# Week 1 required 200 rows - Week 2 bumps it to 350.
num_records = 350

# List of 10 employee IDs: EMP001, EMP002, ... EMP010
# zfill(3) pads with leading zeros so EMP1 becomes EMP001
employee_ids = [f"EMP{str(i).zfill(3)}" for i in range(1, 11)]

# The five physical zones in the building employees can badge into.
# Server Room and Data Center are "restricted" - we flag those later.
zones = ["Server Room", "Executive Floor",
         "Data Center", "HR Office", "Main Lobby"]

# --- Generate random timestamps spread across 30 days ---
# We want timestamps scattered across nights and weekends, not just
# 9-to-5 business hours, because insider threats often happen off-hours.
# Strategy: pick a random number of seconds between 0 and 30 days worth
# of seconds, then add that offset to our start date.
start_date = pd.Timestamp("2024-01-01")
# 30 days * 24 hours * 60 minutes * 60 seconds = 2,592,000 seconds
random_seconds = np.random.randint(0, 30 * 24 * 60 * 60, size=num_records)
# Build a list of actual datetime objects by adding each offset to start_date
timestamps = [start_date + pd.Timedelta(seconds=int(s))
              for s in random_seconds]

# --- Build the main DataFrame ---
# np.random.choice picks random values from each list for every row.
# p=[0.75, 0.25] means 75% of swipes are granted, 25% denied.
# That ratio is realistic - most badge swipes succeed.
df = pd.DataFrame({
    "employee_id":   np.random.choice(employee_ids, size=num_records),
    "zone":          np.random.choice(zones, size=num_records),
    "timestamp":     timestamps,
    "access_result": np.random.choice(["granted", "denied"],
                                      size=num_records, p=[0.75, 0.25])
})

# --- Inject intentional data quality problems for cleaning practice ---
# Real log exports from different systems often have inconsistent formatting.
# We inject problems here so Stage 2 has real work to do.

# Problem 1: Mixed casing in access_result - simulates two log sources
# using different conventions ("granted" vs "GRANTED")
df.loc[np.random.choice(df.index, 20, replace=False),
       "access_result"] = "GRANTED"

# Problem 2: Mixed casing in zone - same cause, different export format
df.loc[np.random.choice(df.index, 10, replace=False), "zone"] = "SERVER ROOM"

# Problem 3: Duplicate rows - simulates overlapping SIEM export windows
# pd.concat stacks the original df and 10 random rows on top of each other
df = pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)

# Problem 4: Missing employee IDs - simulates a badge reader that failed
# to read the card but still logged the door event
df.loc[np.random.choice(df.index, 5, replace=False), "employee_id"] = np.nan

# Problem 5: Corrupt timestamps - simulates a sensor that sent garbled data
# FIX: We must convert timestamp to string (object) dtype BEFORE injecting
# "BAD_TIMESTAMP" as a string value. The column was created as datetime64
# by pandas, and you cannot assign a plain string into a datetime64 column.
# Converting to str first makes the column accept any string value,
# and then pd.to_datetime() in Stage 2 will handle the bad ones gracefully.
df["timestamp"] = df["timestamp"].astype(str)
df.loc[np.random.choice(df.index, 3, replace=False),
       "timestamp"] = "BAD_TIMESTAMP"

# --- Build the second DataFrame: Employee Profile Table ---
# This is the enrichment source for the required pd.merge() in Stage 5.
# It gives us department, role, and clearance_level for each employee.
# Before the merge we only know WHO swiped and WHEN.
# After the merge we know if they SHOULD have access to that zone at all.
employee_profiles = pd.DataFrame({
    "employee_id":     [f"EMP{str(i).zfill(3)}" for i in range(1, 11)],
    "department":      ["IT", "HR", "Finance", "IT", "Facilities",
                        "Executive", "IT", "HR", "Finance", "Security"],
    "role":            ["sysadmin", "recruiter", "analyst", "network_eng", "janitor",
                        "VP", "developer", "manager", "accountant", "guard"],
    "clearance_level": ["high", "low", "medium", "high", "low",
                        "high", "medium", "low", "medium", "high"]
})

# --- 5 required inspection commands ---
# These confirm the data was built correctly before we start cleaning.
print("Shape:",              df.shape)            # (rows, columns)
print("\nData Types:\n",     df.dtypes)            # confirms column types
print("\nFirst 5 Rows:\n",   df.head())            # visual spot-check
print("\nSummary:\n",        df.describe(include="all"))  # stats per column
print("\nMissing Values:\n", df.isnull().sum())    # counts empty cells

# Data quality problems identified during inspection:
# - timestamp is dtype 'object' (string) instead of datetime64
# - access_result has both "granted" and "GRANTED" as distinct values
# - zone has both "Server Room" and "SERVER ROOM" as distinct values
# - employee_id has NaN entries from the badge reader failures
# - duplicate rows exist from the overlapping export window


# =============================================================
# STAGE 2 - CLEAN
# Goal: Fix all 5 data quality problems found during inspection.
# Cleaning MUST happen before analysis - if "granted" and "GRANTED"
# both exist, groupby() will count them as two different values and
# all our access statistics will be wrong.
# =============================================================

# Save the row count before any cleaning so we can report the difference
rows_before = len(df)

# --- Technique 1: Timestamp parsing with error handling ---
# pd.to_datetime() converts string timestamps to proper datetime objects.
# errors="coerce" is critical here: instead of crashing when it hits
# "BAD_TIMESTAMP", it silently replaces that value with NaT (Not a Time).
# After coercion, we drop those NaT rows because we cannot do any
# time-based analysis (hour of day, date grouping) without a valid timestamp.
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df.dropna(subset=["timestamp"], inplace=True)
# inplace=True modifies df directly instead of returning a new DataFrame

# --- Technique 2: String normalization on zone ---
# .str.strip() removes any accidental leading or trailing spaces
# .str.lower() converts "SERVER ROOM" to "server room" so it matches
# "Server Room" which also becomes "server room" - both unified
df["zone"] = df["zone"].str.strip().str.lower()

# --- Technique 3: String normalization on access_result ---
# Same reason as zone - "GRANTED" and "granted" must be unified
# or every filter like df[df["access_result"] == "granted"] will
# miss the uppercase rows and undercount granted swipes
df["access_result"] = df["access_result"].str.strip().str.lower()

# --- Technique 4: Fill missing employee IDs ---
# We use fillna("UNKNOWN") instead of dropping these rows entirely.
# Reasoning: a badge swipe with no employee ID is still a real security
# event - someone triggered a door sensor. Dropping it would hide the
# incident from our analysis entirely, which is worse than keeping it
# with a placeholder label.
df["employee_id"] = df["employee_id"].fillna("UNKNOWN")

# --- Technique 5: Remove duplicate rows ---
# drop_duplicates() with no subset argument checks ALL columns at once.
# A row is only considered a duplicate if every single column matches
# another row exactly. This prevents accidentally removing legitimate
# swipes where the same employee entered the same zone twice.
df.drop_duplicates(inplace=True)

# --- Technique 6: Regex extraction on employee_id ---
# .str.extract() uses a regular expression to pull a specific part
# of a string out into a new column.
# Pattern breakdown: r"EMP(\d+)"
#   - EMP    = matches the literal characters E, M, P
#   - (\d+)  = the parentheses create a "capture group"
#              \d means "any digit", + means "one or more of them"
#   - extract() returns ONLY what is inside the parentheses
# Result: "EMP007" becomes "7", then .astype(float) converts to 7.0
# Rows that don't match (like "UNKNOWN") return NaN in this column
df["emp_number"] = df["employee_id"].str.extract(r"EMP(\d+)").astype(float)

# Report how many rows were removed across all cleaning steps
print(f"\nRows before cleaning: {rows_before} | "
      f"Rows after: {len(df)} | "
      f"Removed: {rows_before - len(df)}")
print("\nMissing values after cleaning:\n", df.isnull().sum())
print("\nCleaned preview:\n", df.head())


# =============================================================
# STAGE 3 - ANALYZE
# Goal: Answer 3 security questions using GroupBy and .agg().
# Each finding is printed with a clear label so someone reading
# the output can immediately understand what they are looking at.
# =============================================================

# Extract hour (0-23) and calendar date from the timestamp column.
# We store these as separate columns so we can group and filter by
# time of day and date without re-parsing the timestamp each time.
df["hour"] = df["timestamp"].dt.hour   # .dt accesses datetime properties
df["date"] = df["timestamp"].dt.date   # returns a Python date object

# Define "after hours" as before 7am OR at/after 7pm.
# The pipe | means OR - a row is after-hours if either condition is true.
# We store this as a boolean mask (True/False per row) to reuse it
# across multiple questions without rewriting the filter each time.
after_hours = (df["hour"] < 7) | (df["hour"] >= 19)

# The two zones that require the highest security clearance.
# Any after-hours access to these zones is a priority flag.
restricted_zones = ["server room", "data center"]

# --- Question 1: After-hours access to restricted zones ---
# Apply both filters simultaneously using & (AND).
# The row must be after-hours AND in a restricted zone to appear here.
# Parentheses around each condition are required - without them Python
# evaluates & before < and >= and produces a wrong result.
q1 = df[after_hours & df["zone"].isin(restricted_zones)]

print("\n" + "="*60)
print("FINDING 1: After-hours restricted zone access")
print("="*60)
print(f"Total swipes matching this pattern: {len(q1)}")
print("Employees involved:", q1["employee_id"].unique())
# to_string(index=False) prints the table without the row index numbers
print(q1[["employee_id", "zone", "timestamp",
          "hour", "access_result"]].to_string(index=False))

# --- Question 2: After-hours swipe volume AND denial rate per employee ---
# Named aggregations let us compute multiple statistics in one .agg() call
# and give each result column a descriptive name.
# Tuple format: new_column_name = (source_column, aggregation_function)
#
# deny_rate logic: (x == "denied") creates a True/False Series.
# Python treats True as 1 and False as 0, so:
#   .sum()  = counts how many rows were denied
#   .mean() = gives the FRACTION denied (e.g. 0.40 means 40% were denied)
# A high deny_rate means someone is repeatedly trying doors they can't open,
# which is a classic sign of probing for unauthorized access.
after_hours_agg = (
    df[after_hours]
    .groupby("employee_id")
    .agg(
        after_hours_swipes=("access_result", "count"),
        denied_count=("access_result", lambda x: (x == "denied").sum()),
        deny_rate=("access_result", lambda x: round((x == "denied").mean(), 2))
    )
    .sort_values("after_hours_swipes", ascending=False)
    .reset_index()
    # reset_index() moves employee_id from the index back into a regular column
)

print("\n" + "="*60)
print("FINDING 2: After-hours volume and denial rate per employee")
print("="*60)
print(after_hours_agg.to_string(index=False))

# --- Question 3: Employees denied in zones they normally access ---
# Approach: build two sets - one for GRANTED events and one for DENIED.
# Then find employees who appear in BOTH sets for the SAME zone.
# That overlap is suspicious: they normally get in, but were also turned
# away at some point. Could mean badge revoked, cloned, or access downgraded.
granted = df[df["access_result"] == "granted"][
    ["employee_id", "zone"]].drop_duplicates()
denied = df[df["access_result"] == "denied"][
    ["employee_id", "zone"]].drop_duplicates()

# how="inner" keeps ONLY rows that appear in BOTH DataFrames.
# The merge key is the combination of both columns, so the same
# employee must appear in the same zone in both tables to be flagged.
suspicious_denials = pd.merge(
    granted, denied, on=["employee_id", "zone"], how="inner")

print("\n" + "="*60)
print("FINDING 3: Employees denied in zones they normally access")
print("="*60)
if len(suspicious_denials) == 0:
    print("No suspicious denial patterns found.")
else:
    print(f"Flagged employee/zone pairs: {len(suspicious_denials)}")
    print(suspicious_denials.to_string(index=False))


# =============================================================
# STAGE 4 - VISUALIZE
# Goal: Turn the Stage 3 findings into charts.
# Charts let you spot a spike, outlier, or pattern at a glance
# instead of reading through rows of printed numbers.
# Chart 1: WHO is most suspicious (ranking bar chart)
# Chart 2: WHEN suspicious activity peaks (hourly line chart)
# Chart 3: WHICH departments have the most after-hours activity
#           (only possible after the merge in Stage 5 - built last)
# =============================================================

# --- Chart 1: Top employees by after-hours access attempts ---
# A tall red bar means that employee badged in many times outside
# business hours - the taller, the more suspicious.
top10 = after_hours_agg.head(10)  # show only the 10 most suspicious employees
fig, ax = plt.subplots(figsize=(10, 6))  # create a 10 x 6 inch figure
ax.bar(top10["employee_id"], top10["after_hours_swipes"],
       color="crimson", label="After-Hours Swipes")
ax.set_title("Top Employees by After-Hours Access Attempts", fontsize=14)
ax.set_xlabel("Employee ID", fontsize=12)
ax.set_ylabel("Number of After-Hours Swipes", fontsize=12)
plt.xticks(rotation=45, ha="right")  # rotate labels to prevent overlap
ax.legend(loc="upper right")
ax.yaxis.grid(True, color="gray", alpha=0.7)  # horizontal gridlines only
plt.tight_layout()   # adjusts spacing so labels are not cut off at edges
plt.savefig("chart1_top_afterhours.png", dpi=150)
# savefig() must come BEFORE show() - after show() the figure is cleared
plt.show()
print("Chart 1 saved.")

# --- Chart 2: After-hours swipe activity by hour of day ---
# This shows WHEN the suspicious activity happens.
# A spike at 3am is far more alarming than a spike at 6:30am
# (which could just be an early employee arriving before 7am).
hourly_counts = (
    df[after_hours]
    .groupby("hour")
    .size()               # count rows per hour bucket
    .reset_index(name="swipe_count")
)

fig, ax = plt.subplots(figsize=(10, 6))
# marker="o" draws a visible dot at each data point
ax.plot(hourly_counts["hour"], hourly_counts["swipe_count"],
        marker="o", color="darkorange", linewidth=2, label="Swipes Per Hour")
# axvspan shades the area between x=0 and x=5 (midnight to 5am) in light red
# This draws the reader's eye to the most critical time window immediately
ax.axvspan(0, 5, alpha=0.1, color="red", label="Deep Night (12am-5am)")
ax.set_title("After-Hours Badge Swipe Activity by Hour of Day", fontsize=14)
ax.set_xlabel("Hour of Day (24-Hour Format)", fontsize=12)
ax.set_ylabel("Number of Swipes", fontsize=12)
# Only show tick marks for actual after-hours values to avoid decimals like 19.5
ax.set_xticks(list(range(0, 7)) + list(range(19, 24)))
ax.legend(loc="upper right")
ax.grid(True, color="black", alpha=0.5)
plt.tight_layout()
plt.savefig("chart2_hourly_activity.png", dpi=150)
plt.show()
print("Chart 2 saved.")


# =============================================================
# STAGE 5 - ACT AND ENRICH
# Two components this week:
# Part A: Automated alerting checks (threshold + deep night flag)
# Part B: The required pd.merge() with the employee profile table,
#         followed by a new analysis question only possible post-merge
# =============================================================

# --- Alert 1: Flag employees who exceeded the after-hours swipe limit ---
# Storing the limit as a variable means we only change one number
# if the security policy changes, instead of hunting through the code.
print("\n" + "="*60)
print("ALERT 1: After-Hours Threshold Check (limit = 5 swipes)")
print("="*60)
threshold = 5
# Boolean filter: only keep rows where the swipe count exceeds our limit
flagged = after_hours_agg[after_hours_agg["after_hours_swipes"] > threshold]
if len(flagged) == 0:
    print(f"No employees exceeded {threshold} after-hours swipes.")
else:
    # .iterrows() lets us loop through each flagged employee one at a time
    # _ is a Python convention meaning "I don't need the index, just the row data"
    for _, row in flagged.iterrows():
        print(f"  ALERT: {row['employee_id']} --- "
              f"{row['after_hours_swipes']} after-hours swipes | "
              f"deny rate: {row['deny_rate']}")

# --- Alert 2: Granted access to critical zones between 12am and 5am ---
# Even a SINGLE successful Server Room entry at 2am is serious enough
# to warrant immediate investigation - this alert catches those events.
print("\n" + "="*60)
print("ALERT 2: Deep Night Critical Zone Access (12am-5am, granted only)")
print("="*60)
# range(0, 5) = [0, 1, 2, 3, 4] which covers midnight through 4:59am
# We only flag "granted" entries because denied attempts are less urgent
deep_night_flags = df[
    df["zone"].isin(["server room", "data center"]) &
    df["hour"].isin(range(0, 5)) &
    (df["access_result"] == "granted")
]
if len(deep_night_flags) == 0:
    print("No critical zone access detected between 12am and 5am.")
else:
    print(f"WARNING: {len(deep_night_flags)} entries found:\n")
    print(deep_night_flags[["employee_id", "zone",
          "timestamp", "hour"]].to_string(index=False))

# --- THE REQUIRED MERGE ---
# pd.merge() joins two DataFrames on a shared column (employee_id).
#
# how="left" means: keep EVERY row from the left DataFrame (df),
# and attach matching profile columns where a match exists.
# If an employee_id has no match in employee_profiles (e.g. "UNKNOWN"),
# that row is still kept but gets NaN in the new profile columns.
#
# Why not how="inner"?
# An inner join would silently DROP rows with no matching profile.
# In a security context that is dangerous - an unidentified person
# badging into a door IS the incident we are trying to detect.
#
# What the merge adds:
#   BEFORE: we know WHO swiped and WHEN
#   AFTER:  we also know their ROLE, DEPARTMENT, and CLEARANCE LEVEL
print("\n" + "="*60)
print("ENRICHMENT: Merging employee profiles into badge log")
print("="*60)
df_enriched = pd.merge(df, employee_profiles, on="employee_id", how="left")
print(f"Rows before merge: {len(df)} | Rows after merge: {len(df_enriched)}")
print("New columns added:", [
      c for c in df_enriched.columns if c not in df.columns])
print("\nEnriched preview:")
print(df_enriched[["employee_id", "zone", "access_result",
                   "department", "role",
                   "clearance_level"]].head(8).to_string(index=False))

# --- Post-merge analysis: Clearance violations ---
# This question was IMPOSSIBLE before the merge because clearance_level
# did not exist in the original badge log.
# Low-clearance employees should never be granted access to high-security
# zones. If it happened, either the access control list has an error
# or someone tailgated a legitimate employee through the door.
high_security_zones = ["server room", "data center"]
clearance_violations = df_enriched[
    df_enriched["zone"].isin(high_security_zones) &    # restricted area
    (df_enriched["clearance_level"] == "low") &        # low-clearance person
    (df_enriched["access_result"] == "granted")        # actually got in
]

print("\n" + "="*60)
print("ENRICHED FINDING: Low-clearance employees granted high-security access")
print("(This finding was impossible before the merge)")
print("="*60)
if len(clearance_violations) == 0:
    print("No clearance violations found.")
else:
    print(f"Total violations: {len(clearance_violations)}\n")
    print(clearance_violations[["employee_id", "role", "department",
                                "clearance_level", "zone",
                                "timestamp"]].to_string(index=False))

# --- Chart 3: After-hours violations by department (post-merge data) ---
# This chart only exists because of the merge - "department" was not
# in the original badge log. It shows whether after-hours activity is
# concentrated in one team (e.g. IT working late legitimately) or
# spread strangely across departments that have no reason to be in
# the building overnight.
dept_afterhours = (
    # NOTE: We rebuild the after_hours condition directly on df_enriched here
    # instead of reusing the boolean mask defined in Stage 3.
    # Reason: pd.merge() resets the index of df_enriched, so the old mask
    # (built against df's index) no longer aligns and raises an IndexingError.
    # Always rebuild boolean masks when applying them to a DataFrame whose
    # index differs from the one the mask was originally created from.
    df_enriched[(df_enriched["hour"] < 7) | (df_enriched["hour"] >= 19)]
    .groupby("department")
    .size()
    .sort_values(ascending=False)
    .reset_index(name="after_hours_swipes")
)
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(dept_afterhours["department"], dept_afterhours["after_hours_swipes"],
       color="steelblue", label="After-Hours Swipes")
ax.set_title(
    "After-Hours Swipes by Department (Post-Merge Finding)", fontsize=14)
ax.set_xlabel("Department", fontsize=12)
ax.set_ylabel("Number of After-Hours Swipes", fontsize=12)
plt.xticks(rotation=45, ha="right")
ax.legend(loc="upper right")
ax.yaxis.grid(True, color="gray", alpha=0.7)
plt.tight_layout()
plt.savefig("chart3_dept_afterhours.png", dpi=150)
plt.show()
print("Chart 3 saved.")

# --- Final Summary Report ---
# Combines all key numbers into one printout for a manager or SOC team.
# This is the deliverable someone would actually act on.
print("\n" + "="*60)
print("FINAL SUMMARY REPORT")
print("="*60)
print(f"Total records analyzed:              {len(df_enriched)}")
# nunique() counts distinct values - tells us how many different people badged
print(
    f"Unique employees in log:             {df_enriched['employee_id'].nunique()}")
print(f"Zones monitored:                     {df_enriched['zone'].nunique()}")
# Rebuild the after_hours filter on df_enriched (same index fix as Chart 3)
after_hours_enriched = (df_enriched["hour"] < 7) | (df_enriched["hour"] >= 19)
print(
    f"Total after-hours swipes:            {len(df_enriched[after_hours_enriched])}")
print(f"Employees exceeding threshold:       {len(flagged)}")
print(f"Deep night critical zone entries:    {len(deep_night_flags)}")
print(f"Clearance violations (post-merge):   {len(clearance_violations)}")
# iloc[0] grabs the first row since after_hours_agg is sorted descending -
# the employee with the highest count is always in position 0
print(f"Top suspect:                         {after_hours_agg.iloc[0]['employee_id']} "
      f"({after_hours_agg.iloc[0]['after_hours_swipes']} after-hours swipes)")
print("\nRecommended Actions:")
print("  1. Review CCTV footage for all deep night critical zone entries")
print("  2. Interview employees flagged for clearance violations")
print("  3. Audit badge permissions for employees exceeding the threshold")
print("  4. Cross-reference with network login logs for same timeframes")
print("  5. Escalate to HR if access pattern conflicts with employee role")
print("\n" + "="*60)
print("END OF REPORT")
print("="*60)
