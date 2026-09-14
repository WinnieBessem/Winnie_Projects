"""
Cybersecurity Incident Response System
ITCS 3301 - Data Structures & Algorithms
Module V Assignment: Stacks & Queues
"""

import datetime
import time
import random


# ============================================================
# PART 1: SECURE STACK IMPLEMENTATION
# ============================================================

class SecureStack:
    """A stack implementation that supports secure memory erasure."""

    def __init__(self):
        self._data = []

    def push(self, item):
        """Add an item to the top of the stack."""
        self._data.append(item)

    def pop(self):
        """Remove and return the top item; raise exception if empty."""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def top(self):
        """Return the top item without removing it."""
        if self.is_empty():
            raise IndexError("top of empty stack")
        return self._data[-1]

    def is_empty(self):
        """Return True if the stack has no elements."""
        return len(self._data) == 0

    def __len__(self):
        """Return the number of elements currently stored."""
        return len(self._data)

    def secure_clear(self):
        """Clear the stack and overwrite all stored data with None."""
        for i in range(len(self._data)):
            self._data[i] = None
        self._data.clear()
        print("  [SecureStack] Memory securely erased.")


class FirewallAuditor:
    """Manages firewall rules using a SecureStack for audit trailing."""

    def __init__(self):
        self._rule_stack = SecureStack()
        self._audit_trail = []  # Full chronological history

    def add_rule(self, rule: str):
        """Push a firewall rule onto the stack with a timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {"rule": rule, "timestamp": timestamp, "action": "ADD"}
        self._rule_stack.push(entry)
        self._audit_trail.append(entry)
        print(f"  [+] Rule added    | {timestamp} | {rule}")

    def undo(self):
        """Pop and revert the most recent rule change."""
        if self._rule_stack.is_empty():
            print("  [!] Nothing to undo — stack is empty.")
            return
        removed = self._rule_stack.pop()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        undo_entry = {"rule": removed["rule"], "timestamp": timestamp, "action": "UNDO"}
        self._audit_trail.append(undo_entry)
        print(f"  [-] Rule undone   | {timestamp} | {removed['rule']}")

    def validate_syntax(self, expression: str) -> bool:
        """
        Validate bracket syntax using the parentheses-matching algorithm (Slide 8).
        Returns True if all brackets are properly matched, False otherwise.
        """
        stack = SecureStack()
        matching = {')': '(', ']': '[', '}': '{'}
        openers = set('([{')
        closers = set(')]}')

        for char in expression:
            if char in openers:
                stack.push(char)
            elif char in closers:
                if stack.is_empty() or stack.pop() != matching[char]:
                    result = False
                    print(f"  [VALIDATE] '{expression}' => INVALID")
                    return False

        result = stack.is_empty()
        status = "VALID" if result else "INVALID"
        print(f"  [VALIDATE] '{expression}' => {status}")
        return result

    def show_audit_trail(self):
        """Display the complete chronological history of rule changes."""
        print("\n  ====== FIREWALL AUDIT TRAIL ======")
        if not self._audit_trail:
            print("  (No history recorded)")
            return
        for i, entry in enumerate(self._audit_trail, 1):
            action_icon = "+" if entry["action"] == "ADD" else "-"
            print(f"  {i:>2}. [{action_icon}] {entry['action']:<5} | {entry['timestamp']} | {entry['rule']}")
        print(f"  Current rules in stack: {len(self._rule_stack)}")
        print("  ==================================\n")


# ============================================================
# PART 2: ALERT QUEUE SYSTEM
# ============================================================

class CircularAlertQueue:
    """
    Circular array-based queue (Slide 17 approach).
    Uses _data, _size, and _front instance variables.
    """

    DEFAULT_CAPACITY = 8

    def __init__(self):
        self._data = [None] * CircularAlertQueue.DEFAULT_CAPACITY
        self._size = 0
        self._front = 0

    def __len__(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def first(self):
        """Peek at the front alert without removing it."""
        if self.is_empty():
            raise IndexError("queue is empty")
        return self._data[self._front]

    def enqueue(self, alert):
        """Add an alert to the rear of the queue."""
        if self._size == len(self._data):
            self._resize(2 * len(self._data))
        rear = (self._front + self._size) % len(self._data)
        self._data[rear] = alert
        self._size += 1

    def dequeue(self):
        """Remove and return the front alert; raise exception if empty."""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        answer = self._data[self._front]
        self._data[self._front] = None  # Help garbage collection
        self._front = (self._front + 1) % len(self._data)
        self._size -= 1
        return answer

    def _resize(self, capacity):
        """Automatically double capacity when the queue is full."""
        old = self._data
        self._data = [None] * capacity
        walk = self._front
        for k in range(self._size):
            self._data[k] = old[walk]
            walk = (walk + 1) % len(old)
        self._front = 0
        print(f"    [Queue] Resized: {len(old)} -> {capacity} slots")


class SOCAlertProcessor:
    """
    Processes security alerts using separate priority queues.
    Always handles CRITICAL > HIGH > MEDIUM > LOW.
    """

    PRIORITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

    def __init__(self):
        self._queues = {level: CircularAlertQueue() for level in self.PRIORITY_ORDER}
        self._stats = {
            "total_received": 0,
            "total_processed": 0,
            "by_severity": {level: 0 for level in self.PRIORITY_ORDER}
        }

    def receive_alert(self, alert: dict):
        """Accept a security alert and enqueue it in the correct priority queue."""
        severity = alert.get("severity", "LOW").upper()
        if severity not in self._queues:
            severity = "LOW"
        self._queues[severity].enqueue(alert)
        self._stats["total_received"] += 1
        self._stats["by_severity"][severity] += 1

    def process_next(self):
        """Process the highest-severity non-empty queue first."""
        for level in self.PRIORITY_ORDER:
            if not self._queues[level].is_empty():
                alert = self._queues[level].dequeue()
                self._stats["total_processed"] += 1
                print(f"  [PROCESS] [{alert['severity']:<8}] Type: {alert['alert_type']:<20} "
                      f"| Source: {alert['source_ip']:<15} | {alert['timestamp']}")
                return alert
        print("  [PROCESS] All queues are empty.")
        return None

    def print_statistics(self):
        """Print processing statistics."""
        print("\n  ====== SOC ALERT STATISTICS ======")
        print(f"  Total Received : {self._stats['total_received']}")
        print(f"  Total Processed: {self._stats['total_processed']}")
        remaining = self._stats["total_received"] - self._stats["total_processed"]
        print(f"  Still Queued   : {remaining}")
        print("  --- By Severity ---")
        for level in self.PRIORITY_ORDER:
            count = self._stats["by_severity"][level]
            bar = "█" * count
            print(f"  {level:<8}: {count:>3}  {bar}")
        print("  ==================================\n")


# ============================================================
# PART 3: INTEGRATED INCIDENT RESPONSE SYSTEM
# ============================================================

class IncidentResponseSystem:
    """
    Combines SecureStack and CircularAlertQueue for a realistic SOC workflow.
    """

    def __init__(self):
        self._processor = SOCAlertProcessor()
        self._analyst_stacks = {}     # analyst_id -> SecureStack of actions
        self._incident_log = []       # Full incident report log
        self._processed_count = 0

    def _get_analyst_stack(self, analyst_id: str) -> SecureStack:
        if analyst_id not in self._analyst_stacks:
            self._analyst_stacks[analyst_id] = SecureStack()
        return self._analyst_stacks[analyst_id]

    def ingest_alert(self, alert: dict):
        """Step 1: Ingest new alerts into the priority queue."""
        self._processor.receive_alert(alert)

    def triage_next(self, analyst_id: str):
        """
        Step 2: Dequeue the highest-priority alert, then push triage action
        onto the analyst's action stack.
        """
        alert = self._processor.process_next()
        if alert is None:
            return None

        action = {
            "action": "TRIAGE",
            "alert": alert,
            "analyst": analyst_id,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "status": "ACTIVE"
        }
        stack = self._get_analyst_stack(analyst_id)
        stack.push(action)
        self._incident_log.append(action)
        self._processed_count += 1
        return action

    def undo_last_action(self, analyst_id: str):
        """
        Step 3: If an analyst makes a mistake, undo their last action
        using the stack.
        """
        stack = self._get_analyst_stack(analyst_id)
        if stack.is_empty():
            print(f"  [UNDO] Analyst {analyst_id}: No actions to undo.")
            return
        last = stack.pop()
        undo_record = {
            "action": "UNDO",
            "original_action": last,
            "analyst": analyst_id,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        }
        self._incident_log.append(undo_record)
        print(f"  [UNDO] Analyst {analyst_id} undid triage of "
              f"[{last['alert']['severity']}] {last['alert']['alert_type']}")

    def generate_incident_report(self):
        """Step 4: Generate a final incident report."""
        print("\n" + "=" * 60)
        print("          INCIDENT RESPONSE SYSTEM — FINAL REPORT")
        print("=" * 60)
        print(f"  Report Time : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Total Events Logged: {len(self._incident_log)}")
        print(f"  Alerts Processed: {self._processed_count}")

        print("\n  --- Queue Processing Order (Triaged Alerts) ---")
        count = 1
        for entry in self._incident_log:
            if entry["action"] == "TRIAGE":
                alert = entry["alert"]
                print(f"  {count:>3}. [{alert['severity']:<8}] {alert['alert_type']:<22} "
                      f"| Analyst: {entry['analyst']}")
                count += 1

        print("\n  --- Action Audit Trails by Analyst ---")
        for analyst_id, stack in self._analyst_stacks.items():
            print(f"  Analyst '{analyst_id}': {len(stack)} active action(s) in stack")

        print("\n  --- Key Metrics ---")
        self._processor.print_statistics()

    def dos_simulation(self):
        """
        Step 5: Simulate a DoS scenario — flood with LOW-severity alerts
        while CRITICALs are waiting. Demonstrate priority still holds.
        """
        print("\n" + "=" * 60)
        print("  [DoS SIMULATION] Flooding queue with LOW alerts...")
        print("=" * 60)

        # First, add a few CRITICAL alerts
        criticals = [
            {"severity": "CRITICAL", "alert_type": "Ransomware Detected",
             "source_ip": "192.168.1.50", "timestamp": _ts()},
            {"severity": "CRITICAL", "alert_type": "Root Compromise",
             "source_ip": "10.0.0.99", "timestamp": _ts()},
        ]
        for c in criticals:
            self.ingest_alert(c)
            print(f"  [INGEST] CRITICAL alert queued: {c['alert_type']}")

        # Now flood with 50 LOW alerts
        print(f"\n  [DoS] Injecting 50 LOW-severity flood alerts...")
        for i in range(50):
            self.ingest_alert({
                "severity": "LOW",
                "alert_type": f"Port Scan #{i+1}",
                "source_ip": f"203.0.113.{i % 256}",
                "timestamp": _ts()
            })

        print(f"\n  [DoS] Queue flooded. Now processing — CRITICALs must come first!\n")

        # Process first 5 to demonstrate priority
        for _ in range(5):
            self._processor.process_next()

        print(f"\n  [DoS] Result: CRITICAL alerts processed BEFORE any LOW alert.")
        print("  Priority queue successfully defends against DoS queue exhaustion!\n")


# ============================================================
# HELPERS
# ============================================================

def _ts():
    """Return current timestamp string."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def make_alert(severity, alert_type, source_ip):
    return {
        "severity": severity,
        "alert_type": alert_type,
        "source_ip": source_ip,
        "timestamp": _ts()
    }


# ============================================================
# MAIN DEMONSTRATION
# ============================================================

def main():
    sep = "=" * 60

    # --------------------------------------------------------
    # PART 1 DEMO: FirewallAuditor
    # --------------------------------------------------------
    print(sep)
    print("  PART 1: SECURE STACK — FIREWALL AUDITOR")
    print(sep)

    auditor = FirewallAuditor()

    # Add 5 rules
    rules = [
        "ALLOW TCP 443 FROM 10.0.0.0/8",
        "ALLOW TCP 80 FROM ANY",
        "DENY UDP 53 FROM 192.168.0.0/16",
        "ALLOW ICMP FROM 172.16.0.0/12",
        "DENY TCP 22 FROM 0.0.0.0/0",
    ]
    print("\n  >> Adding 5 firewall rules:")
    for rule in rules:
        auditor.add_rule(rule)

    # Undo 2 times
    print("\n  >> Performing 2 undo operations:")
    auditor.undo()
    auditor.undo()

    # Validate 3 expressions (mix of valid/invalid)
    print("\n  >> Validating 3 rule syntax expressions:")
    expressions = [
        "ALLOW (TCP 443) FROM [10.0.0.0/8]",    # valid
        "DENY (UDP 53 FROM {192.168.0.1})",       # valid
        "ALLOW (TCP 80 FROM [192.168.1.0/24)",    # invalid — mismatched
    ]
    for expr in expressions:
        auditor.validate_syntax(expr)

    # Show full audit trail
    auditor.show_audit_trail()

    # --------------------------------------------------------
    # PART 2 DEMO: SOCAlertProcessor
    # --------------------------------------------------------
    print(sep)
    print("  PART 2: CIRCULAR ALERT QUEUE — SOC ALERT PROCESSOR")
    print(sep)

    processor = SOCAlertProcessor()

    alerts = [
        make_alert("CRITICAL", "SQL Injection Attack",      "203.0.113.10"),
        make_alert("HIGH",     "Brute Force Login",         "198.51.100.5"),
        make_alert("MEDIUM",   "Suspicious Outbound DNS",   "10.0.1.22"),
        make_alert("LOW",      "Port Scan Detected",        "192.168.10.3"),
        make_alert("CRITICAL", "Ransomware Behavior",       "10.0.0.55"),
        make_alert("HIGH",     "Privilege Escalation",      "172.16.0.10"),
        make_alert("LOW",      "Failed SSH Login x3",       "203.0.113.77"),
        make_alert("MEDIUM",   "Unusual Data Transfer",     "10.0.2.8"),
        make_alert("HIGH",     "Malware C2 Callback",       "198.51.100.20"),
        make_alert("CRITICAL", "Data Exfiltration",         "10.0.0.100"),
        make_alert("LOW",      "ICMP Flood",                "203.0.113.1"),
        make_alert("MEDIUM",   "SMB Lateral Movement",      "192.168.1.5"),
        make_alert("HIGH",     "Credential Dumping",        "172.16.0.33"),
        make_alert("LOW",      "Unauthorized Port Access",  "10.0.1.99"),
        make_alert("CRITICAL", "Kernel Exploit Attempt",    "192.168.0.200"),
    ]

    print("\n  >> Ingesting 15 alerts across all severity levels:")
    for alert in alerts:
        processor.receive_alert(alert)
        print(f"  [RECV] [{alert['severity']:<8}] {alert['alert_type']}")

    print("\n  >> Processing all 15 alerts (highest severity first):")
    while True:
        result = processor.process_next()
        if result is None:
            break

    processor.print_statistics()

    # --------------------------------------------------------
    # PART 3 DEMO: Integrated Incident Response System
    # --------------------------------------------------------
    print(sep)
    print("  PART 3: INTEGRATED INCIDENT RESPONSE SYSTEM")
    print(sep)

    irs = IncidentResponseSystem()

    # Build a stream of 30 events
    stream = [
        make_alert("CRITICAL", "Zero-Day Exploit",          "10.0.0.1"),
        make_alert("HIGH",     "Spear Phishing Email",      "198.51.100.3"),
        make_alert("MEDIUM",   "Abnormal Login Time",       "192.168.5.5"),
        make_alert("LOW",      "Config Change",             "10.0.2.1"),
        make_alert("CRITICAL", "Lateral Movement Detected", "172.16.1.1"),
        make_alert("HIGH",     "Pass-the-Hash Attack",      "192.168.1.8"),
        make_alert("MEDIUM",   "Unusual Registry Edit",     "10.0.1.15"),
        make_alert("CRITICAL", "Ransomware Spreading",      "10.0.0.50"),
        make_alert("LOW",      "DNS Lookup Anomaly",        "192.168.2.2"),
        make_alert("HIGH",     "Reverse Shell Opened",      "198.51.100.9"),
        make_alert("MEDIUM",   "Large File Upload",         "10.0.3.5"),
        make_alert("CRITICAL", "AD Kerberoasting",          "172.16.0.5"),
        make_alert("LOW",      "Network Scan",              "203.0.113.4"),
        make_alert("HIGH",     "DLL Injection",             "192.168.1.50"),
        make_alert("MEDIUM",   "Scheduled Task Created",    "10.0.0.88"),
        make_alert("LOW",      "USB Device Inserted",       "10.0.1.10"),
        make_alert("CRITICAL", "Wiper Malware Detected",    "10.0.0.77"),
        make_alert("HIGH",     "Shadow Copy Deletion",      "172.16.0.8"),
        make_alert("MEDIUM",   "HTTP Beaconing",            "192.168.3.3"),
        make_alert("LOW",      "Cleartext Password Found",  "10.0.2.99"),
        make_alert("HIGH",     "Token Impersonation",       "198.51.100.15"),
        make_alert("MEDIUM",   "Process Hollowing",         "10.0.1.77"),
        make_alert("CRITICAL", "Domain Controller Breach",  "172.16.0.1"),
        make_alert("LOW",      "Failed RDP Login",          "203.0.113.20"),
        make_alert("HIGH",     "Firewall Rule Deleted",     "192.168.0.10"),
        make_alert("MEDIUM",   "SSL Certificate Anomaly",   "10.0.3.9"),
        make_alert("LOW",      "Unused Account Login",      "10.0.1.5"),
        make_alert("CRITICAL", "Backdoor Installed",        "192.168.100.1"),
        make_alert("HIGH",     "Memory Injection",          "172.16.0.20"),
        make_alert("MEDIUM",   "PowerShell Obfuscation",    "10.0.0.44"),
    ]

    # Step 1: Ingest all 30 alerts
    print(f"\n  >> [Step 1] Ingesting 30 alerts into priority queues:")
    for alert in stream:
        irs.ingest_alert(alert)
    print(f"  30 alerts ingested across CRITICAL/HIGH/MEDIUM/LOW queues.")

    # Step 2: Triage 20 alerts with 2 analysts
    print(f"\n  >> [Step 2] Triaging alerts — Analysts Alice & Bob:")
    analysts = ["Alice", "Bob"]
    for i in range(20):
        analyst = analysts[i % 2]
        irs.triage_next(analyst)

    # Step 3: Undo last action for both analysts
    print(f"\n  >> [Step 3] Both analysts undo their most recent action:")
    irs.undo_last_action("Alice")
    irs.undo_last_action("Bob")

    # Step 4: Generate incident report
    print(f"\n  >> [Step 4] Generating Incident Report:")
    irs.generate_incident_report()

    # Step 5: DoS simulation
    print(f"\n  >> [Step 5] DoS Simulation:")
    irs2 = IncidentResponseSystem()
    irs2.dos_simulation()

    print(sep)
    print("  ALL PARTS COMPLETE — Simulation finished successfully.")
    print(sep)


if __name__ == "__main__":
    main()