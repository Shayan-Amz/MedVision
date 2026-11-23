"""DEPRECATED LEGACY SCRIPT.
Superseded by `audit_step1_nih14_25596_exact.py`, which fixes label mapping, Patient ID normalization,
and Patient-Level Cluster Paired Bootstrap Delta-AUROC 95% CIs. Always delegates execution directly.
"""
import subprocess, sys

if __name__ == "__main__":
    print("[DEPRECATED] Executing canonical pipeline: audit_step1_nih14_25596_exact.py", flush=True)
    subprocess.run([sys.executable, "./audit_step1_nih14_25596_exact.py"], check=True)
