"""
D2 / Risk A - Deserialization arbitrary-code-execution (an INGRESS threat).

We demonstrate the *mechanism* with a HARMLESS marker only: a crafted pickle
whose __reduce__ runs benign code (writes a marker file + prints a warning)
when loaded. NO malicious payload is used. We then show that picklescan flags
the crafted pickle, and that skops/ONNX artifacts do not carry executable code.

This isolates Risk A as a property of the *loading* step (i.e. of importing an
untrusted artifact INTO the environment), which is orthogonal to the egress
(leakage) question of Risk B.
"""
from __future__ import annotations
import os, pickle
import pandas as pd
from picklescan import scanner

MARKER = "RISK_A_MARKER.txt"


class BenignExecPayload:
    """On unpickling, __reduce__ triggers code execution. Payload is harmless:
    it writes a marker file. A real attacker could run anything here -- that is
    precisely the point. We never do."""
    def __reduce__(self):
        cmd = (
            "open(r'%s','w').write("
            "'arbitrary code executed at pickle-load time (benign marker)')"
            % os.path.join(os.getcwd(), MARKER)
        )
        return (eval, ("exec(%r)" % cmd,))


def run(out_dir: str) -> dict:
    art = os.path.join(out_dir, "artifacts")
    os.makedirs(art, exist_ok=True)
    marker_path = os.path.join(os.getcwd(), MARKER)
    if os.path.exists(marker_path):
        os.remove(marker_path)

    # 1. craft + write malicious-but-benign pickle
    mal_path = os.path.join(art, "risk_a_payload.pkl")
    with open(mal_path, "wb") as f:
        pickle.dump(BenignExecPayload(), f)

    # 2. scan BEFORE loading (the safe order)
    scan = scanner.scan_file_path(mal_path)
    detected = scan.issues_count > 0 or scan.infected_files > 0

    # 3. demonstrate execution on load (in this sandbox only)
    executed = False
    try:
        with open(mal_path, "rb") as f:
            pickle.load(f)
        executed = os.path.exists(marker_path)
    except Exception:
        executed = os.path.exists(marker_path)
    if os.path.exists(marker_path):
        os.remove(marker_path)

    result = {
        "pickle_executes_code_on_load": bool(executed),
        "picklescan_detected": bool(detected),
        "picklescan_globals_flagged": [
            f"{g.module}.{g.name}" for g in getattr(scan, "globals", [])
        ][:10],
        "note": ("Detection is heuristic: scanners flag known-dangerous globals "
                 "(eval/exec/os.system/posix.system) but can be bypassed by "
                 "obfuscation or novel gadget chains -- so scanning mitigates but "
                 "does not eliminate Risk A. Safe formats remove the attack surface."),
    }
    return result
