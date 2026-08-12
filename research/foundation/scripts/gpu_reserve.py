#!/usr/bin/env python3
"""Reserve N GPUs on the shared lab box, claiming only STABLY-free cards.

Why this exists alongside /home/liangsheng/brian/acquire_gpus.py
---------------------------------------------------------------
acquire_gpus.py claims a card the moment nvidia-smi shows no compute process on
it. On this box that is not careful enough. Another user restarts vLLM servers
often, and during a restart their cards look free for one to three minutes. A
claim in that window takes a card away from a job that only reloads weights.
CLAUDE.md states the rule: wait for a stable free window, never race.

This script keeps the safety rule and adds three things:

  1. STABILITY GATE -- a card must look free continuously for --stable-seconds
     before this script claims it. A server restart never satisfies the gate.
  2. SESSION SURVIVAL -- gpu_reserve_supervisor.sh runs it under setsid,
     restarts it if it dies, and forwards SIGTERM so the cards are released.
  3. STATUS FILE -- JSON written every poll, so anyone can see the state
     without reading the log.

This file is self-contained on purpose. It does not import acquire_gpus.py,
because that file lives outside the repository and a container wipe deletes it.

THE SAFETY RULE IS ABSOLUTE. A card that has any compute process on it is never
claimed, never killed, and never co-located on. This script only ever waits.
"""

import argparse
import ctypes
import fcntl
import json
import os
import signal
import subprocess
import sys
import time

MIB = 1024 * 1024
LOCK_DIR = "/tmp/gpu_locks"
FREE_MEM_MIB = 1024          # a card above this is not free, even with no process
MIN_HOLD_BYTES = 32 * MIB    # fallback hold: small, but visible in nvidia-smi


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---------- nvidia-smi queries ----------

def gpu_table():
    """Returns [(index, uuid, memory_used_mib)] for every GPU."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        if not line.strip():
            continue
        idx, uuid, used = [p.strip() for p in line.split(",")]
        rows.append((int(idx), uuid, int(used)))
    return rows


def busy_uuids():
    """UUIDs of GPUs that have at least one compute process attached."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout
    return {line.strip() for line in out.splitlines() if line.strip()}


# ---------- claim machinery ----------

class GPULock:
    """Non-blocking flock on a per-GPU lockfile, held for the process lifetime.

    This stops two copies of this script from claiming the same card. It does
    NOT stop other users -- only the CUDA hold below is visible to them.
    """

    def __init__(self, lock_dir, index):
        os.makedirs(lock_dir, exist_ok=True)
        self.path = os.path.join(lock_dir, f"gpu_{index}.lock")
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)

    def acquire(self):
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(self.fd, 0)
            os.write(self.fd, f"{os.getpid()}\n".encode())
            return True
        except BlockingIOError:
            os.close(self.fd)
            return False

    def release(self):
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(self.fd)
        except OSError:
            pass


class CudaHold:
    """A real CUDA context plus an allocation, so others see the card as taken.

    The allocation size is --hold-frac of the card's free memory. A large hold
    is what actually reserves the card: a 32MB hold is visible in nvidia-smi but
    does not stop another job from allocating the rest and co-locating.
    """

    def __init__(self, index, frac):
        self.index = index
        self.held_bytes = 0
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.dev = ctypes.c_int()
        self.ctx = ctypes.c_void_p()
        self.ptr = ctypes.c_void_p()
        self._ck(self.lib.cuInit(0), "cuInit")
        self._ck(self.lib.cuDeviceGet(ctypes.byref(self.dev), index), "cuDeviceGet")
        self._ck(self.lib.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, self.dev),
                 "cuCtxCreate")
        try:
            free_b, total_b = ctypes.c_size_t(), ctypes.c_size_t()
            self._ck(self.lib.cuMemGetInfo_v2(ctypes.byref(free_b),
                                              ctypes.byref(total_b)),
                     "cuMemGetInfo")
            want = int(free_b.value * frac)
            # Step down until one size succeeds, so a slightly-too-large request
            # degrades to a smaller hold instead of losing the card entirely.
            for size in (want, want // 2, want // 4, MIN_HOLD_BYTES):
                size = max(int(size), MIN_HOLD_BYTES)
                if self.lib.cuMemAlloc_v2(ctypes.byref(self.ptr),
                                          ctypes.c_size_t(size)) == 0:
                    self.held_bytes = size
                    return
            raise RuntimeError(f"GPU {index}: no hold buffer could be allocated")
        except Exception:
            self.release()
            raise

    def _ck(self, err, what):
        if err != 0:
            raise RuntimeError(f"GPU {self.index}: CUDA error {err} in {what}")

    def release(self):
        try:
            if self.ptr.value:
                self.lib.cuMemFree_v2(self.ptr)
                self.ptr = ctypes.c_void_p()
        except Exception:
            pass
        try:
            if self.ctx.value:
                self.lib.cuCtxDestroy_v2(self.ctx)
                self.ctx = ctypes.c_void_p()
        except Exception:
            pass


class Claim:
    def __init__(self, index, lock, hold):
        self.index, self.lock, self.hold = index, lock, hold

    def release(self):
        if self.hold is not None:
            self.hold.release()
        self.lock.release()


def try_claim(index, lock_dir, frac):
    """Lock, re-check freedom, then hold.

    Returns (Claim, reason). Exactly one of the two is None. The caller needs
    the reason, because the reasons need different responses:

      "locked"   another copy of this script owns the card -- leave it alone
      "busy"     the card became busy between the scan and the lock -- the
                 stability streak is genuinely broken, so reset it
      "cuda"     the card IS free but the hold failed -- the streak must NOT
                 reset, or a driver hiccup silently costs another full wait
                 at the exact moment a card finally became available
    """
    lock = GPULock(lock_dir, index)
    if not lock.acquire():
        return None, "locked"             # another copy of this script has it
    # Re-check AFTER locking: closes the window between "listed free" and "locked".
    try:
        busy = busy_uuids()
        row = [r for r in gpu_table() if r[0] == index]
        if not row or row[0][1] in busy or row[0][2] > FREE_MEM_MIB:
            lock.release()
            return None, "busy"
    except Exception as e:
        log(f"GPU {index}: re-check failed ({e}); not claiming")
        lock.release()
        return None, "busy"
    try:
        hold = CudaHold(index, frac)
    except Exception as e:
        log(f"GPU {index}: CUDA HOLD FAILED ({e}) -- card is free but could not "
            f"be held; will retry next poll")
        lock.release()
        return None, "cuda"
    return Claim(index, lock, hold), ""


# ---------- main loop ----------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--num-gpus", type=int, default=2)
    ap.add_argument("--poll-interval", type=float, default=30.0)
    ap.add_argument("--stable-seconds", type=float, default=300.0,
                    help="a card must look free this long before we claim it")
    ap.add_argument("--hold-frac", type=float, default=0.90,
                    help="fraction of free memory to hold (1.0 = reserve all)")
    ap.add_argument("--lock-dir", default=LOCK_DIR)
    ap.add_argument("--state-dir", required=True,
                    help="directory for status.json and READY")
    args = ap.parse_args()

    os.makedirs(args.state_dir, exist_ok=True)
    status_path = os.path.join(args.state_dir, "status.json")
    ready_path = os.path.join(args.state_dir, "READY")

    need = max(1, int(round(args.stable_seconds / args.poll_interval)))
    claims, streak, stop, announced = {}, {}, {"flag": False}, False
    hold_fails = {}                       # gpu index -> consecutive CUDA failures

    def on_signal(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    def release_all():
        for c in claims.values():
            c.release()
        claims.clear()
        try:
            os.remove(ready_path)
        except OSError:
            pass

    def write_status(state, note=""):
        payload = {
            "pid": os.getpid(),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "state": state,
            "note": note,
            "target_gpus": args.num_gpus,
            "held_gpus": sorted(claims),
            "held_mib": {str(i): c.hold.held_bytes // MIB
                         for i, c in claims.items() if c.hold},
            "stable_seconds_required": args.stable_seconds,
            "free_streak_polls": {str(k): v for k, v in sorted(streak.items()) if v},
            "polls_needed": need,
            "cuda_hold_failures": {str(k): v for k, v in sorted(hold_fails.items())},
        }
        tmp = status_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, status_path)      # atomic: readers never see a half file

    log(f"Reserving {args.num_gpus} GPU(s). A card must be free for "
        f"{args.stable_seconds:.0f}s ({need} polls) before it is claimed. "
        f"Hold fraction {args.hold_frac}. Cards in use are never touched.")

    try:
        while not stop["flag"]:
            if len(claims) < args.num_gpus:
                try:
                    table, busy = gpu_table(), busy_uuids()
                except Exception as e:
                    log(f"nvidia-smi failed ({e}); retrying")
                    write_status("smi-error", str(e))
                    table = None
                if table is not None:
                    for idx, uuid, used in table:
                        if idx in claims or len(claims) >= args.num_gpus:
                            continue
                        free = uuid not in busy and used <= FREE_MEM_MIB
                        if not free:
                            if streak.get(idx):
                                log(f"GPU {idx} became busy again; streak reset")
                            streak[idx] = 0
                            continue
                        streak[idx] = streak.get(idx, 0) + 1
                        if streak[idx] == 1:
                            log(f"GPU {idx} looks free; needs {need} consecutive "
                                f"checks before a claim")
                        if streak[idx] < need:
                            continue
                        claim, why = try_claim(idx, args.lock_dir, args.hold_frac)
                        if claim is None:
                            # Only a genuinely busy card breaks the streak. A
                            # failed hold retries next poll instead of costing
                            # another full stability wait.
                            if why == "busy":
                                streak[idx] = 0
                            elif why == "cuda":
                                hold_fails[idx] = hold_fails.get(idx, 0) + 1
                            continue
                        hold_fails.pop(idx, None)
                        claims[idx] = claim
                        log(f"CLAIMED GPU {idx} "
                            f"({claim.hold.held_bytes // MIB} MiB held) -- "
                            f"{len(claims)}/{args.num_gpus}")

            done = len(claims) >= args.num_gpus
            write_status("holding" if done else "waiting")
            if done and not announced:
                announced = True
                devs = ",".join(str(i) for i in sorted(claims))
                with open(ready_path, "w") as f:
                    f.write(devs + "\n")
                log(f"TARGET REACHED. Holding GPUs [{devs}]. "
                    f"Wrote {ready_path}. Holding until SIGTERM.")

            slept = 0.0
            while slept < args.poll_interval and not stop["flag"]:
                time.sleep(min(1.0, args.poll_interval - slept))
                slept += 1.0

        log("Signal received; releasing every card this process holds.")
        return 0
    finally:
        release_all()
        try:
            write_status("stopped")
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
