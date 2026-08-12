# GPU reservation monitor

This monitor waits for free GPUs on the shared lab box. It claims 2 cards when
they become free, and it holds them until you release them. Then you can start a
training run or an experiment at any time.

Written in Simplified Technical English.

## The safety rule

**The monitor never touches a card that another person uses.** It claims a card
only when `nvidia-smi` shows zero compute processes on that card and less than
1024 MiB of memory in use. It never sends a signal to another process.

## The stability gate

Another user on this box restarts vLLM servers frequently. During a restart the
cards of that user look free for 1 to 3 minutes. A claim in that window takes a
card from a job that only reloads weights.

Thus the monitor adds a **stability gate**. A card must look free continuously
for 300 seconds (10 polls of 30 seconds) before the monitor claims it. A server
restart does not satisfy the gate.

This is the rule in `CLAUDE.md`: wait for a stable free window, never race.

## The files

| File | Function |
|---|---|
| `gpu_reserve.py` | Finds free cards, applies the stability gate, claims and holds. Writes `status.json` each poll. |
| `gpu_reserve_supervisor.sh` | Keeps `gpu_reserve.py` alive. Restarts it after a crash. Sends SIGTERM to it at shutdown, so the cards are released. One copy only, controlled by a lock file. |
| `gpu_reserve_ctl.sh` | The control command: `start`, `status`, `log`, `stop`, `handoff`. |

`gpu_reserve.py` is self-contained. It does not import
`/home/liangsheng/brian/acquire_gpus.py`, because that file is outside the
repository and a container wipe deletes it.

## The commands

```bash
cd research/foundation/scripts

./gpu_reserve_ctl.sh start     # start the monitor, detached from the terminal
./gpu_reserve_ctl.sh status    # show the state and the live nvidia-smi output
./gpu_reserve_ctl.sh log 40    # show the last 40 log lines
./gpu_reserve_ctl.sh stop      # release the cards and stop the monitor
./gpu_reserve_ctl.sh handoff   # release the cards, then print CUDA_VISIBLE_DEVICES
```

## How to start a training run

The monitor holds 90 percent of the memory of each card. Thus the monitor must
release the cards before your training job can allocate memory. Use `handoff`:

```bash
eval $(./gpu_reserve_ctl.sh handoff)   # sets CUDA_VISIBLE_DEVICES
<your training command>
./gpu_reserve_ctl.sh start             # start the monitor again after the run
```

`handoff` prints the device list of the cards that the monitor held. Thus the
next command uses the same cards, and the gap is only a few seconds.

## The state directory

`/mnt/src/liangsheng/cassi_foundation/gpu_reserve/`

This directory is on persistent storage, so a container wipe does not delete it.

| File | Content |
|---|---|
| `status.json` | The state, written each poll. Written atomically, so a reader never sees a partial file. |
| `reserve.log` | The full log of the monitor and the supervisor. |
| `READY` | Exists only when the monitor holds all 2 cards. Contains the device list, for example `3,5`. |
| `supervisor.pid` | The process ID of the supervisor. |
| `STOP` | Exists only after a stop request. The supervisor exits when it sees this file. |

## The settings

The defaults are in `gpu_reserve_supervisor.sh`. Change them with environment
variables:

| Variable | Default | Function |
|---|---|---|
| `GPU_RESERVE_NUM` | `2` | The number of cards to reserve |
| `GPU_RESERVE_STABLE` | `300` | The stability gate, in seconds |
| `GPU_RESERVE_POLL` | `30` | The interval between checks, in seconds |
| `GPU_RESERVE_HOLD_FRAC` | `0.90` | The fraction of free memory to hold |

### A note on `GPU_RESERVE_HOLD_FRAC`

A hold of 32 MiB is visible in `nvidia-smi`, but it does not stop another job
from allocating the remaining memory on the same card. Only a large hold
reserves the card fully.

The default of `0.90` therefore reserves the card. This is not a risk to other
people, because the monitor claims only cards that are already free. But it does
keep a free card unused while you are away. Set the value to `0.01` if you prefer
a light hold that only shows your name on the card.
