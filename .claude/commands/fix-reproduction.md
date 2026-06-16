# fix-reproduction

Automatically diagnose and fix runtime errors in reproduction code by reading training logs,
locating the failing code, cross-referencing the original paper's design, and applying
minimal targeted fixes. Loops up to max-attempts times until the code runs cleanly.

## Arguments

$ARGUMENTS

- First argument (required): paper name slug (e.g. `attention-is-all-you-need`)
- Second argument (optional): run name (e.g. `run_20260615_143022`). Defaults to the most recent run.
- Third argument (optional): max attempts, default `5`

## Steps

---

**Step 1 — Locate the target run.**

Find the log directory:
```bash
ls -t logs/{name}/ 2>/dev/null | head -5
```

If a run name was provided, use `logs/{name}/{run_name}/`. Otherwise use the most recent directory
(first result of `ls -t`). Store as `{LOG_DIR}`.

Check that `{LOG_DIR}/train.log` exists. If the directory or log file is missing, run the code
first to generate an error:
```bash
cd outputs/{name}/reproduction && timeout 60 python train.py --run-name debug_fix_0 2>&1 | tail -50
```
Then set `{LOG_DIR}` to `logs/{name}/debug_fix_0/`.

---

**Step 2 — Extract the error.**

Run:
```bash
python3 scripts/parse_errors.py {LOG_DIR}/train.log
```

Parse the JSON output. If `found` is `false`, report:
> "No error found in `{LOG_DIR}/train.log`. The last run may have completed without errors.
>  To verify, run: `cd outputs/{name}/reproduction && python test.py`"
Then stop.

From the parsed output, record:
- `ERROR_TYPE` — e.g. `RuntimeError`
- `ERROR_MSG` — full error line
- `PRIMARY_FILE` — `primary_frame.file`
- `PRIMARY_LINE` — `primary_frame.line`
- `PRIMARY_CODE` — the code snippet at the failing line
- `RAW_TB` — `raw_traceback` (for context)

---

**Step 3 — Load context for diagnosis.**

Read the relevant files to understand BOTH what the code is doing and what it should do:

1. **Read the failing file**: Read `{PRIMARY_FILE}` (the full file, or at minimum lines ±50 around `{PRIMARY_LINE}`)

2. **Read the paper's design**: Determine which section of innovations.md is most relevant:
   - If `PRIMARY_FILE` contains `model` → read the section titled "Model Architecture" in `analyses/{name}/innovations.md`
   - If `PRIMARY_FILE` contains `loss` → read the section titled "Loss Design"
   - If `PRIMARY_FILE` contains `train` → read the section titled "Training Strategy"
   - If `PRIMARY_FILE` contains `dataset` → read "Training Strategy" + "Implementation Notes"
   - Otherwise → read full `analyses/{name}/innovations.md`
   - **If the targeted section is absent** (paper may not have a custom loss, etc.) → read the full `analyses/{name}/innovations.md` instead

3. **Read the full traceback context**: If the primary frame is in torch internals (no user frame),
   also read the user frame just before the torch frame to understand what triggered it.

---

**Step 4 — Diagnose and apply a targeted fix.**

Classify the error and apply the appropriate fix strategy:

### Error type → Fix strategy

**`ImportError` / `ModuleNotFoundError`**
- If missing a standard package: run `pip install {package}` then re-verify (don't modify code)
- If missing a local module: check the `sys.path.insert` line at the top of the file — ensure it points to the project root two directories up

**`RuntimeError: Expected X but got Y` (shape mismatch)**
- Trace the tensor that has the wrong shape back through the call chain
- Compare the actual shape to what `innovations.md` Section 3.2 specifies for that component
- Fix the dimension that diverged (usually a wrong `nn.Linear` in/out size, or a missing reshape)
- Do NOT change the model's mathematical operation — only fix the dimension

**`TypeError` (wrong argument type or count)**
- Check the PyTorch API for the correct signature
- Fix the call site; do not restructure the surrounding logic

**`AttributeError`**
- A method or attribute name is wrong. Check the PyTorch version compatibility
- Fix the name; if an API was renamed between PyTorch versions, use the current name

**`CUDA / device mismatch`**
- Find the tensor that is on the wrong device
- Add `.to(device)` at the point where it enters the model or is combined with another tensor

**`KeyError` / `IndexError` in dataset**
- Check what keys/shapes the DataLoader yields and what the model expects
- Fix either the dataset `__getitem__` or the unpacking in `train_step`

**`ValueError` (wrong value range or format)**
- Fix the preprocessing step that produces the invalid value

### Fix rules (must follow all)

1. **Never alter the algorithm**: do not change forward pass logic, loss equations, or mathematical operations that implement the paper's method
2. **Minimal change only**: fix only the specific line(s) causing the error; no opportunistic cleanup
3. **Paper-consistent**: if unsure between two fixes, choose the one consistent with `innovations.md`
4. **No new abstractions**: do not introduce new helper classes or functions unless truly required

Apply the fix using the Edit tool.

After fixing, briefly note what you changed and why in one sentence.

---

**Step 5 — Verify the fix.**

Run the code for a short time to check that the error is gone:
```bash
cd outputs/{name}/reproduction && timeout 90 python train.py --run-name debug_fix_{attempt} 2>&1 | head -120
```

Parse the output:
- **No error in output**: fix is successful → go to Step 7
- **Same error at same line**: the fix was wrong or incomplete → go to Step 6 (retry)
- **New error at a different line**: this fix worked but exposed another bug → treat as a new error, go to Step 2 with the new error and increment `{attempt}`
- **Timeout with no error**: training ran past 90 s without crashing — fix is successful → go to Step 7

---

**Step 6 — Track attempts and decide whether to retry.**

Keep a mental (or written) record:
```
Attempt {N}:
  error_type: {ERROR_TYPE}
  file: {PRIMARY_FILE}:{PRIMARY_LINE}
  fix: {one-line description of what was changed}
  result: fixed | same-error | new-error | timeout-ok
```

**Retry conditions**:
- `{attempt}` < max-attempts (default 5)
- The same error type has not appeared more than 2 times at the same line
  (repeating at same location means the root cause is deeper — stop and report)

If retry: go back to Step 2 with the updated log from `debug_fix_{attempt}`.
If stop: go to Step 7 with status "failed".

---

**Step 7 — Final report.**

**On success**:
```
✓ Fixed: {name}
  Attempts: {N}
  Error resolved: {ERROR_TYPE} in {PRIMARY_FILE}:{PRIMARY_LINE}
  Fix summary:
    Attempt 1: {description}
    ...
  Verified: training ran {K} steps without error in debug_fix_{attempt}/

  Next steps:
    cd outputs/{name}/reproduction
    python train.py --run-name exp_01
```

**On failure** (max attempts reached):
```
✗ Auto-fix failed after {N} attempts: {name}

  Error history:
    Attempt 1: {ERROR_TYPE} @ {file}:{line} — tried: {fix}
    Attempt 2: ...

  Recommended manual investigation:
    1. Read {PRIMARY_FILE} around line {PRIMARY_LINE}
    2. Cross-check analyses/{name}/innovations.md Section {N}
    3. Check tensor shapes with debug prints:
       print(f"shape: {tensor.shape}")  # add before the failing line

  Last error:
    {RAW_TB}
```

**If the same error recurs at the same location 2+ times**:
Report that the fix location may be wrong — the root cause is likely upstream:
```
✗ Root cause not at {file}:{line} — error persists after {N} attempts.
  The actual source of the problem is likely in an earlier part of the data pipeline.
  Suggestion: add shape debug prints starting from the model's first layer.
```
