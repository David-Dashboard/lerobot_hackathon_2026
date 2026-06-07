# Hackathon Retrospective & Working-Style Profile
*Prepared for the user's personal AI assistant. Source: the SO-101 / LeRobot hackathon, 6 Claude Code sessions across 2026-06-06 → 2026-06-07.*

---

## 0. How to use this document
You (the assistant) are receiving this so you can (a) understand how your principal works, (b) anticipate the failure modes that cost them this hackathon, and (c) actively help them avoid those next time. Sections 3–6 are the behavioural analysis; Section 7 is what *you* should do about it; Sections 8–9 are the concrete unfinished work and the technical landmines, so neither of you re-hits them.

**Confidence note:** I had full access to one session (the latest) and the metadata (titles/timestamps) + final repo state of the other five. Transcript search across the older sessions was blocked, so claims about those are *inferred from titles + artifacts*, not read line-by-line. Treat specifics as "very likely" rather than "verified," and let the user correct.

---

## 1. Project context (the facts)
- **Goal:** Build a tabletop trash-collecting system on an SO-101 robot arm using LeRobot — and, as the headline ML deliverable, **train a learned policy (ACT / a VLA) to grasp**.
- **Span:** ~6 sessions over 2 days. Arc visible from session titles: `LeRobot+Rerun setup` → `auto-record` (×2, incl. a fork) → `invariant to camera pose` → `project status`.
- **Repo:** `C:\Users\Succe\lerobot-hackathon` (git: `main` + `coord-grasp` branch).
- **What got built and works:**
  - A clean, decoupled `so101` robot wrapper (mock + real), teleoperation **leader** wrapper, and a **teleop dataset recorder** (`record_teleop.py`) with manual ENTER-controlled episodes.
  - A full **scripted Phase-1 pipeline** (`trash_arm/`): config, perception (OWL-ViT), homography localization, planar-IK motion + `pick_and_place`, safety limits, orchestrator. ~95 unit tests, all green.
  - A verified **recording** (`recorded/trash_teleop`: **2 episodes / 900 frames** with real scene-camera images + state + action).
  - A separate **`coord-grasp` branch**: a self-calibrating, *coordinate-only* (pixel-free) grasp experiment — ArUco camera-pose estimation, forward kinematics, hand-eye, coordinate-only recorder — plus the start of **OAK-D-PRO depth-camera** integration.
- **What did NOT happen:** **No ACT / policy was ever trained.** This was the stated hero goal.

---

## 2. The headline outcome (said plainly)
The user built a *lot* of solid, well-tested infrastructure and learned a great deal — but **missed the one deliverable that defined success** (a trained grasp policy). This was **not a capability failure**; it was a **sequencing + environment-friction failure**. The pieces needed to train were nearly all there; the time to actually do it got eaten by tooling, hardware, and exploration.

---

## 3. Working style — genuine strengths
1. **Strong conceptual curiosity / systems thinking.** Asked excellent deep questions: camera-pose invariance, a comparison of GR00T N1.5 vs π0.5 vs X-VLA, and how to build a *vision-free, goal-conditioned* grasp policy. These are the right questions, asked well.
2. **Good architectural instincts.** Wanted modular, separable stages; insisted on a *separate branch* for the experimental idea so it wouldn't tangle with the main pipeline; cared about version isolation.
3. **Effective delegation.** Used the agent decisively ("build everything", "yes go") and let it carry heavy lifting while staying in the loop on decisions.
4. **Hands-on and persistent with hardware.** Worked through power, cabling, and calibration problems without giving up.
5. **Quality-conscious.** Didn't cut corners on tests or structure even under time pressure.
6. **Self-aware about energy.** Said "I am tired," asked for simplified step-by-step plans — good metacognition, if acted on.

---

## 4. Working-style tendencies that cost time *(with evidence)*
1. **Frequent context-switching / pivots.** Jumped between software building, hardware bring-up, conceptual research, and new branches — often *before the current thread reached a usable endpoint*. (E.g., mid-build of the pipeline → "Wait" → pivot to hardware; later, deep foundation-model Q&A in the middle of execution.) Each switch carries reload cost.
2. **Scope *expansion* under time pressure.** Added the `coord-grasp` self-calibration experiment **and** OAK-D depth-camera integration — both genuinely interesting — *before the MVP existed*. Classic shiny-object pull during a timed event.
3. **The critical path was never protected.** "Collect N demos → train ACT" kept sliding behind infrastructure and exploration. Nothing was explicitly designated "this must go green first, everything else waits."
4. **Training data was under-collected.** The verified dataset was **2 episodes / 900 frames**; ACT typically needs **~30–50+ demonstrations**. Recording was treated as "prove it works," not "collect the real dataset." Even a flawless pipeline couldn't have trained a useful policy from that.
5. **Environment was not pre-hardened.** A large fraction of wall-clock went to *preventable* Windows yak-shaving (see §9). None of it was deep — all of it was time.

---

## 5. The bottlenecks, by root cause
| # | Bottleneck | Root cause | Preventable? |
|---|---|---|---|
| 1 | Windows OS friction: camera-access privacy toggle off; cv2 backend quirks; `rerun.exe` not on PATH; PowerShell execution policy blocked venv activation; uv-venv has no `pip`. | Environment not validated end-to-end before the event. | **Yes** — pre-event smoke test. |
| 2 | Dependency conflict: installing `transformers` pulled v5 + `huggingface_hub` 1.x, **breaking the LeRobot import** that teleop relies on. | Unpinned install mid-event. | **Yes** — pin to LeRobot's constraints upfront. |
| 3 | Hardware fragility: one arm's barrel-jack power not connected; `wrist_flex` (motor 4) daisy-chain cable loose → calibration crash; arm calibration "a bit off." | Physical rig not fully seated/validated before software. | **Mostly** — pre-flight hardware check. |
| 4 | Sequencing: built the entire scripted pipeline + a second experimental branch — valuable, but *off the critical path* to "train an ACT." | No single, explicitly-protected MVP goal with a timebox. | **Yes** — pick one hero goal. |
| 5 | Data volume: 2 episodes insufficient to train. | Recording treated as a test, not a collection push. | **Yes** — set a demo-count target. |

**Pattern:** none of these were hard problems. They were *many small, mostly-preventable frictions* plus *scope that grew instead of shrank as the clock ran down*. That combination is what blocked the hero deliverable.

---

## 6. Feedback — what to do differently next time
1. **Name ONE hero goal and protect it.** e.g. *"A trained ACT that attempts the pick, even badly, evaluated on the real arm."* Write it where it's always visible. Everything else is optional until it's green.
2. **Run a pre-event readiness pass (the day before).** Full environment smoke: cameras open in cv2, Rerun viewer launches, `import lerobot` works, a **2-step `lerobot-train` runs on a dummy dataset**, requirements pinned. Arms: power + calibrate + teleop verified; ports pinned by serial. *Catch the Windows gotchas then, not during.*
3. **Front-load data collection.** The moment recording works, collect the **full** dataset (30–50 demos) in one focused block. "It records" ≠ "I have a dataset."
4. **Timebox exploration.** Park conceptual rabbit holes and new-architecture ideas into a "later" list during execution windows. Foundation-model comparisons and self-calibration branches are great — *after* the hero goal, or in the final slack hour.
5. **Shrink scope as the clock runs down, never grow it.** Resist adding hardware (OAK-D) or new policy paradigms mid-event.
6. **Keep a visible two-column board:** `Critical path` vs `Side quests`. Anything not on the left waits.

The encouraging part: the infrastructure built is real and reusable. The user is **one focused 3-hour block** (collect demos → train → eval) away from the result they wanted. This was a near-miss, not a wipeout.

---

## 7. How you (the assistant) should help — concrete behaviours
- **At the start of any timed build, force the hero-goal question:** "What's the single deliverable that must be green by the deadline?" Pin it, then periodically ask *"are we still on the critical path to it?"*
- **When they propose a new branch/idea mid-execution, surface the trade-off, gently:** "Love it — but the hero goal (X) isn't green yet; park this on the side-quest list?" Don't block curiosity; *time-shift* it.
- **Watch for the 'it works → move on' reflex on data tasks.** Push a concrete volume target ("we have 2 demos; ACT wants ~40 — want to do a 30-min collection block now?").
- **Maintain two running lists:** a blockers log, and a "preventable next time" list that becomes the pre-event checklist.
- **Be the pre-event checklist runner.** Before the next event, proactively walk the §9 checklist with them.
- **Pace for energy.** They flag fatigue and ask for step-by-step. When tired, give them *the single next physical action*, not a plan with 9 branches.
- **Default to finishing over starting.** When in doubt, nudge toward closing the current thread to a usable state before opening a new one.

---

## 8. Unfinished critical path — how to actually get an ACT trained
1. **Lock the hardware.** Both arms powered; recalibrate cleanly (ensure the `wrist_flex` cable is seated); verify teleop drives the follower.
2. **Collect a real dataset.** 30–50 manual-mode demos of the pick task with the scene camera: `python record_teleop.py --robot-port COM5 --teleop-port COM4 --camera scene=0:640x480 --no-calibrate --manual --episodes 50 ...`. Use consistent start poses + varied object placement.
3. **Pin the env, then train.** Locally for a smoke (`lerobot-train --policy.type=act --policy.device=cpu --num_workers=0 ...`), or push the dataset to the HF Hub and run a **Qualia smolvla** finetune (token already configured; ~400 credits were available).
4. **Evaluate on the real arm** via the deploy path; record a success rate.
5. **Only then** expand into the camera-agnostic / depth / coordinate-only direction (the `coord-grasp` branch is already scaffolded for it).

---

## 9. Technical gotchas reference *(pre-event checklist — verify each is green)*
- **Windows camera:** Settings → Privacy → Camera → **"Camera access" ON** (global toggle; off = cv2 can open nothing). Then cv2 default backend opens the external cam at its index.
- **Rerun viewer:** `rr.spawn()` needs `rerun.exe` on PATH → either `.\.venv\Scripts\Activate.ps1` **or** `$env:Path = "$PWD\.venv\Scripts;$env:Path"` before running.
- **PowerShell execution policy** may block `Activate.ps1`: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` (one-time).
- **venv is `uv`-managed → no `pip`.** Install with `uv pip install ...`, never `python -m pip`.
- **Pin ML deps for LeRobot 0.4.4:** `transformers>=4.57,<5` and `huggingface-hub<0.36`. (transformers 5 / hub 1.x **breaks** the LeRobot import.) Don't run an unpinned `install` mid-event.
- **SO-101 bring-up:** each arm needs **USB *and* barrel-jack power**. `Incorrect status packet` on connect ≈ power/cable. A per-motor `no status packet` *during calibration* = a loose daisy-chain cable at that motor (reseat it).
- **Pin serial ports by board serial** (`USB\VID_1A86&PID_55D3\<serial>`), not the COM number — COM numbers drift on replug. (Follower = `5B41531706`, Leader = `5B41531811`.)
- **lerobot-train on Windows/CPU:** `--num_workers=0` (Windows multiprocessing) and `--policy.device=cpu` (torch here is CPU-only); `--wandb.enable=false`.
- **Beware the `| tail` exit-code trap:** piping an install to `tail` masks the real exit code — a failed install can look like success.

---

*Bottom line for the assistant: your principal is a strong, curious builder who lost the hackathon's hero deliverable to preventable friction and unprotected scope, not to lack of skill. Your highest-leverage job is to (1) run the pre-event readiness pass, (2) protect the single hero goal, and (3) keep "it works" from masquerading as "it's done" — especially on data collection and training.*
