# SDD ledger — plan: docs/superpowers/plans/2026-09-23-ppe-expanded-candidates.md
Setup Ruling: repository root is not Git and D-FINE/master has pre-existing dirty work — execute in place as explicitly selected; do not create commits or touch unrelated changes — cost if wrong: rollback depends on scoped file edits and recorded production hashes.
Baseline: 61 pytest tests and 10 subtests passed; ppe-5c CURRENT=6a4dd4f1292433d7; audit/config hashes recorded in session.
Pre-flight Task 1 -> Task 2: candidate CLI consumes taxonomy keyword arguments from the parameterized builder; interface names match.
Pre-flight Task 2 -> Task 3: candidate CLI consumes pinned registry archives and publishes content-addressed builds; cache filenames match source ID plus hash prefix.
Pre-flight Task 3 -> Task 4: D-FINE configs consume candidate `current/annotations` paths; publication convention matches production.
Pre-flight Ruling: candidate builds must not mutate ppe-5c and must not use source C — enforce by exact source selection and before/after hashes — cost if wrong: production reproducibility or license isolation breaks.
Pre-flight Ruling: plan commit steps are skipped because no safe repository boundary covers all plan files — cost if wrong: no atomic Git rollback for this run.
Task 1: complete (tests: tests.test_ppe_dataset → 17 passed; custom taxonomy and duplicate grouping verified).
Task 2: complete (tests: build CLI + registry → 9 passed; exact source isolation, taxonomy order, pinned hashes and failure atomicity verified).
Task 3: complete (ppe-13c=f9f77e2427b162b2, pants-2c=f84670822dfee0cf; zero cross-split duplicates, all classes present in every split, CC-BY-4.0 only, ppe-5c hashes unchanged).
Task 4: complete (tests: full pytest → 65 passed, 14 subtests; configs resolve 13/2/5 classes; both candidate verifiers pass; ppe-5c hashes unchanged).
Final review: self-review (subagents prohibited by session policy); one Important finding fixed.
Final: fixed candidate output could overwrite another pipeline — test_candidate_rejects_output_for_another_pipeline RED→GREEN, suite 66 passed / 14 subtests.
Final: no deferred minors.
Finish Ruling: retain the plan ledger because no Git history covers the root-level artifacts — cost if wrong: one small scratch directory remains under D-FINE.
