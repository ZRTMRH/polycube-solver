"""Simulate the professor's grading: build_scale_suite + solve_size_gated.

This is the most honest test — it uses the exact same case generators
and normalization that the professor will use.
"""
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grading_harness import build_scale_suite, verify_solution
from hybrid_solver import solve_size_gated

CASE_TIMEOUT = 120


def main():
    total = 0
    correct = 0
    failures = []
    times_by_grid = {}

    # Test each grid size the professor might use
    for N in [3, 4, 5, 7, 9, 12]:
        n_cases = 5 if N <= 5 else 3
        for suite_type in (["mixed"] if N >= 7 else [None]):
            print(f"\n{'='*60}")
            kwargs = {}
            if suite_type:
                kwargs["large_suite_type"] = suite_type
            try:
                cases, gen_type = build_scale_suite(N, n_cases=n_cases, seed=561, **kwargs)
            except Exception as e:
                print(f"N={N} ({suite_type}): GEN ERROR ({e})")
                continue
            print(f"N={N}: {len(cases)} cases from {gen_type}")
            print(f"{'='*60}")
            sys.stdout.flush()

            for case in cases:
                total += 1
                t0 = time.time()
                try:
                    res = solve_size_gated(case.pieces, case.grid_size)
                    elapsed = time.time() - t0
                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"  {case.case_id}: ERROR ({e}) {elapsed:.1f}s")
                    failures.append(case.case_id)
                    continue

                got_solution = res['solution'] is not None
                expected = case.expected_solvable

                if expected and got_solution:
                    valid = verify_solution(res['solution'], case.grid_size)
                    if valid:
                        correct += 1
                        status = "PASS"
                    else:
                        status = "INVALID"
                        failures.append(case.case_id)
                elif not expected and not got_solution:
                    correct += 1
                    status = "PASS"
                else:
                    status = "FAIL"
                    failures.append(case.case_id)

                method = res.get('submethod', '?')
                print(f"  {case.case_id}: {status} {elapsed:.1f}s method={method}")
                times_by_grid.setdefault(N, []).append(elapsed)
                sys.stdout.flush()

    # Also test striped at large grids
    for N in [7, 9, 12]:
        try:
            cases, gen_type = build_scale_suite(N, n_cases=3, seed=561, large_suite_type="striped")
        except Exception as e:
            print(f"N={N} striped: GEN ERROR ({e})")
            continue
        print(f"\nN={N} striped: {len(cases)} cases from {gen_type}")
        for case in cases:
            total += 1
            t0 = time.time()
            res = solve_size_gated(case.pieces, case.grid_size)
            elapsed = time.time() - t0
            got_solution = res['solution'] is not None
            if got_solution and verify_solution(res['solution'], case.grid_size):
                correct += 1
                print(f"  {case.case_id}: PASS {elapsed:.1f}s method={res.get('submethod','?')}")
            else:
                failures.append(case.case_id)
                print(f"  {case.case_id}: FAIL {elapsed:.1f}s")
            sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"PROFESSOR GRADING SIMULATION: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'='*60}")
    if failures:
        print(f"Failures: {failures}")
    for N, times in sorted(times_by_grid.items()):
        avg = sum(times) / len(times)
        mx = max(times)
        print(f"  N={N}: avg={avg:.2f}s max={mx:.2f}s")


if __name__ == "__main__":
    main()
