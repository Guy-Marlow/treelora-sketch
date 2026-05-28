"""Run all DCMS tests and print a summary."""
import subprocess
import sys
import os

here = os.path.dirname(os.path.abspath(__file__))
suites = ['test_dyadic_cover', 'test_insert']

results = {}
for suite in suites:
    print(f'\n{"="*65}')
    print(f'  {suite}')
    print('='*65)
    r = subprocess.run(
        [sys.executable, '-m', 'unittest', suite, '-v'],
        cwd=here
    )
    results[suite] = r.returncode

print(f'\n{"="*65}')
print('SUMMARY')
print('='*65)
all_ok = True
for suite, rc in results.items():
    status = 'PASS' if rc == 0 else 'FAIL'
    if rc != 0:
        all_ok = False
    print(f'  {status}  {suite}')
print('='*65)
sys.exit(0 if all_ok else 1)
