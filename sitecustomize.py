# sitecustomize.py
import coverage
import os

if os.getenv("COVERAGE_PROCESS_START"):
    coverage.process_startup()
