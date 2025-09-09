# Entry point for `python -m felix`
import runpy
if __name__ == "__main__":
    runpy.run_module("felix.quantum_cat", run_name="__main__")
