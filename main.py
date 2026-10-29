"""Entry point used by the organizers' verification setup: python main.py INPUT_FOLDER OUTPUT_FOLDER"""
import os
import runpy
import sys

if __name__ == "__main__":
    sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference", "infer.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
