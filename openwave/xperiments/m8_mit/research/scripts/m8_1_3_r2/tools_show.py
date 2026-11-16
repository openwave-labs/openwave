"""print a log without its PASS lines: ./py tools_show.py name.log"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(HERE, sys.argv[1])):
    if not line.startswith("PASS"):
        print(line, end="")
