"""Paired comparison of two policies on the same test layouts.
usage: python scripts/compare.py <labelA[,labelA_seed2]> <labelB[,labelB_seed2]>"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from phone2robot.stats import paired_diff

r = paired_diff(sys.argv[1].split(","), sys.argv[2].split(","))
print(f"A {100 * r['a']:.1f}%  B {100 * r['b']:.1f}%  A-B {100 * r['diff']:+.1f} pts  "
      f"95% CI [{100 * r['ci'][0]:+.1f}, {100 * r['ci'][1]:+.1f}]  "
      f"({r['layouts']} layouts: A better {r['a_better']}, B better {r['b_better']}, tie {r['ties']})")
