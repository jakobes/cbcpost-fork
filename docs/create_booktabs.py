import os
import re
#from IPython import embed

with open("_build/cbcpost.tex", 'r') as f:
    r = f.read()

#r = r.replace("tabulary","tabularx")
with open("_build/cbcpost.tex", 'w') as f:
    while True:
        try:
            i = r.index(r"\begin{tabulary}")
        except:
            f.write(r)
            break
        try:
            j = r.index(r"\end{tabulary}")+14
        except:
            f.write(r)
            break
        assert j > 0
        print i, j
        f.write(r[:i])
        table = r[i:j]
        
        new_table = table
        count = table.count("\hline")
        new_table = table
        new_table = new_table.replace("\\hline", "\\toprule", 1)
        new_table = new_table.replace("\\hline", "\\midrule", 1)
        new_table = new_table.replace("\\hline", "", count-3)
        new_table = new_table.replace("\\hline", "\\bottomrule\n", 1)
        f.write(new_table)
        
        r = r[j:]
        
import time
time.sleep(1)
