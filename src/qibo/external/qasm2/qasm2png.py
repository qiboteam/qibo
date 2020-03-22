import os
from .qasm2tex import *

def qasm2png(base):
    qasm2tex(base)
    path2tex = os.path.dirname(os.path.abspath(__file__))
    path2texfile = path2tex+"/Circuit"
    os.system('mv %(base)s.tex %(path2texfile)s.tex; cd %(path2tex)s; latex %(path2texfile)s.tex' % locals())
    os.system('cd %(path2tex)s; dvips -D2400 -E -o %(path2texfile)s.eps %(path2texfile)s.dvi' % locals())
    os.system('gs -sDEVICE=pnmraw -r400 -dNOPAUSE -sOutputFile=%(path2texfile)s.pbm %(path2texfile)s.eps -c quit' % locals())
    os.system('pnmcrop %(path2texfile)s.pbm | pnmtopng > %(base)s.png' % locals())



