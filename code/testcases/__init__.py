import os,sys,inspect

TESTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
MAIN = os.path.dirname(TESTDIR)
sys.path.insert(0, MAIN) 


FIGDIR = MAIN + "/figures/"
SOLVERDIR = MAIN + "/solvers/"
PDFDIR = MAIN + "/pdf_data/"
LEARNDIR = MAIN + "/learn_data/"
MCDIR = MAIN + "/mc_data/"

sys.path.append(os.path.abspath(SOLVERDIR))

