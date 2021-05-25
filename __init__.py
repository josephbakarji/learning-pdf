import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# TODO: unify all __init__ files, and use os.path.join to properly define paths
MAIN = current_dir
FIGDIR = current_dir + "/figures/"
SOLVERDIR = current_dir + "/solvers/"
LEARNDIR = current_dir + "/learn_data/"
MCDIR = current_dir + "/mc_data/"
PDFDIR = current_dir + "/pdf_data/" # Change name