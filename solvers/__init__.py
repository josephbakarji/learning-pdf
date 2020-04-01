import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# FIX THIS! - import init from parent directory
MAIN = parent_dir 
DATAFILE = parent_dir+"/datafile/"
FIGFILE = parent_dir+"/figures/"

MCDIR = current_dir + '/MCresults/' 
