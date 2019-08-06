import sys

# print(sys.argv)
_, args = sys.argv

from rtrl import *


args = eval("dict(" + args + ")")

print(args)