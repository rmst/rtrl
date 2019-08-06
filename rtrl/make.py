import sys

# print(sys.argv)
_, path, cmds = sys.argv

from rtrl import *

agent = eval(cmds)

print(agent)

# save to path