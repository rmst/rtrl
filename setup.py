from setuptools import setup
from setuptools import find_packages
from pip._internal import main as pipmain
from os.path import join, dirname
import sys
if sys.version_info < (3, 7):
  sys.exit('Sorry, Python < 3.7 is not supported')

with open(join(dirname(__file__), "version"), 'r') as f:
  __version__ = f.read()

setup(name='rtrl',
      version=__version__,
      description='Real Time Reinforcement Learning',
      author='Simon Ramstedt',
      author_email='simonramstedt@gmail.com',
      url='https://github.com/rmst/rtrl',
      download_url='',
      license='',
      install_requires=[
            'numpy',
            'torch',
            'imageio',
            'imageio-ffmpeg',
            'pandas',
            'gym',
            'pyyaml',
            # 'line_profiler',
      ],
      extras_require={

      },
      scripts=[],
      packages=find_packages())


