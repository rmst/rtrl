from setuptools import setup
from setuptools import find_packages
from pip._internal import main as pipmain
from os.path import join, dirname
import sys
if sys.version_info < (3, 7):
  sys.exit('Sorry, Python < 3.7 is not supported')


setup(name='rtrl',
      version="0.1",
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
      scripts=[
            "scripts/rtrl-run"
      ],
      packages=find_packages())


