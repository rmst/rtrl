from setuptools import setup
from setuptools import find_packages
from pip._internal import main as pipmain

import sys
if sys.version_info < (3, 6):
  sys.exit('Sorry, Python < 3.6 is not supported')


setup(name='rtrl',
      version='0.1',
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
            # 'line_profiler',
      ],
      extras_require={

      },
      scripts=[],
      packages=find_packages())


