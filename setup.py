from setuptools import setup

setup(name='myshkin',
      version='0.0.1',
      description='Utilities for training deep nets in TensorFlow',
      author='Jake Snell',
      author_email='jsnell10@gmail.com',
      install_requires=[
          'attrdict',
          'docopt',
          'h5py',
          'numpy',
          'pyyaml',
          'tensorflow'
      ],
      packages=['myshkin'])
