from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='vec_hsqc',
      version='0.1',
      description='Predictive analytics on 2D NMR data',
      classifiers=[
	'Development Status :: 2 - Pre-Alpha',
	'Programming Language :: Python :: 2.7',
	'Intended Audience :: Science/Research'
      ], 
      url='http://github.com/kieranrimmer/vec_hsqc',
      author='Kieran Rimmer',
      author_email='kieranrimmer@gmail.com',
      license='',
      packages=['vec_hsqc'],
      install_requires=[
	'nmrglue==0.4', 'numpy==1.6.1', 'scipy', 'matplotlib', 'scikit-learn==0.13.1',
      ],
      zip_safe=False)
