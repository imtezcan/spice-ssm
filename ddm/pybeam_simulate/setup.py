from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(name='pybeam',
      version='0.23',
      packages=find_packages(),
      ext_modules = cythonize(["pybeam/precoded/functions_constant.pyx", 
                               "pybeam/precoded/functions_leakage.pyx",
                               "pybeam/precoded/functions_changing_thresholds.pyx",
                               "pybeam/precoded/functions_UGM.pyx",
                               "pybeam/precoded/functions_UGM_flip.pyx",
                               "pybeam/precoded/functions_DMC.pyx"]),
      zip_safe=False)
