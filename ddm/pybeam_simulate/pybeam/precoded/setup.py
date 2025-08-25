from setuptools import setup
from Cython.Build import cythonize

setup(
    name=["functions_constant",
          "functions_leakage"
          "functions_changing_thresholds"
          "functions_UGM"
          "functions_UGM_flip",
          "functions_DMC"],
    ext_modules=cythonize(["functions_constant.pyx", 
                           "functions_leakage.pyx",
                           "functions_changing_thresholds.pyx",
                           "functions_UGM.pyx",
                           "functions_UGM_flip.pyx",
                           "functions_DMC.pyx"]),
    zip_safe=False,
)