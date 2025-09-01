from setuptools import setup, Extension
import subprocess
import sys

def get_pybind_include():
    """Get pybind11 include path"""
    try:
        import pybind11
        return pybind11.get_include()
    except ImportError:
        # If pybind11 is not installed, install it first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11"])
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'power_svd_cpp',
        ['src/power_svd.cpp'],
        include_dirs=[
            'include/',
            get_pybind_include(),
        ],
        language='c++',
        extra_compile_args=['-std=c++11'],
    ),
]

setup(
    name='power_svd',
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=["numpy>=1.19.0", "pybind11>=2.6.0"],
)