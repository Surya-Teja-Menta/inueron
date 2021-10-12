from setuptools import setup

setup(
    name='src',
    version='0.0.1',
    author='Surya Teja Menta',
    description='A tiny package for hand written image classification',
    author_email='mentasuryateja@gmail.com',
    packages=['src'],
    python_requires='>=3.7',
    install_requires=[
        'tensorflow',
        'matplotlib',
        'seaborn',
        'numpy',
        'pandas'
    ]
)