from setuptools import setup, find_packages

setup(
    name='genetic_algorithms',
    version='0.1.0',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'pandas',
        'numpy',
        'scipy'
    ],
    entry_points={
        'console_scripts': [
            'ga-map=main:main'
        ]
    }
)
