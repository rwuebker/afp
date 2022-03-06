import setuptools

setuptools.setup(
    name="afpapi",
    version="0.1.2",
    url="https://github.com/rwuebker/afpapi",
    author="Rick Wuebker",
    author_email="richard_wuebker@berkeley.edu",
    description="Python package to check returns of any weights of Ken French 49 industry portfolios against MVO and Equal Weights",
    packages=setuptools.find_packages(),
    install_requires=[],
    classifiers=[
        'Programming Language :: Python :: 3'
    ],
    include_package_data=True
)
