import setuptools 
from anndata_tensorstore._version import version

version = version
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="anndata-tensorstore",
    version=version,
    url="https://github.com/xueziwei/anndata-tensorstore",
    author="Ziwei Xue",
    author_email="xueziweisz@gmail.com",
    description="Anndata Tensorstore Extension: Save and Load anndata to/from Tensorstore for random access",
    long_description_content_type='text/plain',
    packages=setuptools.find_packages(exclude=[
        "*docs*",
    ]),
    install_requires=[
        'anndata',
        'tensorstore',
        'pandas',
        'numpy',
        'scipy',
        'anndata'
    ],
    include_package_data=False,
)
