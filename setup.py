import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='dgc',
    url='https://github.com/uncbiag/dgc',
    author='Yifeng Shi',
    author_email='yifengs@cs.unc.edu',
    # Needed to actually package something
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    # Needed for dependencies
    install_requires=['numpy','torch','matplotlib'],
    # *strongly* suggested for sharing
    version='0.1.3',
    # The license can be anything you like
    license='UNC',
    description='An accompanying package for the paper, Deep Goal-Oriented Clustering',
    # We will also need a readme eventually (there will be a warning)
    long_description=long_description,
    long_description_content_type="text/markdown",
)
