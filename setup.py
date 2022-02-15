from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='dgc',
    url='https://github.com/uncbiag/dgc',
    author='Yifeng Shi',
    author_email='yifengs@cs.unc.edu',
    # Needed to actually package something
    packages=['dgc'],
    # Needed for dependencies
    install_requires=['numpy','pytorch'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='UNC',
    description='An accompanying package for the paper, Deep Goal-Oriented Clustering',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.txt').read(),
)
