from distutils.core import setup


def readme():
    """Import the README.md Markdown file and try to convert it to RST format."""
    try:
        import pypandoc
        return pypandoc.convert('README.md', 'rst')
    except(IOError, ImportError):
        with open('README.md') as readme_file:
            return readme_file.read()


setup(
    name='NLP classification',
    version='0.1',
    description='Insincere Questions classification',
    long_description=readme(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    url='https://github.com/yungnien/production',
    author='yungnien yang', 
    author_email='yungnien@gmail.com',  # Substitute your email
    license='MIT',
    packages=['production'],
    install_requires=[
        'pypandoc>=1.4',
        'pandas>=0.25.1',
        'scikit-learn>=0.21.3',
        'scipy>=1.3.1',
        'matplotlib>=3.1.1',
	'pytest>=5.2.0',
        'pytest-runner>=5.1',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],

)
