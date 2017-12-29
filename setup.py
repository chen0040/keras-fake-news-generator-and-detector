from setuptools import find_packages
from setuptools import setup


setup(name='fake_news',
      version='0.0.1',
      description='Fake News Generator and Detector',
      author='Xianshun Chen',
      author_email='xs0040@gmail.com',
      url='https://github.com/chen0040/keras-fake-news-generator-and-detector',
      download_url='https://github.com/chen0040/keras-fake-news-generator-and-detector/tarball/0.0.1',
      license='MIT',
      install_requires=['Keras==2.1.2'],
      packages=find_packages())
