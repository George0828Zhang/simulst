# Flashlight Installation
Assume that you use conda, and the conda environment is called `env_name`. Other options such as venv or local install can work, but not described here. Unfortunately many of the procedure assumes sudoer privilege.
```bash
conda activate env_name
``` 
## Dependencies
1. CUDA (nvcc compiler). See [nvcc installation](./apex_installation.md#install-matching-nvcc-compiler).
2. OpenBLAS
```bash
sudo apt update
sudo apt install libopenblas-dev
```
2. Intel mkl
```bash
sudo apt install intel-mkl
conda install mkl-include  # or pip install
```
3. KenLM
- See and install [KenLM dependencies](https://kheafield.com/code/kenlm/dependencies/).
- For instance, after building boost, link it to cmake:
```bash
export CMAKE_INCLUDE_PATH=`realpath ~/utility/boost_1_75_0`
export CMAKE_LIBRARY_PATH=`realpath ~/utility/boost_1_75_0/lib`
```
- KenLM
```bash
cd ~/utility
wget https://kheafield.com/code/kenlm.tar.gz
tar zxvf kenlm.tar.gz
mkdir kenlm/build
cd kenlm/build
cmake .. -DCMAKE_C_COMPILER=/usr/bin/gcc-9 -DCMAKE_CXX_COMPILER=/usr/bin/g++-9  # change to an appropriate version for your system. I use 9.
make -j2
```

4. FFTW3
```bash
cd ~/utility
wget http://www.fftw.org/fftw-3.3.10.tar.gz
tar zxvf fftw-3.3.10.tar.gz
cd fftw-3.3.10/
mkdir build && cd build
```
- Option 1: System install
```bash
cmake ..
make -j 4
make install
sudo cp FFTW3LibraryDepends.cmake /usr/local/lib/cmake/fftw3/
```
- Option 2: User install
```bash
cmake .. -DCMAKE_INSTALL_PREFIX:PATH=~/utility/fftw-3.3.10/install
make -j 4
make install
```


<!-- 5. Googletest
```bash
git clone https://github.com/google/googletest.git -b release-1.11.0
cd googletest
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX:PATH=~/utility/googletest/install -DCMAKE_C_COMPILER=/usr/bin/gcc-9 -DCMAKE_CXX_COMPILER=/usr/bin/g++-9
make
make install
# export GTEST_INCLUDE_DIR="~/utility/googletest/install/include"
# export GTEST_LIBRARY="~/utility/googletest/install/lib"
# export GTEST_MAIN_LIBRARY="~/utility/googletest/install/lib"
``` -->

## Install Flashlight
1. Clone repo
```bash
git clone https://github.com/flashlight/flashlight.git
cd flashlight/bindings/python
```

<!-- 2. Add CC and CXX versions in `setup.py`:
```python
# Add these lines to cmake_args
# Set versions yourself.
cmake_args += [
    "-DCMAKE_C_COMPILER=/usr/bin/gcc-9",    
    "-DCMAKE_CXX_COMPILER=/usr/bin/g++-9"
]
```
Also change host compiler for nvcc:
```bash
export CUDAHOSTCXX=/usr/bin/g++-9
``` -->

2. (Optional) Add local install paths (if any)
```bash
# conda's mkl-include
export CMAKE_INCLUDE_PATH=`realpath ~/anaconda3/envs/env_name/include`
# kenlm
export KENLM_ROOT=~/utility/kenlm
# fftw3
export FFTW3_DIR=~/utility/fftw-3.3.10/install
```
- Add paths in `setup.py`:
```python
# Add these lines to cmake_args
# Chang <home> to your $HOME (e.g. /home/user).
cmake_args += [
    "-DKENLM_LIB=<home>/utility/kenlm",
    "-DKENLM_UTIL_LIB=<home>/utility/kenlm/util",
]
```

4. Install
```bash
python setup.py install
```