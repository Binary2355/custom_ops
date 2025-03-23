# if nvjitlink not in LD_LIBRARY_PATH, add it
if [[ ":$LD_LIBRARY_PATH:" != *":$(/root/miniconda3/bin/python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):"* ]]; then
    export LD_LIBRARY_PATH=$(/root/miniconda3/bin/python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):$LD_LIBRARY_PATH
fi

mkdir -p build
cd build

cmake \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DPYTHON_EXECUTABLE=/root/miniconda3/bin/python \
  -DPYTHON_INCLUDE_DIR=/root/miniconda3/include/python3.10 \
  -DPYTHON_LIBRARY=/root/miniconda3/lib/libpython3.10.so \
  -DCMAKE_PREFIX_PATH=`/root/miniconda3/bin/python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
make -j