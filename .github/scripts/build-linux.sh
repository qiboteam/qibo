#!/bin/bash

set -e -x

bash --version
rpm -q centos-release

# enter repository folder
cd /io

# load retry script
source .github/scripts/retry.sh

# check python versions
ls /opt/python

if [ $PYTHON_VERSION == "3.5" ]; then
    PYBIN="/opt/python/cp35-cp35m/bin"
elif [ $PYTHON_VERSION == "3.6" ]; then
    PYBIN="/opt/python/cp36-cp36m/bin"
elif [ $PYTHON_VERSION == "3.7" ]; then
    PYBIN="/opt/python/cp37-cp37m/bin"
elif [ $PYTHON_VERSION == "3.8" ]; then
    PYBIN="/opt/python/cp38-cp38/bin"
elif [ $PYTHON_VERSION == "3.9" ]; then
    PYBIN="/opt/python/cp39-cp39/bin"
else
    echo "Unsupported Python version $PYTHON_VERSION"
    exit 1
fi

# install cuda-11.0
curl https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-repo-rhel7-10.2.89-1.x86_64.rpm -o cuda-repo.rpm
rpm -i cuda-repo.rpm
retry yum install -y cuda-compiler-11-0 cuda-libraries-devel-11-0

# set env variables
export PYTHON=$PYBIN/python
export CUDA_PATH=/usr/local/cuda-11.0/

# build wheel
retry $PYTHON -m pip install -r requirements.txt
retry $PYTHON setup.py bdist_wheel --dist-dir=wheelhouse

# patch auditwheel
POLICY_JSON=$(find / -name policy.json)
sed -i "s/libresolv.so.2\"/libresolv.so.2\", \"libtensorflow_framework.so.2\"/g" $POLICY_JSON
cat $POLICY_JSON

# repair wheel
auditwheel show wheelhouse/*.whl
auditwheel repair wheelhouse/*.whl -w wheelhouse

# move the wheel to dist/ folder
mkdir -p dist
mv wheelhouse/qibo*manylinux2010*.whl dist/
