# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.08-py3
FROM ${FROM_IMAGE_NAME}

RUN FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX" \
    CPATH=/usr/local/cuda/include \
    pip install --no-cache-dir torch-geometric==1.7.2

ADD ./requirements.txt .
RUN pip install --disable-pip-version-check -r requirements.txt

RUN conda install -y -c rapidsai-nightly -c conda-forge --no-deps\
    libcugraphops=22.10.00a220822=cuda11_g58eeb79_23\
    librmm=22.10.00a220822=cuda11_gadcfb934_9\
    spdlog=1.8.5=h4bd325d_1 &&\
    conda clean -ya

WORKDIR /workspace/oc20
ADD . /workspace/oc20 

RUN rm -rf /opt/cugraph-ops-binding/build &&\
    mkdir -p /opt/cugraph-ops-binding/build &&\
    cd /opt/cugraph-ops-binding/build &&\
    cmake /workspace/oc20/cugraph-ops-binding/ &&\
    make -j2
ENV PYTHONPATH="${PYTHONPATH}:/opt/cugraph-ops-binding/build"

RUN cp pyg_patch/__init__.py /opt/conda/lib/python3.8/site-packages/torch_geometric
RUN cp pyg_patch/data/* /opt/conda/lib/python3.8/site-packages/torch_geometric/data

RUN pip install -e .
ENV OMP_NUM_THREADS=2
