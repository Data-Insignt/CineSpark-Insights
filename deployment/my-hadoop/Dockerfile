# 使用官方 Apache Hadoop 镜像作为基础镜像
FROM apache/hadoop:3

# 切换到 root 用户以安装软件包
USER root

# 安装 wget、bzip2、ca-certificates 和 zip
RUN yum update -y && \
    yum install -y wget bzip2 ca-certificates zip && \
    yum clean all

# 下载并安装最新版本的 Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# 设置环境变量以包括 Conda
ENV PATH="/opt/conda/bin:$PATH"

# 创建一个名为 pyspark 的 Conda 环境，安装 Python 3.8
RUN conda create -n pyspark python=3.8 -y

# 激活 pyspark 环境，并在其中安装 PySpark
RUN echo "source activate pyspark" > ~/.bashrc && \
    /bin/bash -c "source activate pyspark && conda install -c conda-forge pyspark=3.3.1 -y"

# 设置 Spark 和 Python 相关的环境变量
ENV PYSPARK_PYTHON=/opt/conda/envs/pyspark/bin/python
ENV PATH /opt/conda/envs/pyspark/bin:$PATH

# 当容器启动时，默认激活 pyspark 环境
CMD [ "/bin/bash", "-c", "source /opt/conda/bin/activate pyspark && /bin/bash" ]

# 构建镜像 docker build -t my-hadoop:3 .