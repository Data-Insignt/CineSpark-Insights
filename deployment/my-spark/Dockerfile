# 使用官方 Apache Spark 镜像作为基础镜像
FROM apache/spark:3.3.1

# 切换到 root 用户以安装软件包
USER root

# 安装 wget 和 bzip2，这对于安装 Miniconda 是必需的
RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates zip && \
    rm -rf /var/lib/apt/lists/*

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

# 设置 Spark 和 PySpark 相关的环境变量
ENV SPARK_HOME=/opt/spark
ENV PYSPARK_PYTHON=/opt/conda/envs/pyspark/bin/python
ENV PYTHONSTARTUP=/opt/conda/envs/pyspark/lib/python3.8/site-packages/pyspark/shell.py
ENV PATH /opt/conda/envs/pyspark/bin:$PATH

# 确保目标目录存在
RUN mkdir -p /opt/spark/python
RUN mkdir -p /opt/spark/python/lib

# 创建符号链接，指向 Conda 环境中的 PySpark
RUN ln -s /opt/conda/envs/pyspark/lib/python3.8/site-packages/pyspark /opt/spark/python/pyspark

# 自动找到 Py4J JAR 版本，并创建相应的符号链接
RUN PY4J_JAR_PATH=$(ls /opt/spark/jars/py4j-*.jar | head -n 1) && \
    PY4J_VERSION=$(basename $PY4J_JAR_PATH | sed 's/py4j-\(.*\).jar/\1/') && \
    ln -s $PY4J_JAR_PATH /opt/spark/python/lib/py4j-$PY4J_VERSION-src.zip

# 创建 pyspark.zip 并移动到 Spark 的 python/lib 目录
RUN cd /opt/conda/envs/pyspark/lib/python3.8/site-packages && \
    zip -r pyspark.zip pyspark && \
    mkdir -p /opt/spark/python/lib && \
    mv pyspark.zip /opt/spark/python/lib/

# 当容器启动时，默认激活 pyspark 环境
CMD [ "/bin/bash", "-c", "source /opt/conda/bin/activate pyspark && /bin/bash" ]

# 构建镜像 docker build -t my-spark:3.3.1 .