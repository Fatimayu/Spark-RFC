FROM openjdk:11
ENV ANT_VERSION 1.10.12
RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*
RUN wget --no-check-certificate https://dlcdn.apache.org//ant/binaries/apache-ant-${ANT_VERSION}-bin.tar.gz
RUN tar -xzf apache-ant-${ANT_VERSION}-bin.tar.gz
RUN mv apache-ant-${ANT_VERSION} /opt/ant
RUN rm apache-ant-${ANT_VERSION}-bin.tar.gz
RUN mkdir predictML
RUN mkdir predictML/rfc_model.model
RUN mkdir predictML/third-party
RUN mkdir predictML/lib
ENV ANT_HOME /opt/ant
ENV PATH ${PATH}:/opt/ant/bin

# COPY TrainML.jar ./predictML/
COPY ValidationDataset.csv ./predictML/
COPY TrainingDataset.csv ./predictML/
COPY build.xml ./predictML/
COPY TrainML.java ./predictML/
COPY MakePrediction.java ./predictML/
ADD rfc_model.model ./predictML/rfc_model.model
ADD lib ./predictML/lib
ADD third-party ./predictML/third-party
CMD cd predictML && ant predict



