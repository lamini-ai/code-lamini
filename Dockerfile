FROM python:3.9 as base

ARG PACKAGE_NAME="code-lamini"

# Install Ubuntu libraries
RUN apt-get -yq update

# Install python packages
WORKDIR /app/${PACKAGE_NAME}
COPY ./requirements.txt /app/${PACKAGE_NAME}/requirements.txt
RUN pip install -r requirements.txt

# Copy all files to the container
COPY ./code_lamini /app/${PACKAGE_NAME}/code_lamini
COPY ./scripts /app/${PACKAGE_NAME}/scripts

WORKDIR /app/${PACKAGE_NAME}

RUN chmod a+x /app/${PACKAGE_NAME}/scripts/start.sh

ENV PACKAGE_NAME=$PACKAGE_NAME
ENTRYPOINT ["/app/code-lamini/scripts/start.sh"]



