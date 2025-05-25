# -*- coding: utf-8 -*-
# file: run_label_studio_serv.sh[


set -x

CURR_DIR=$(pwd)
PYTHON=$(which python3)
PYTHON_ENV_DIR=${CURR_DIR}/_pyenv_label_studio
LABEL_STUDIO_VERSION="1.16.0"
WORKSPACE_DIR="./_label_studio"

USERNAME="admin@labelstudio.com"
PASSWORD="1111"
DATA_DIR=${WORKSPACE_DIR}/data
PORT=18089


function init() {
  ${PYTHON} -m venv ${PYTHON_ENV_DIR} --copies
  ${PYTHON_ENV_DIR}/bin/pip install label-studio==${LABEL_STUDIO_VERSION}
  mkdir -p ${DATA_DIR}
}


function start() {
  INACTIVITY_SESSION_TIMEOUT_ENABLED=0 \
  USE_ENFORCE_CSRF_CHECKS=false \
  ${PYTHON_ENV_DIR}/bin/label-studio start \
    --data-dir ${DATA_DIR} \
    --username ${USERNAME} \
    --password ${PASSWORD} \
    --port ${PORT}
}


function main() {
  echo "Starting"
  init
  start
  echo "Finished"
}


main
