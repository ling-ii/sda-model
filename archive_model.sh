#!/usr/bin/env bash

NAME="best_model"
MODEL_NAME="${NAME}.pt"
CONFIG_NAME="${NAME}_config.txt"

# Check files exist
if [ ! -f ${MODEL_NAME} ]; then
	exit 1
fi

if [ ! -f ${CONFIG_NAME} ]; then
	exit 1
fi

# Check archive dir exists
ARCHIVE_NAME="model_archives"
if [ ! -d ${ARCHIVE_NAME} ]; then
	echo "${ARCHIVE_NAME} does not exist. Making new directory."
	mkdir ${ARCHIVE_NAME}
fi

# Check model number
LAST=$(ls -1 ${ARCHIVE_NAME} | tail -1 | cut -d'_' -f2)
if [ ! -n ${LAST} ]; then
	NEW=$(( ${LAST} + 1 ))
else
	NEW=0
fi

# Create new archive
MODEL_ARCHIVE_NAME="${ARCHIVE_NAME}/model_${NEW}/"
echo Creating new archive ${MODEL_ARCHIVE_NAME}
mkdir ${MODEL_ARCHIVE_NAME}

# Move files to archive
mv ${MODEL_NAME} ${CONFIG_NAME} ${MODEL_ARCHIVE_NAME}
