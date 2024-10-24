#!/bin/bash

KEY_FILE="/home/anhpham/.ssh/playmaker-ai.pem"
USERNAME="playmakeradmin"
SERVER="ec2-3-131-152-54.us-east-2.compute.amazonaws.com"
DATE="20240108"

DB_NAMES=(
    "melinda_maria_staging"
    "chan_luu_staging"
    "miz_mooz_staging"
    "raquel_allegra_staging"
    "hammitt_staging"
    "as98_staging"
    "melinda_maria"
    "chan_luu"
    "miz_mooz"
    "hammitt"
    "raquel_allegra"
    "as98"
)

for DB_NAME in "${DB_NAMES[@]}"; do
    mongorestore -h playmaker-mongo.cluster-cxmxugle6j7t.us-east-2.docdb.amazonaws.com \
        -d "$DB_NAME" \
        --username playmakeradmin \
        --password dGMVFeywAGK2xGZV \
        --gzip \
        --archive < "./backup/${DB_NAME}_$DATE.archive.gz" \
        --ssl \
        --sslCAFile rds-combined-ca-bundle.pem
done
