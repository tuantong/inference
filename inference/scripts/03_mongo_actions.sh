#!/bin/bash

KEY_FILE="/home/anhpham/.ssh/playmaker-ai.pem"
USERNAME="playmakeradmin"
SERVER="ec2-3-131-152-54.us-east-2.compute.amazonaws.com"

DB_NAMES=(
    "chan_luu"
    "chan_luu_staging"
    "melinda_maria"
    "melinda_maria_staging"
    "miz_mooz"
    "miz_mooz_staging"
    "as98"
    "as98_staging"
    "raquel_allegra"
    "raquel_allegra_staging"
    "hammitt"
    "hammitt_staging"
)

ssh -i "$KEY_FILE" "$USERNAME@$SERVER" << EOF
    mongo --host playmaker-mongo.cluster-cxmxugle6j7t.us-east-2.docdb.amazonaws.com --port 27017 --ssl --sslCAFile rds-combined-ca-bundle.pem -u playmakeradmin -p dGMVFeywAGK2xGZV << INNER_EOF
    $(for i in "${DB_NAMES[@]}"; do echo "use $i; db.dropDatabase()"; done)
    quit()
INNER_EOF
EOF