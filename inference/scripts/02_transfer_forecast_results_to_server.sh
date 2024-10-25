#!/bin/bash

# Set the key file and server details
KEY_FILE="/home/anhpham/.ssh/playmaker-ai.pem"
USERNAME="playmakeradmin"
SERVER="ec2-3-131-152-54.us-east-2.compute.amazonaws.com"
REMOTE_DIR="backup/"

# Set the local directory where your files are located
LOCAL_DIR="/tmp"

# Set the date variable
DATE="20240108"

# Set the list of files to be transferred
FILES=(
    $LOCAL_DIR/"melinda_stg_$DATE*"
    $LOCAL_DIR/"chanluu_stg_$DATE*"
    $LOCAL_DIR/"mizmooz_stg_$DATE*"
    $LOCAL_DIR/"hammitt_stg_$DATE*"
    $LOCAL_DIR/"raquel_stg_$DATE*"
    $LOCAL_DIR/"as98_stg_$DATE*"
    $LOCAL_DIR/"melinda_$DATE*"
    $LOCAL_DIR/"chanluu_$DATE*"
    $LOCAL_DIR/"mizmooz_$DATE*"
    $LOCAL_DIR/"hammitt_$DATE*"
    $LOCAL_DIR/"raquel_$DATE*"
    $LOCAL_DIR/"as98_$DATE*"
)

# SSH into the server and copy the files
sftp -i "$KEY_FILE" "$USERNAME@$SERVER" << EOF
    cd "$REMOTE_DIR"
    $(for i in "${FILES[@]}"; do echo "put $i"; done)
    exit
EOF

echo "Files transferred successfully!"
