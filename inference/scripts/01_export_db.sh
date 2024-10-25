#!/bin/bash

cd /home/vienhuynh/vienhuynh_research_sda/oclean

sudo docker-compose exec -T mongo bash -c "mongodump -d melinda_maria_staging --archive --gzip" > /tmp/melinda_stg_20240108.archive.gz && \
sudo docker-compose exec -T mongo bash -c "mongodump -d chan_luu_staging --archive --gzip" > /tmp/chanluu_stg_20240108.archive.gz && \
sudo docker-compose exec -T mongo bash -c "mongodump -d miz_mooz_staging --archive --gzip" > /tmp/mizmooz_stg_20240108.archive.gz && \
sudo docker-compose exec -T mongo bash -c "mongodump -d hammitt_staging --archive --gzip" > /tmp/hammitt_stg_20240108.archive.gz && \
sudo docker-compose exec -T mongo bash -c "mongodump -d raquel_allegra_staging --archive --gzip" > /tmp/raquel_stg_20240108.archive.gz && \
sudo docker-compose exec -T mongo bash -c "mongodump -d as98_staging --archive --gzip" > /tmp/as98_stg_20240108.archive.gz && \
sudo docker-compose exec -T mongo bash -c "mongodump -d melinda_maria --archive --gzip" > /tmp/melinda_20240108.archive.gz && \
sudo docker-compose exec -T mongo bash -c "mongodump -d chan_luu --archive --gzip" > /tmp/chanluu_20240108.archive.gz && \
sudo docker-compose exec -T mongo bash -c "mongodump -d miz_mooz --archive --gzip" > /tmp/mizmooz_20240108.archive.gz && \
sudo docker-compose exec -T mongo bash -c "mongodump -d hammitt --archive --gzip" > /tmp/hammitt_20240108.archive.gz && \
sudo docker-compose exec -T mongo bash -c "mongodump -d raquel_allegra --archive --gzip" > /tmp/raquel_20240108.archive.gz && \
sudo docker-compose exec -T mongo bash -c "mongodump -d as98 --archive --gzip" > /tmp/as98_20240108.archive.gz
