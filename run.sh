#!/bin/bash
cd findar
git pull
gcloud storage cp -r gs://area-findar/data .
docker-compose up
shutdown now
