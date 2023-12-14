#!/bin/bash
cd findar
git pull
gcloud storage cp -r gs://area-findar/data .
docker-compose up
gcloud storage cp -r output gs://area-findar/output
gcloud storage cp -r data/current_data.csv gs://area-findar/data/existing_dataset.csv
shutdown now
