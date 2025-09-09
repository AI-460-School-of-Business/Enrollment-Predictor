#!/bin/sh
set -e

echo "Waiting for Postgres..."
until nc -z $DB_HOST $DB_PORT; do
  sleep 1
done
echo "Postgres is up!"

exec python app.py
