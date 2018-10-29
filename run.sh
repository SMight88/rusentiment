#!/usr/bin/env bash

gunicorn --timeout 130 --bind 0.0.0.0:8000 wsgi:app