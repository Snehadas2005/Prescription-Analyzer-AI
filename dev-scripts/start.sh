#!/bin/bash
apt-get update && apt-get install -y python3 python3-pip
cd backend
python3 main.py
