#!/bin/bash
if [ ! -d "venv" ]; then
    python -m venv venv
    ./venv/bin/pip install -r requirements.txt
fi
./venv/bin/streamlit run app.py
