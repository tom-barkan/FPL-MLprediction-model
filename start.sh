#!/bin/sh
exec streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
