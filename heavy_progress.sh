#!/usr/bin/env bash
python3 /tmp/meter.py
echo ""
echo "log: tail -n 80 heavy.log"
echo "resume: bash heavy_resume.sh"
