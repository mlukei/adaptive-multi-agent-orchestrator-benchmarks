#!/bin/bash

cd /

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

# Install what the in-container app scripts (apps/*/*.py) import, plus
# pandas/scikit-learn for ad-hoc python usage by the shell agent.
/root/.local/bin/uv pip install --system --break-system-packages \
    fire==0.5.0 \
    "pandas>=2.1,<3" \
    "scikit-learn>=1.4,<2" \
    python-docx==1.1.0 \
    pillow \
    "pymupdf>=1.27.2,<1.27.2.2" \
    pypdf2 \
    pdf2docx \
    pytesseract \
    openai \
    icalendar \
    openpyxl \
    "xlrd>=2.0.1"
