#!/bin/bash

export PATH=/usr/local/texlive/2022/bin/x86_64-linux:$PATH
pdflatex --shell-escape main.tex
