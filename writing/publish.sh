#!/bin/bash

cp -r ../spark/artifacts figures/

pdflatex --shell-escape main.tex
