#!/bin/bash

makepdf(){
    pdflatex --shell-escape main
}

makeglossary() {
    makeglossaries main
}

makebib() {
    # bibtex main
    biber main
}

makepdf
makebib
makeglossary
makepdf
makepdf

