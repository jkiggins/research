#!/bin/bash

makepdf(){
    pdflatex --shell-escape main
}

makeglossary() {
    makeglossaries main
}

makebib() {
    bibtex main
}

makepdf
makebib
makeglossary
makepdf
makepdf

