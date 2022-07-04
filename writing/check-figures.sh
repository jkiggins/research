#!/bin/bash

ack asvg objectives/obj1.tex | grep -Eo "figures/.+svg" | xargs -I {} ls {}
