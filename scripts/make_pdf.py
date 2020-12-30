
# Constructs the pdf from the html content.  Run it from the manipulation root:
# python3 scripts/make_pdf.py && xpdf manipulation.pdf
# or, this *should* work (but isn't):
# xpdf -remote foo "openFile(manipulation.pdf)" &  
# python3 scripts/make_pdf.py && xpdf -remote foo reload
# 
# Requires PrinceXML
# sudo apt install prince

# Also requires puppeteer for offline rendering of mathjax, etc.
# 
# cd ~
# curl -sL https://deb.nodesource.com/setup_10.x -o nodesource_setup.sh
# sudo bash nodesource_setup.sh
# sudo apt install -y nodejs
# npm i puppeteer --save
# 
# Note: I had to manually npm i some of the missing deps on my first try to get # the installer to complete without errors.

import json
import os

chapters = json.load(open("chapters.json"))
html_files = ['index.html'] + [s + '.html' for s in chapters['chapter_ids']]
input_files = ' '.join(html_files) 

# Working on offline rendering 
# - User node render_html.js to make a local .html with js evaluated
# - mathjax rendering is still off (extra line breaks).
#   - tried tex-svg.js, but it has the same line break issue.

# MAYBE: prince-books?  https://www.princexml.com/doc/prince-for-books/
# TODO: Remove prince watermark from first page?  (How much does it cost?)
os.system(f"prince -s scripts/pdf.css {input_files} -o manipulation.pdf")
