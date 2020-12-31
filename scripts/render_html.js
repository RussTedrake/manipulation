// Run with 
// node render_html.js url output_file

'use strict';

const puppeteer = require('puppeteer');
const fs = require('fs');

(async function main() {
  try {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    
    const url = process.argv[2];
    const output_file = process.argv[3];
    await page.goto(url);
    const html = await page.content();
    fs.writeFileSync(output_file, html);
    await browser.close();
  } catch (err) {
    console.error(err);
  }
})();