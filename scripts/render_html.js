'use strict';

const puppeteer = require('puppeteer');
const fs = require('fs');

(async function main() {
  try {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();

    await page.goto('http://manipulation.csail.mit.edu/pick.html');
//    await page.goto('file:///home/russt/manipulation/pick.html');
    const html = await page.content();
    fs.writeFileSync('rendered_pick.html', html);
    await browser.close();
  } catch (err) {
    console.error(err);
  }
})();