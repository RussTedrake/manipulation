#!/usr/bin/perl

use strict;
use warnings;

use CGI;
my $r = new CGI;

print $r->header();
print "<p>pulling repo...<br/>";
system 'git fetch origin && git reset --hard origin/master';
system 'git submodule update --init --recursive';
print "<br/>done.</p>";

print "<p>building documentation...<br/>";
chdir "..";

# When I've had dependencies get "stuck" on the server, I've had to update them
# manually with e.g. `sudo -u www-data /var/www/manipulation/venv/bin/pip
# install drake==0.0.20250110 --extra-index-url
# https://drake-packages.csail.mit.edu/whl/nightly/`
my $status = system('/bin/bash', '-c', '
    source venv/bin/activate &&
    poetry install --only docs &&
    sphinx-build -M html manipulation /tmp/manip_doc &&
    rm -rf book/python &&
    cp -r /tmp/manip_doc/html book/python
');

if ($status == 0) {
    print "<br/>done.</p>";
} else {
    print "<br/>Error occurred: $status</p>";
}

print $r->end_html;
