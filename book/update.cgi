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
system 'source venv/bin/activate';
system 'poetry install';
system 'pip3 install sphinx myst-parser sphinx_rtd_theme';
print "</p><p>";
system 'rm -rf book/python && sphinx-build -M html manipulation /tmp/manip_doc && cp -r /tmp/manip_doc/html book/python';
print "<br/>done.</p>";

print $r->end_html;
