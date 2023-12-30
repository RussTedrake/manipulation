#!/usr/bin/perl

use strict;
use warnings;

use CGI;
my $r = new CGI;

print $r->header();
print "pulling repo...<br/>";
system 'git fetch origin && git reset --hard origin/master';
system 'git submodule update --init --recursive';
print "<br/>done.";

print "building documentation...<br/>";
system 'rm -rf book/python && sphinx-build -M html manipulation /tmp/manip_doc && cp -r /tmp/manip_doc/html book/python';
print "<br/>done.";

print $r->end_html;
