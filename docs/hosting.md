# CSAIL hosting

These operational details were provided by the repository owner in September
2026. SSH access, Apache configuration, and MySQL connectivity have since been
verified. The CGI installation described below was deployed and verified over
public HTTPS on September 7, 2026 (UTC).

## SSH access

The hosting VM is `underactuated-r1.csail.mit.edu`. The owner's usual access path
is to SSH to `login.csail.mit.edu`, complete interactive two-factor
authentication, and then SSH from that host to `underactuated-r1`:

```sh
# From the local machine:
ssh login.csail.mit.edu

# From login.csail.mit.edu, after authenticating:
ssh -o ProxyJump=none underactuated-r1
```

The owner can assist with interactive login and has sudo access on the VM.
Do not store credentials or authentication codes in this document.

The login host's SSH configuration adds a jump host for the VM; explicitly
disabling it permits the direct internal connection. For agent-assisted access,
the owner can keep a local SSH control master open using `ssh -M -N -S
/tmp/codex-csail.sock login.csail.mit.edu`; subsequent SSH commands can reuse that
socket with `-S`. Sudo on the VM requires an interactive password.

## Live checkout

The repository is checked out at `/var/www/manipulation` on the VM, and the site
is actively hosted from that directory. Changes to this checkout may affect the
live site. Apache serves `/var/www/manipulation/book` at
`https://manipulation.csail.mit.edu/` and already enables `.cgi` execution.

The book directory is owned by `www-data:www-data` with mode 755; the SSH user
cannot write it directly. Deployment needs sudo. Do not reset or clean the live
checkout: it contains untracked historical course directories.

## Bibliography database access

CSAIL administrators have restricted access to `mysql.csail.mit.edu` from
outside the immediate CSAIL network. The owner reports that the hosting VM can
reach that database server. A read-only query through the VM successfully
retrieved `Mason18`; all 162 current citation records were also verified.

The updated `book/htmlbook/install_html_meta_data.py` reads `elib_url` from
`book/chapters.json` and POSTs a JSON array of citation tags. Reference rendering
remains local. The metadata check test still requires the live endpoint.

The implementation is shared in `book/htmlbook/elib.cgi` and is served directly
at `https://manipulation.csail.mit.edu/htmlbook/elib.cgi`. It re-executes using
the repository's `.venv/bin/python` when started by Apache's system Python.
The existing `venv` used by
`update.cgi` is separate and is not changed. The new `.venv` needs MySQL
Connector/Python (deployment pins 9.4.0).

The CGI reads the existing read-only database credentials from `/etc/elib.json`
(root:www-data, mode 640), outside the web document root. It returns JSON with
`entries` keyed by tag and a `missing` array, limits requests to 1000 tags and
128 KiB, uses parameterized SQL, and omits private paper URLs. No authentication
is required for this limited bibliography metadata endpoint.

The htmlbook submodule is also consumed by the underactuated repository. Keep
project-specific URLs in each book's `chapters.json`, not in htmlbook. Its CGI
implementation can be reused by either book.

Deployment verification retrieved all 162 current citations and confirmed their
rendered references match the former direct-MySQL implementation. The live
endpoint returns HTTP 405 for GET and HTTP 400 for an invalid tag-array body.

After changing the shared CGI, update the htmlbook submodule commit as well as
the parent repository's submodule pointer. The deployed shared CGI must match
its committed version. No Apache restart is required for CGI
source updates.
