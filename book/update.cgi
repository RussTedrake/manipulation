#!/usr/bin/env python3

import html
import os
import subprocess
import sys

# Disable chunked encoding - must be set before any output
os.environ["no-gzip"] = "1"
os.environ["no-chunked-encoding"] = "1"
os.environ["HTTP_NO_CHUNKED_ENCODING"] = "1"

try:
    # Do all work first
    git_output = subprocess.run(
        ["git", "fetch", "origin"], capture_output=True, text=True, check=True
    ).stdout

    git_output += subprocess.run(
        ["git", "reset", "--hard", "origin/master"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    git_output += subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    os.chdir("..")

    # Start the build process in background
    with open("/tmp/manipulation_build_docs.log", "w") as log_file:
        subprocess.Popen(
            [
                "/bin/bash",
                "-c",
                "source venv/bin/activate && "
                "poetry install --only docs && "
                "sphinx-build -M html manipulation /tmp/manip_doc && "
                "rm -rf book/python && "
                "cp -r /tmp/manip_doc/html book/python",
            ],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    # Build response body first
    body = (
        "<html><body>\n"
        "<p>pulling repo...</p>\n"
        f"<pre>{html.escape(git_output)}</pre>\n"
        "<p>done.</p>\n"
        "<p>Documentation build started in the background.</p>\n"
        "<p>Check /tmp/manipulation_build_docs.log for progress.</p>\n"
        "</body></html>\n"
    )

    # Build complete response
    response = (
        "Content-Type: text/html\r\n"
        "\r\n"
        "<html><body>\n"
        "<p>pulling repo...</p>\n"
        f"<pre>{html.escape(git_output)}</pre>\n"
        "<p>done.</p>\n"
        "<p>Documentation build started in the background.</p>\n"
        "<p>Check /tmp/manipulation_build_docs.log for progress.</p>\n"
        "</body></html>\n"
    )

    # Write response and immediately exit
    sys.stdout.write(response)
    sys.stdout.flush()

except Exception as e:
    error_response = (
        "Content-Type: text/html\r\n"
        "\r\n"
        "<html><body>\n"
        f"<p>Error: {html.escape(str(e))}</p>\n"
        "</body></html>\n"
    )
    sys.stdout.write(error_response)
    sys.stdout.flush()
