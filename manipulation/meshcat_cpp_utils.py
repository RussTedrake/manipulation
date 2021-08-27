import os
import sys

from pydrake.common import set_log_level
from pydrake.geometry import Meshcat
from pyngrok import ngrok

def StartMeshcat():
  """
  A wrapper around the Meshcat constructor that supports Deepnote and Google Colab via ngrok when necessary.
  """
  prev_log_level = set_log_level("warn")
  use_ngrok = False
  if ("DEEPNOTE_PROJECT_ID" in os.environ):
    # Deepnote exposes port 8080 (only).  If we need multiple meshcats, then we # fall back to ngrok.
    try:
      meshcat = Meshcat(8080)
    except:
      use_ngrok = True
    else:
      meshcat.set_web_url(
        f'https://{os.environ["DEEPNOTE_PROJECT_ID"]}.deepnoteproject.com')
      set_log_level(prev_log_level)
      print(f"Meshcat is now available at {meshcat.web_url()}");
      return meshcat

  if 'google.colab' in sys.modules:
    use_ngrok = True

  meshcat = Meshcat()
  if use_ngrok:
    http_tunnel = ngrok.connect(meshcat.port(), bind_tls=False)
    meshcat.set_web_url(http_tunnel.publich_url());

  set_log_level(prev_log_level)
  print(f"Meshcat is now available at {meshcat.web_url()}");
  return meshcat
