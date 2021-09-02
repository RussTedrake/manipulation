
Over the course of the fall 2021 semester, we will be porting most chapters/notebooks from Colab to Deepnote.  We expect to use Colab only for notebooks that require the GPU, since Deepnote does not currently offer GPU machines on the free/student plans.

Our workflow with Google Colab is currently in a sad state, because Colab upgraded to Python 3.7 (Drake supports python 3.6, 3.8, and 3.9 by default, since those are the apt/homebrew supported configurations on ubuntu 18.04, 20.04, and mac, respectively).

As of today, `setup_manipulation_colab.py` works by pulling manually generated drake binaries for Python 3.7.  We use a somewhat scary header in the notebook to fetch that script and run the install.

Very soon, we expect to have `pip install drake` working on colab.  We should follow this with `pip install manipulation`, and the entire preamble can be simplified to a commented line like `#!pip install manipulation  # Uncomment this line and run it once to set up Google Colab.` 