## To Run the Unit Tests

Make sure that you have done a recursive checkout in this repository, or have run

```bash
git submodule update --init --recursive
```
Then run
```bash
bazel test //...
```

If you would like to `bazel` to use a local installation of drake, you can set
the `DRAKE_INSTALL_DIR` environment variable. Otherwise it will look in
`/opt/drake`.

## Tips for developers

These are things that I often add to my preamble of the notebook (ever since vs code broke my pythonpath importing)
```
%load_ext autoreload
%autoreload 2
import sys
sys.path.append('/home/russt/drake-install/lib/python3.6/site-packages')
sys.path.append('/home/russt/manipulation')
```
