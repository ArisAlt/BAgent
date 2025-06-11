import importlib, os, sys
_repo_root = os.path.dirname(os.path.dirname(__file__))
importlib.import_module('sitecustomize')
src = os.path.join(_repo_root, 'src')
if src not in sys.path:
    sys.path.insert(0, src)
