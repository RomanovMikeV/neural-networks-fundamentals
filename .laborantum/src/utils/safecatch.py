import sys, traceback
from contextlib import contextmanager

@contextmanager
def safecatch():
    try:
        yield
    except Exception:
        print("!!! EXCEPTION !!!")
        print("If you see this message -- something is wrong here")
        traceback.print_exc(file=sys.stdout)
        print("!!!!!!!!!!!!!!!!!")
