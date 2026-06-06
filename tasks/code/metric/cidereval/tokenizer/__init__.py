__author__ = 'hfang'
# edited by Michele Cafagna
from .ptbtokenizer import PTBTokenizer
try:
    from .simpletokenizer import SimpleTokenizer
except Exception:
    SimpleTokenizer = None
