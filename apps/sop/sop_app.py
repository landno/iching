#
from apps.sop.qt_50etf import Qt50etf

class SopApp(object):
    def __init__(self):
        self.name = ''

    def startup(self, args={}):
        print('股票期权平台 v0.0.3')
        etf = Qt50etf()
    