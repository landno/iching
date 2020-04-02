#
from apps.ogml.ogml_app import OgmlApp
from apps.tp.tp_app import TpApp
from apps.rxgb.rxgb_app import RxgbApp

def main():
    print('易经量化交易系统 v0.0.1')
    #app = OgmlApp()
    #app = TpApp()
    app = RxgbApp()
    app.startup()

if '__main__' == __name__:
    main()