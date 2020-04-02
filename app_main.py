#
from apps.ogml.ogml_app import OgmlApp
from apps.tp.tp_app import TpApp
from apps.rxgb.rxgb_app import RxgbApp
from apps.asml.asml_app import AsmlApp

def main():
    print('易经量化交易系统 v0.0.1')
    #app = OgmlApp()
    #app = TpApp()
    #app = RxgbApp()
    app = AsmlApp()
    app.startup()

if '__main__' == __name__:
    main()