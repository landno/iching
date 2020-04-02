#
from apps.ogml.ogml_app import OgmlApp

def main():
    print('易经量化交易系统 v0.0.1')
    app = OgmlApp()
    app.startup()

if '__main__' == __name__:
    main()