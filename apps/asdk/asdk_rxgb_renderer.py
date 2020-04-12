#
from ann.fme.fme_renderer import FmeRenderer

class AsdkRxgbRenderer(FmeRenderer):
    def __init__(self):
        self.name = 'apps.asdk.AsdkRxgbRenderer'

    def render_obs(self, obs):
        print('AsdkRxgbRenderer.render_obs...')