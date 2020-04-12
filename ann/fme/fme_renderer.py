#

class FmeRenderer(object):
    RENDER_MODE_CONSOLE = 1
    RENDER_MODE_GRAPH = 2

    def __init__(self, render_mode=RENDER_MODE_CONSOLE):
        self.name = 'apps.fme.FmeRender'
        self.render_mode = render_mode

    def render_obs(self, obs):
        pass