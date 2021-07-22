# -*- coding: utf-8 -*-

from gimmik.backends import get_backend

class BaseTest(object):
    def __init__(self, platform):
        self.platform = platform

        cfg = default_cfg()
        self.cfg = cfg

        backend = get_backend(platform, cfg)
        self.backend = backend