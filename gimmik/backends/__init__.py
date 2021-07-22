# -*- coding: utf-8 -*-

from gimmik.backends.base import BaseBackend
from gimmik.backends.cuda import CUDABackend
from gimmik.backends.openmp import OpenMPBackend
from gimmik.util import subclass_where


def get_backend(name, cfg):
    return subclass_where(BaseBackend, name=name.lower())(cfg)
