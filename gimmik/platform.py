# -*- coding: utf-8 -*-

class _Enum(type):
    def __getattr__(self, name):
        return self._list.index(name)

class Platform:
	"""Enumerates the available platforms for code generation.
	
	Currently available platforms are %s.
	"""
	_list = ['CUDA', 'OPENCL']
	__metaclass__ = _Enum
	__doc__ %= _list