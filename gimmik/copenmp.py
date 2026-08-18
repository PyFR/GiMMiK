from gimmik.base import MatMul


class COpenMPMatMul(MatMul):
    platform = 'c-openmp'
    basemeta = {}

    def _kernel_generators(self, dtype, dsize, *, sigs):
        yield ('cstream', {}, {})

