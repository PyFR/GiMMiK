import itertools as it
from importlib import resources
import json
from pathlib import Path
import pkgutil
import re

from mako.lookup import TemplateLookup
from mako.template import Template
import numpy as np


class _PlatformTemplateLookup(TemplateLookup):
    def __init__(self, platform):
        self.platform = platform

    def adjust_uri(self, uri, relto):
        return uri

    def get_template(self, name):
        platform = self.platform
        src = pkgutil.get_data(__name__, f'kernels/{platform}/{name}.mako')

        return Template(src, lookup=self)


def _dot(bfn, row, maxsplit=1):
    nzixs, = np.nonzero(row)

    if not nzixs.size:
        return '0.0'

    nsplit = max(min(maxsplit, nzixs.size // 3), 1)
    snzixs = np.array_split(nzixs, nsplit)

    frags = [' + '.join(f'{row[i]}*{bfn(i)}' for i in ix) for ix in snzixs]
    return ' + '.join(f'({f})' for f in frags)


def _partition(mat, into, by):
    if by == 'rows':
        return [list(range(i, len(mat), into)) for i in range(into)]
    elif by == 'cols':
        return [list(range(i, len(mat.T), into)) for i in range(into)]
    else:
        raise ValueError('Invalid partition by')


def _chunk(l, chunksz):
    l, n = iter(l), len(l)
    nchunks = -(-n // chunksz)

    return [list(it.islice(l, chunksz)) for i in range(nchunks)]


class MatMul:
    platform = None
    _float_suffix = 'f'

    def __init__(self, A, beta=0.0, aligne=None, n=None, ldb=None, ldc=None):
        self.A = A
        self.beta = beta
        self.aligne = aligne

        if n is None and ldb is None and ldc is None:
            self.n = self.ldb = self.ldc = None
        elif n is not None and ldb is not None and ldc is not None:
            if aligne is not None and (ldb % aligne or ldc % aligne):
                raise ValueError('ldb/ldc not compatible with aligne')

            self.n, self.ldb, self.ldc = n, ldb, ldc
        else:
            raise ValueError('Must provide all of (n, ldb, ldc) or none')

        # Check the matrix has a non-zero
        if not A.any():
            raise ValueError('A can not be empty')

        # Extract the shape of A
        self.m, self.k = m, k = A.shape

        # Determine the index of the first and last non-zero in each row of A
        self.afix = (A != 0).argmax(axis=1)
        self.alix = k - 1 - (A != 0)[:, ::-1].argmax(axis=1)

        # Mark rows of A which are all zero
        self.afix = np.where(np.any(A != 0, axis=1), self.afix, -1)
        self.alix = np.where(np.any(A != 0, axis=1), self.alix, -1)
        self.has_zero_rows = np.any(self.afix == -1)

        # Determine which entries of B partake in the multiplication
        self.bix = np.nonzero(np.any(A != 0, axis=0))[0]
        self.bix = {kx: k for k, kx in enumerate(self.bix)}

        # Create config cache
        self._config_cache = {}

    def kernels(self, dtype, kname='gimmik_mm', **kwargs):
        basemeta = self.basemeta

        # Process the data type
        dtype = np.dtype(dtype).type
        if dtype == np.float32:
            dtype, dsize = 'float', 4
        elif dtype == np.float64:
            dtype, dsize = 'double', 8
        else:
            raise ValueError('Invalid floating point data type')

        # Common template arguments
        baseargs = self._base_template_args(dtype, kname)

        # Incrementally generate and render the kernels
        gen = self._kernel_generators(dtype, dsize, **kwargs)
        try:
            resp = None
            while True:
                # Generate the next kernel in the sequence
                name, exargs, exmeta = gen.send(resp)

                # Merge in the base arguments and metadata
                args = baseargs | exargs
                meta = basemeta | exmeta

                # Render the kernel template
                src = self._render_kernel(dtype, name, args)

                # Post-process the metadata
                meta['tplname'] = name
                self._process_meta(meta)

                # Yield the source and metadata and await a response
                resp = yield (src, meta)
        except StopIteration:
            pass

    def _base_template_args(self, dtype, kname):
        return {
            'dtype': dtype, 'kname': kname,
            'A': self.A, 'beta': self.beta, 'width': 1,
            'm': self.m, 'n': self.n, 'k': self.k,
            'ldb': self.ldb, 'ldc': self.ldc,
            'afix': self.afix, 'alix': self.alix, 'bix': self.bix,
            'dot': _dot, 'partition': _partition, 'chunk': _chunk
        }

    def _process_meta(self, meta):
        pass

    def _get_config(self, key):
        try:
            return self._config_cache[key]
        except KeyError:
            cfgpath = Path('configs') / self.platform / f'{key}.json'
            cfgdata = (resources.files('gimmik') / cfgpath).read_text()
            self._config_cache[key] = json.loads(cfgdata)
            return self._config_cache[key]

    def _eval_condition(self, condition, stats):
        if 'all' in condition:
            return all(self._eval_condition(c, stats)
                       for c in condition['all'])
        if 'any' in condition:
            return any(self._eval_condition(c, stats)
                       for c in condition['any'])
        if 'not' in condition:
            return not self._eval_condition(condition['not'], stats)

        value = stats[condition['field']]
        op = next(k for k in condition if k != 'field')
        expected = condition[op]

        match op:
            case 'eq':
                return value == expected
            case 'ne':
                return value != expected
            case 'lt':
                return value is not None and value < expected
            case 'lte':
                return value is not None and value <= expected
            case 'gt':
                return value is not None and value > expected
            case 'gte':
                return value is not None and value >= expected
            case 'in':
                return value in expected
            case 'is_null':
                return value is None
            case 'is_not':
                return value is not None
            case 'divisible_by':
                return value is not None and value % expected == 0
            case 'is_null_or_divisible_by':
                return (value is None or value % expected == 0)
            case _:
                raise ValueError(f'op `{op}` not supported')

    def _render_kernel(self, dtype, tplname, tplargs):
        tpl = _PlatformTemplateLookup(self.platform).get_template(tplname)
        src = tpl.render(**tplargs)

        if dtype == 'float' and self._float_suffix:
            src = re.sub(r'(?<![\w.])(?=\d*[.eE])(?=\.?\d)'
                         r'\d*\.?\d*(?:[eE][+-]?\d+)?',
                         rf'\g<0>{self._float_suffix}', src)

        # Cleanup
        src = re.sub(r'[ \t]+$', '', src.strip(), flags=re.M)
        src = re.sub(r'\n{3,}', '\n\n', src) + '\n'
        return src
