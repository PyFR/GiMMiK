import itertools as it
from importlib import resources
import json
from pathlib import Path
import pkgutil
import re
from uuid import uuid4

from mako.lookup import TemplateLookup
from mako.template import Template
import numpy as np


# Call signatures a kernel may have
SIG_BC = 'bc'
SIG_ABC = 'abc'
SIG_BDESC_CDESC = 'bdesc-cdesc'
SIG_BDESC_C = 'bdesc-c'

# Argument lists per signature, for a baked and for a runtime n
_SIG_ARGS = {
    SIG_BC: {
        'baked': ('b', 'c'),
        'runtime': ('n', 'b', 'ldb', 'c', 'ldc')
    },
    SIG_ABC: {
        'baked': ('a', 'b', 'c'),
        'runtime': ('a', 'n', 'b', 'ldb', 'c', 'ldc')
    },
    SIG_BDESC_CDESC: {
        'baked': ('b_desc', 'c_desc'),
        'runtime': None
    },
    SIG_BDESC_C: {
        'baked': ('b_desc', 'c'),
        'runtime': None
    }
}

SIGS = frozenset(_SIG_ARGS)

# Operand kinds an argument may need the caller to prepare
OPERAND_BUFFER = 'buffer'
OPERAND_TENSORMAP = 'tensormap'


def sig_of(meta):
    # The signature a kernel reports, defaulting to the ubiquitous one
    return meta.get('sig', SIG_BC)


def tensormap_spec(operand, dtype, box, global_dim, global_stride,
                   elem_stride=None, interleave='none', swizzle='none',
                   l2_promotion='none', oob_fill='none'):
    # Describes a tensor map the caller must encode for a descriptor argument
    if len(box) != len(global_dim):
        raise ValueError('box and global_dim must have equal rank')

    if len(global_stride) != len(global_dim) - 1:
        raise ValueError('global_stride must have rank - 1 entries')

    return {
        'kind': OPERAND_TENSORMAP,
        'operand': operand,
        'rank': len(box),
        'dtype': np.dtype(dtype),
        'box': tuple(box),
        'global_dim': tuple(global_dim),
        'global_stride': tuple(global_stride),
        'elem_stride': tuple(elem_stride or (1,)*len(box)),
        'interleave': interleave,
        'swizzle': swizzle,
        'l2_promotion': l2_promotion,
        'oob_fill': oob_fill
    }


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

    # Metadata every kernel from this platform carries
    basemeta = {}

    # Call signatures this platform is capable of emitting
    sigs = frozenset({SIG_BC})

    # Thresholds for the default viability heuristic
    max_unique = 28
    max_density = 0.15

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

        # Identity for the metadata we hand out, and the packers behind it
        self._uuid = uuid4().hex
        self._packers = {}

    def _unrolled_viable(self):
        # True when the fully unrolled kernels can beat a vendor GEMM
        nuq = len(np.unique(np.abs(self.A)))
        density = np.count_nonzero(self.A) / self.A.size

        return nuq <= self.max_unique or density <= self.max_density

    def kernels(self, dtype, kname='gimmik_mm', *, sigs=frozenset({SIG_BC}),
                **kwargs):
        dtype, dsize = self._process_dtype(dtype)
        sigs = self._process_sigs(sigs)

        return self._kernels(dtype, dsize, kname, sigs, **kwargs)

    def available_sigs(self, dtype, **kwargs):
        # Signatures offered for the operator and target given
        dtype, dsize = self._process_dtype(dtype)
        gen = self._kernel_generators(dtype, dsize, sigs=self.sigs, **kwargs)

        found, resp = set(), None
        try:
            while True:
                name, exargs, exmeta = gen.send(resp)
                found.add(sig_of(exmeta))
        except StopIteration:
            pass

        return found

    def _process_dtype(self, dtype):
        dtype = np.dtype(dtype).type
        if dtype == np.float32:
            return 'float', 4
        elif dtype == np.float64:
            return 'double', 8
        else:
            raise ValueError('Invalid floating point data type')

    def _process_sigs(self, sigs):
        if isinstance(sigs, str):
            raise ValueError('sigs must be a set of names, not a string')

        sigs = frozenset(sigs)

        if bad := sigs - SIGS:
            raise ValueError(f'Unknown signature(s): {", ".join(sorted(bad))}')

        return sigs

    def _kernels(self, dtype, dsize, kname, sigs, **kwargs):
        basemeta = self.basemeta

        # Common template arguments
        baseargs = self._base_template_args(dtype, kname)

        # Incrementally generate and render the kernels
        gen = self._kernel_generators(dtype, dsize, sigs=sigs, **kwargs)
        try:
            resp = None
            while True:
                # Generate the next kernel in the sequence
                name, exargs, exmeta = gen.send(resp)

                # Never hand back a kernel the caller can not invoke
                if sig_of(exmeta) not in sigs:
                    resp = None
                    continue

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
            'dtype': dtype, 'sdtype': dtype, 'kname': kname,
            'A': self.A, 'beta': self.beta, 'width': 1,
            'm': self.m, 'n': self.n, 'k': self.k,
            'ldb': self.ldb, 'ldc': self.ldc,
            'afix': self.afix, 'alix': self.alix, 'bix': self.bix,
            'dot': _dot, 'partition': _partition, 'chunk': _chunk
        }

    def launch_config(self, meta, n):
        # Resolve the launch description of a kernel for a given n
        if (launch := meta.get('launch')) is None:
            raise ValueError('Kernel has a fixed launch geometry')

        cfg = {}

        for key, axes in launch.items():
            cfg[key] = tuple(self._launch_axis(ax, n) for ax in axes)

        return cfg

    def _launch_axis(self, axis, n):
        match axis:
            case int():
                return axis
            case {'div': div, 'mul': mul, **rest} if not rest:
                return -(-n // div)*mul
            case {'div': div, **rest} if not rest:
                return -(-n // div)
            case _:
                raise ValueError(f'Invalid launch axis: {axis}')

    def _launch_description(self, meta):
        return {}

    def _process_meta(self, meta):
        # Kernels which work out a geometry of their own describe nothing
        if 'grid' in meta:
            meta['launch'] = {}
        else:
            meta['launch'] = self._launch_description(meta)

        # With n baked in the geometry is fixed, so resolve and drop it
        if self.n is not None:
            meta |= self.launch_config(meta, self.n)
            del meta['launch']

        sig = meta.setdefault('sig', SIG_BC)
        args = _SIG_ARGS[sig]['baked' if self.n is not None else 'runtime']

        if args is None:
            raise ValueError(f'Signature {sig} needs n to be baked in')

        meta['args'] = args

        operands = meta.setdefault('operands', {})

        # Keep the packer private, handing back a token which names it
        if (packer := meta.pop('_packer', None)) is not None:
            token = (self._uuid, len(self._packers))
            self._packers[token] = packer

            abuf = operands.setdefault('a', {})
            abuf |= {'kind': OPERAND_BUFFER, 'token': token}

        # An operand the caller must prepare has to be one it is passed
        if bad := set(operands) - set(args):
            raise ValueError('Operands described but not taken as arguments: '
                             f'{", ".join(sorted(bad))}')

        # Conversely anything beyond B, C and their dimensions needs describing
        plain = {'b', 'c', 'n', 'ldb', 'ldc'}
        prepared = {a for a in args if a not in plain}

        if missing := prepared - set(operands):
            raise ValueError('Arguments left undescribed by operands: '
                             f'{", ".join(sorted(missing))}')

    def pack_a(self, meta, a=None):
        # Lay A out as the kernel described by meta expects to find it
        token = meta.get('operands', {}).get('a', {}).get('token')

        if token is None:
            raise ValueError('This kernel does not take an a buffer')

        try:
            packer = self._packers[token]
        except (KeyError, TypeError):
            raise ValueError('Metadata is not from this generator') from None

        if a is None:
            a = self.A
        else:
            a = np.asanyarray(a)

            if a.shape != self.A.shape:
                raise ValueError(f'a must have shape {self.A.shape}')

        return packer(a)

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
            case 'is-null':
                return value is None
            case 'is-not':
                return value is not None
            case 'divisible-by':
                return value is not None and value % expected == 0
            case 'is-null-or-divisible-by':
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
