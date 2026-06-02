import json
import pkgutil

import numpy as np

from gimmik.base import MatMul


class PTXMatMul(MatMul):
    platform = 'ptx'
    basemeta = {
        'block': (128, 1, 1),
        'width': 1,
        'shared': 0,
        'dynamic_shared': 0
    }

    # Map Supported CC -> Minimum PTX version
    PTX_SM = {(8, 0): (7, 0), (9, 0): (8, 0), (10, 0): (8, 7), (10, 3): (8, 7),
              (12, 0): (8, 7), (12, 1): (8, 7)}

    PTX_TEMPLATE_FAMILY = {
        'cstream': 'sparse',
        'bstream': 'sparse',
        'bstream-msplit': 'sparse',
        'cstream-ksplit': 'sparse',
        'cstream-w2': 'sparse',
        'dmma-astream': 'dense',
        'dmma-asmem': 'dense',
        'dmma-steal-ws': 'dense',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config_cache = {}

    @classmethod
    def is_sparse_suitable(cls, arr, cc):
        nnz = np.count_nonzero(arr)
        nuq = len(np.unique(np.abs(arr)))
        density = nnz / arr.size
        return ((nuq <= 28) or (density <= 0.15)) and cc in cls.PTX_SM

    @classmethod
    def is_dense_suitable(cls, arr, cc):
        cc_appropriate = cc in cls.PTX_SM and cc >= (9, 0)
        return (arr.dtype == np.float64 and cc_appropriate
                and arr.shape[0] <= 128 and arr.shape[1] <= 128)

    @classmethod
    def is_suitable(cls, arr, cc):
        return cls.is_sparse_suitable(arr, cc) or cls.is_dense_suitable(arr, cc)

    def _kernel_generators(self, dtype, dsize, *, compute_capability=None,
                           smem_info=None):
        cc = compute_capability or (0, 0)
        smem_info = smem_info or (48*1024, 48*1024)
        config = self._cc_config(cc)

        for kernel_cfg in config['kernels']:
            if not self._usable_config(kernel_cfg, dtype, cc, smem_info):
                continue

            prepared = self._get_render_args(
                kernel_cfg, dtype, dsize, cc, smem_info, tuple(config['ptx'])
            )
            if prepared is not None:
                yield prepared

    def render_config(self, kernel_cfg, dtype, dsize, *, kname='gimmik_mm',
                      compute_capability=None, smem_info=None, config=None):
        cc = compute_capability or (0, 0)
        smem_info = smem_info or (48*1024, 48*1024)
        config = config or self._cc_config(cc)

        if not self._usable_config(kernel_cfg, dtype, cc, smem_info):
            return None

        prepared = self._get_render_args(
            kernel_cfg, dtype, dsize, cc, smem_info, tuple(config['ptx'])
        )
        if prepared is None:
            return None
        tplname, exargs, exmeta = prepared

        args = self._base_template_args(dtype, kname) | exargs
        meta = self.basemeta | exmeta
        meta['tplname'] = tplname
        self._process_meta(meta)
        src = self._render_kernel(dtype, tplname, args)
        return src, args, meta

    def _cc_config(self, cc):
        cc = cc or (0, 0)
        if cc not in self._config_cache:
            cfgname = f'sm{cc[0]}{cc[1]}.json'
            paths = [f'kernels/ptx/config/{cfgname}',
                    'kernels/ptx/config/default.json']

            cfg = None
            for path in paths:
                try:
                    cfgdir = pkgutil.get_data('gimmik', path)
                    cfg = json.loads(cfgdir.decode('utf-8'))
                    break
                except FileNotFoundError:
                    continue
                except json.JSONDecodeError as e:
                    raise ValueError(f'{path}: invalid JSON: {exc}') from e

            if cfg is None:
                raise ValueError('PTX default kernel config is missing')
            self._config_cache[cc] = cfg
        return self._config_cache[cc]

    def _matmul_stats(self, dtype, cc, smem_info):
        nnz = int(np.count_nonzero(self.A))
        return {
            'dtype': dtype,
            'm': self.m,
            'k': self.k,
            'n': self.n,
            'beta': self.beta,
            'beta_zero': self.beta == 0,
            'aligne': self.aligne,
            'nnz': nnz,
            'density': nnz / self.A.size,
            'unique_abs': int(len(np.unique(np.abs(self.A)))),
            'k_used': len(self.bix),
            'cc': list(cc),
            'smem_static': smem_info[0],
            'smem_dynamic': smem_info[1],
        }

    def _eval_condition(self, condition, stats):
        if 'all' in condition:
            return all(self._eval_condition(c, stats) for c in condition['all'])
        if 'any' in condition:
            return any(self._eval_condition(c, stats) for c in condition['any'])
        if 'not' in condition:
            return not self._eval_condition(condition['not'], stats)

        value = stats[condition['field']]
        op = next(k for k in condition if k != 'field')
        expected = condition[op]

        return {
            'eq': lambda: value == expected,
            'ne': lambda: value != expected,
            'lt': lambda: value is not None and value < expected,
            'lte': lambda: value is not None and value <= expected,
            'gt': lambda: value is not None and value > expected,
            'gte': lambda: value is not None and value >= expected,
            'in': lambda: value in expected,
            'is_null': lambda: value is None,
            'is_not': lambda: value is not None,
            'divisible_by': lambda: value is not None and value % expected == 0,
            'is_null_or_divisible_by': lambda: (value is None
                                                or value % expected == 0),
        }[op]()

    def _usable_config(self, kernel_cfg, dtype, cc, smem_info):
        tpl = kernel_cfg['template']
        family = self.PTX_TEMPLATE_FAMILY[tpl]

        if family == 'sparse' and not self.is_sparse_suitable(self.A, cc):
            return False
        elif (family == 'dense'
              and (self.n is None or not self.is_dense_suitable(self.A, cc))):
            return False

        condition = kernel_cfg.get('conditions')
        if condition is None:
            return True
        else:
            stats = self._matmul_stats(dtype, cc, smem_info)
            return self._eval_condition(condition, stats)

    def _get_render_args(self, kernel_cfg, dtype, dsize, cc, smem_info,
                         ptx):
        tpl = kernel_cfg['template']
        block = tuple(kernel_cfg['block'])
        width = kernel_cfg['width']
        params = kernel_cfg.get('params', {})
        base_args = {
            'ptx': ptx,
            'cc': cc,
            'smem_info': smem_info,
            'pred_emit': self._pred_emit,
            'pftype': 'f32' if dtype == 'float' else 'f64',
            'dwidth_i': dsize,
            'fzero': ('0f00000000' if dtype == 'float'
                      else '0d0000000000000000'),
            'beta_zero': self.beta == 0,
            'mbar_maxwait': '0x989680',
            'use_cpasync': cc >= (8, 0),
            'width': width,
        }
        base_meta = {
            'block': block,
            'width': width,
            'desc': kernel_cfg['descriptor'],
        }

        if self.PTX_TEMPLATE_FAMILY[tpl] == 'sparse':
            cfg = self._sparse_args(tpl, params, block, dtype, dsize,
                                    base_args, base_meta)
        elif self.PTX_TEMPLATE_FAMILY[tpl] == 'dense':
            if tpl.endswith('ws'):
                cfg = self._dense_ws_args(kernel_cfg, params, smem_info,
                                          base_args, base_meta)
            else:
                cfg = self._dense_args(kernel_cfg, params, base_args,
                                       base_meta)
        else:
            raise ValueError(f'Unknown PTX template family for {tpl}')

        return cfg

    def _sparse_args(self, tpl, params, block, dtype, dsize, args,
                     meta):
        blockx = block[0]
        args |= {'has_zero_rows': bool(self.has_zero_rows),
                 'row_nz': [[(kx, self.A[j, kx]) for kx in range(self.k)
                     if self.A[j, kx] != 0] for j in range(self.m)],
                }

        match tpl:
            case 'cstream' | 'bstream':
                pass
            case 'bstream-msplit':
                msplit = block[1]
                bsz = params['bsz']
                args |= {'msplit': msplit, 'bsz': bsz, 'blockx': blockx}
                meta['shared'] = 2*bsz*blockx*dsize
            case 'cstream-ksplit':
                ksplit = block[1]
                csz = params['csz']
                args |= {'ksplit': ksplit, 'csz': csz, 'blockx': blockx}
                meta['shared'] = (ksplit - 1)*csz*blockx*dsize
            case _:
                args['blockx'] = blockx
        return tpl, args, meta

    def _dense_args(self, kernel_cfg, params, args, meta):
        nn = params['nn']
        warps = params['warps']
        n_per_cta = 8 * nn * warps
        if n_per_cta > self.n:
            return None

        vector_width = kernel_cfg['vector_width']
        if (vector_width == 2
                and (self.aligne is None or self.aligne % 2
                     or self.n % (8 * nn))):
            return None

        block_steal = bool(params.get('block_stealing', False))
        setup = self._dense_common(nn, warps, block_steal)
        tpl = f"{kernel_cfg['template']}-v{vector_width}"
        args |= setup
        meta['grid'] = (-(-self.n // n_per_cta), 1, 1)

        return tpl, args, meta

    def _dense_common(self, nn, warps_per_cta, block_steal):
        a = self.A
        m, k = a.shape
        m_tiles = (m + 7) // 8
        k_tiles = (k + 3) // 4
        k_rem = k % 4

        # A in DMMA-fragment layout: lane l -> A[mt*8 + l//4][kt*4 + l%4]
        # i.e. an (m_tiles, k_tiles) grid of row-major 8x4 tiles, packed as
        # uint64
        a_pad = np.zeros((m_tiles*8, k_tiles*4))
        a_pad[:m, :k] = a
        tiles = a_pad.reshape(m_tiles, 8, k_tiles, 4).swapaxes(1, 2)
        a_u64 = [f'0x{u:016x}' for u in tiles.view(np.uint64).ravel()]

        n_per_warp = 8 * nn
        n_per_cta  = warps_per_cta * n_per_warp

        # Predicate-elision flags
        n_col_aligned = (self.n is not None and self.n % n_per_warp == 0)
        def pm_runtime(mt):
            return (mt + 1) * 8 > m

        return {
            'warps_per_cta': warps_per_cta,
            'nn': nn,
            'm_tiles': m_tiles,
            'k_tiles': k_tiles,
            'k_rem': k_rem,
            'm_pad': m_tiles * 8,
            'k_pad': k_tiles * 4,
            'a_u64': a_u64,
            'n_per_warp': n_per_warp,
            'n_per_cta': n_per_cta,
            'frag_stride_bytes': 32 * 8,
            'b_kiter_stride': 4 * (self.ldb or 0) * 8,
            'b_ntile_stride': 8 * 8,
            'c_mtile_stride': 8 * (self.ldc or 0) * 8,
            'c_ntile_stride': 8 * 8,
            'n_col_aligned': n_col_aligned,
            'pm_runtime': pm_runtime,
            'block_stealing': block_steal,
        }

    def _dense_ws_args(self, kernel_cfg, params, smem_info, args, meta):
        dynamic_max = smem_info[1]
        nn = params['nn']
        warp_map = kernel_cfg['warp_map']
        n_comp_warps = warp_map['compute_count']
        n_per_cta = 8 * nn * n_comp_warps
        if n_per_cta > self.n:
            return None

        setup = self._dense_common(nn, n_comp_warps, True)

        # Warp Specialism Setup
        b_tile_bytes = setup['k_pad'] * n_per_cta * 8
        c_tile_bytes = setup['m_pad'] * n_per_cta * 8
        a_bytes = setup['m_tiles'] * setup['k_tiles'] * 32 * 8

        regions = [('b1', b_tile_bytes), ('b2', b_tile_bytes),
                   ('c', c_tile_bytes), ('a', a_bytes), ('wid', 16)]
        mbars = ('tma', 'bready', 'cready', 'cstored',
                 'steal', 'wid_new', 'wid_used')
        offsets, dynm_total_bytes = self._dsmem_alloc(regions, mbars)
        ws_setup = {
            'n_comp_warps': n_comp_warps,
            'blockx_total': 32 * (n_comp_warps + 2),
            'prod_warp': warp_map['producer'],
            'steal_warp': warp_map['stealer'],
            'comp_threads': 32 * n_comp_warps,
            'b_tile_bytes': b_tile_bytes,
            'c_mtile_smem_stride': 8 * n_per_cta * 8,
            'c_ntile_smem_stride': 8 * 8,
            'dynm_total_bytes': dynm_total_bytes,
        }

        if ws_setup['dynm_total_bytes'] > dynamic_max:
            return None

        args |= setup | ws_setup | offsets
        meta |= {
            'grid': (-(-self.n // n_per_cta), 1, 1),
            'ws_b_tile': (n_per_cta, setup['k_pad']),
            'dynamic_shared': ws_setup['dynm_total_bytes'],
        }
        if self.beta != 0:
            meta['ws_out_tile'] = (n_per_cta, setup['m_pad'])
        return kernel_cfg['template'], args, meta

    @staticmethod
    def _dsmem_alloc(regions, mbars, align=16):
        out, off = {}, 0
        for name, size in regions:
            off = (off + align - 1) & ~(align - 1)
            out[f'{name}_off'] = off
            off += size
        for name in mbars:
            out[f'{name}_mbar_off'] = off
            off += 8
        total = (off + align - 1) & ~(align - 1)
        return out, total

    @staticmethod
    def _pred_emit(instr, *preds, pred_reg=None, indent=' ' * 8):
        actual = [p for p in preds if p is not None]
        if not actual:
            return instr
        if len(actual) == 1:
            return f'@{actual[0]} {instr}'
        if pred_reg is None:
            raise ValueError('pred_reg required when combining multiple '
                             'predicates')
        lines = [f'.reg .pred {pred_reg};',
                 f'and.pred {pred_reg}, {actual[0]}, {actual[1]};']
        for p in actual[2:]:
            lines.append(f'and.pred {pred_reg}, {pred_reg}, {p};')
        lines.append(f'@{pred_reg} {instr}')
        return f'\n{indent}'.join(lines)

    def _process_meta(self, meta):
        if self.n is not None and 'grid' not in meta:
            div = meta['block'][0]*meta['width']
            meta['grid'] = (-(-self.n // div), 1, 1)
