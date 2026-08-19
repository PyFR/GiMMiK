import numpy as np

from gimmik.base import (SIG_BC, SIG_BDESC_C, SIG_BDESC_CDESC, MatMul,
                         tensormap_spec)


class PTXMatMul(MatMul):
    platform = 'ptx'
    _float_suffix = ''

    sigs = frozenset({SIG_BC, SIG_BDESC_C, SIG_BDESC_CDESC})
    basemeta = {
        'block': (128, 1, 1),
        'width': 1,
        'shared': 0
    }

    # Map explicitly supported CC to minimum PTX version
    PTX_SM = {(8, 0): (7, 0), (9, 0): (8, 6), (10, 0): (8, 7), (10, 3): (8, 7),
              (12, 0): (8, 7), (12, 1): (8, 7)}

    FZERO = {'float': '0f00000000', 'double': '0d0000000000000000'}
    PFTYPE = {'float': 'f32', 'double': 'f64'}
    NPTYPE = {'float': np.float32, 'double': np.float64}

    def _sparse_viable(self, cc):
        # True when the sparse kernels can pay off on the target
        return self._unrolled_viable() and cc >= (7, 0)

    def _dense_viable(self, cc):
        # True when the target and operator admit the dense DMMA kernels
        cc_appropriate = cc in self.PTX_SM and cc >= (9, 0)

        # These templates implement a beta of zero and of one only
        return (self.A.dtype == np.float64 and cc_appropriate and
                self.beta in (0, 1) and self.m <= 128 and self.k <= 128)

    def _kernel_generators(self, dtype, dsize, *, sigs,
                           compute_capability=None, smem_max=None):
        cc = compute_capability or (0, 0)

        if not self._sparse_viable(cc) and not self._dense_viable(cc):
            return

        config = self._platform_config(dtype, cc)

        # When we know the PTX version but there isn't an SM specific config,
        # we can overide the default PTX version
        if cc in self.PTX_SM:
            target_cc = cc
            ptx = self.PTX_SM[cc]
        else:
            target_cc = tuple(config['cc'])
            ptx = tuple(config['ptx'])

        cfgs = config['kernels']
        cfg = [k for k in cfgs if self._usable_config(k, dtype, cc)]

        for k in cfg:
            prepared = self._get_render_args(k, dtype, dsize, target_cc, ptx,
                                             smem_max)
            if prepared:
                yield prepared

    def _get_render_args(self, kernel_cfg, dtype, dsize, cc, ptx, smem_max):
        tpl = kernel_cfg['template']
        family = kernel_cfg['family']
        block = tuple(kernel_cfg['block'])
        width = kernel_cfg['width']
        params = kernel_cfg.get('params', {})
        base_args = {
            'ptx': ptx,
            'cc': cc,
            'pred_emit': self._pred_emit,
            'pftype': self.PFTYPE[dtype],
            'dwidth_i': dsize,
            'fzero': self.FZERO[dtype],
            'beta_zero': self.beta == 0,
            'mbar_maxwait': hex(10000000),
            'use_cpasync': cc >= (8, 0),
            'width': width,
            'reg_list': self._reg_list,
        }
        base_meta = {
            'block': block,
            'width': width,
            'variant': kernel_cfg['variant'],
        }

        match family:
            case 'sparse':
                cfg = self._sparse_args(tpl, params, block, dtype, dsize,
                                        base_args, base_meta)
            case 'dense':
                cfg = self._dense_args(kernel_cfg, params, cc, dtype, dsize,
                                       base_args, base_meta)
            case 'dense-ws':
                cfg = self._dense_ws_args(kernel_cfg, params, cc, smem_max,
                                          dtype, dsize, base_args, base_meta)
            case _:
                raise ValueError(f'Unknown PTX template family for {tpl}')

        return cfg

    def _sparse_args(self, tpl, params, block, dtype, dsize, args, meta):
        blockx = block[0]
        args |= {
            'has_zero_rows': bool(self.has_zero_rows),
            'row_nz': [[(c, r[c]) for c in np.nonzero(r)[0]] for r in self.A],
        }

        match tpl:
            case 'cstream' | 'bstream':
                pass
            case 'bstream-msplit' | 'bstream-msplit-v2':
                bsz = params['bsz']
                args |= {'msplit': block[1], 'bsz': bsz, 'blockx': blockx,
                         'preload_c': bool(params.get('preload-c', False))}
                meta['shared'] = 2*bsz*blockx*dsize*args['width']
            case 'cstream-ksplit' | 'cstream-ksplit-v2':
                csz = params['csz']
                args |= {'ksplit': block[1], 'csz': csz, 'blockx': blockx,
                         'preload_c': bool(params.get('preload-c', False))}
                meta['shared'] = (block[1] - 1)*csz*blockx*dsize*args['width']
            case _:
                args['blockx'] = blockx
        return tpl, args, meta

    def _b_tensormap(self, dtype, dsize, n_per_cta, k_pad):
        # Tensor map for the B panel a descriptor kernel streams in
        return tensormap_spec('b', self.NPTYPE[dtype], (n_per_cta, k_pad),
                              (self.n, self.k), (self.ldb*dsize,))

    def _c_tensormap(self, dtype, dsize, n_per_cta, m_pad):
        # Tensor map for the C panel a descriptor kernel stages out
        return tensormap_spec('c', self.NPTYPE[dtype], (n_per_cta, m_pad),
                              (self.n, self.m), (self.ldc*dsize,))

    def _dense_args(self, kernel_cfg, params, cc, dtype, dsize, args, meta):
        tpl, tile = kernel_cfg['template'], kernel_cfg['tile']
        nn, warps, width = params['nn'], params['warps'], kernel_cfg['width']

        setup = self._dense_common(nn, warps, tile, cc, width)
        if setup is None:
            return None

        args |= setup
        if tpl.startswith('dmma-asmem'):
            stealing = bool(params['block-stealing'])
            args |= {'a_copy_threads': 32*warps, 'block_stealing': stealing}

            # Shared memory: A, plus a barrier and mailbox when stealing
            meta['shared'] = setup['a_elems']*args['dwidth_i'] + 24*stealing
        meta['grid'] = (-(-self.n // setup['n_per_cta']), 1, 1)

        msplit = params.get('msplit')
        if msplit is None:
            return tpl, args, meta

        n_per_cta = setup['n_per_cta']
        k_pad = setup['k_tiles']*setup['tile_k']
        b_tile_bytes = k_pad*n_per_cta*args['dwidth_i']

        args |= {
            'msplit': msplit,
            'b_tile_bytes': b_tile_bytes,
            'b_smem_kiter_stride': setup['tile_k']*n_per_cta*args['dwidth_i'],
            'b_smem_kgroup_stride': 4*n_per_cta*args['dwidth_i'],
            'b_smem_ntile_stride': setup['tile_n']*args['dwidth_i'],
            'blockx_total': 32*warps*msplit,
        }

        # Shared memory: the staged B tile plus the barrier guarding it
        meta['shared'] = b_tile_bytes + 8

        # These stream B through a tensor map but store C through a pointer
        meta['sig'] = SIG_BDESC_C
        meta['operands'] = {
            'b_desc': self._b_tensormap(dtype, dsize, n_per_cta, k_pad)
        }

        return tpl, args, meta

    def _dense_common(self, nn, warps_per_cta, tile, cc, width=None):
        tile_m, tile_n, tile_k = tile['m'], tile['n'], tile['k']
        ptx_shape = f'm{tile_m}n{tile_n}k{tile_k}'

        m_groups, k_groups = tile_m // 8, tile_k // 4
        a_regs = m_groups*k_groups
        b_regs = k_groups
        c_regs = 2*m_groups

        a = self.A
        m, k = a.shape
        m_tiles, k_tiles = -(-m // tile_m), -(-k // tile_k)
        k_rem = k % tile_k
        n_per_warp = tile_n*nn
        n_per_cta = warps_per_cta*n_per_warp

        if n_per_cta > self.n:
            return None

        if (width == 2
                and (self.aligne is None or self.aligne % 2
                     or self.n % n_per_warp)):
            return None

        # A in DMMA-fragment layout, packed in PTX A-operand register order.
        # This will handle 8x8x4 as well as additional sm90 sizes.
        a_pad = np.zeros((m_tiles*tile_m, k_tiles*tile_k), dtype=a.dtype)
        a_pad[:m, :k] = a
        tile_shape = m_tiles, m_groups, 8, k_tiles, k_groups, 4
        tile_order = 0, 3, 4, 1, 2, 5
        a_frags = (a_pad.reshape(*tile_shape).transpose(*tile_order)
                   .reshape(m_tiles*k_tiles, tile_m*tile_k))

        # All-zero (mt, kt) tiles contribute nothing; elide them from the
        # packed A array and let templates skip their loads and MMAs.
        tile_nz = np.any(a_frags != 0, axis=1)
        if not tile_nz.any():
            tile_nz[0] = True
        cidx = np.cumsum(tile_nz) - 1
        a_tile_nz = [[bool(tile_nz[mt*k_tiles + kt])
                      for kt in range(k_tiles)] for mt in range(m_tiles)]
        a_tile_idx = [[int(cidx[mt*k_tiles + kt])
                       for kt in range(k_tiles)] for mt in range(m_tiles)]

        a_tiles = a_frags[tile_nz].ravel()
        a_u64 = [f'0x{u:016x}' for u in a_tiles.view(np.uint64)]

        # Predicate-elision flags
        n_col_aligned = (self.n is not None and self.n % n_per_warp == 0)
        def pm_runtime(mt, mg=0):
            return mt*tile_m + 8*(mg + 1) > m

        return {
            'tile_m': tile_m,
            'tile_n': tile_n,
            'tile_k': tile_k,
            'ptx_mma_shape': ptx_shape,
            'm_groups': m_groups,
            'k_groups': k_groups,
            'a_regs': a_regs,
            'b_regs': b_regs,
            'c_regs': c_regs,
            'a_elems': a_tiles.size,
            'a_tile_nz': a_tile_nz,
            'a_tile_idx': a_tile_idx,
            'nn': nn,
            'm_tiles': m_tiles,
            'k_tiles': k_tiles,
            'k_rem': k_rem,
            'a_u64': a_u64,
            'n_per_warp': n_per_warp,
            'n_per_cta': n_per_cta,
            'frag_stride_bytes': 8*tile_m*tile_k,
            'b_kiter_stride': 8*tile_k*(self.ldb or 0),
            'b_kgroup_stride': 32*(self.ldb or 0),
            'b_ntile_stride': 8*tile_n,
            'c_mtile_stride': 8*tile_m*(self.ldc or 0),
            'c_mgroup_stride': 64*(self.ldc or 0),
            'c_ntile_stride': 8*tile_n,
            'n_col_aligned': n_col_aligned,
            'pm_runtime': pm_runtime,
        }

    def _dense_ws_args(self, kernel_cfg, params, cc, smem_max, dtype, dsize,
                       args, meta):
        tpl = kernel_cfg['template']
        nn = params['nn']
        tile = kernel_cfg['tile']
        warp_map = kernel_cfg['warp-map']

        match tpl:
            case 'dmma-steal-ws':
                if (tile['m'], tile['n'], tile['k']) != (8, 8, 4):
                    return None
                service_warps = 2
            case 'dmma-stride-ws':
                service_warps = 1
            case _:
                raise ValueError('Unknown dense warp-specialized template '
                                 f'{tpl}')

        n_comp_warps = warp_map['compute-count']
        setup = self._dense_common(nn, n_comp_warps, tile, cc)
        if setup is None:
            return None

        n_per_cta = setup['n_per_cta']
        m_pad = setup['m_tiles']*setup['tile_m']
        k_pad = setup['k_tiles']*setup['tile_k']
        b_tile_bytes = 8*k_pad*n_per_cta
        c_tile_bytes = 8*m_pad*n_per_cta
        a_bytes = 8*setup['a_elems']

        # Shared memory: both B stages, A, and the mailboxes and barriers
        match tpl:
            case 'dmma-steal-ws':
                smem = 2*b_tile_bytes + a_bytes + 32 + 13*8
            case 'dmma-stride-ws':
                smem = 2*b_tile_bytes + a_bytes + 5*8

        if self.beta != 0:
            smem += c_tile_bytes

        if smem_max is not None and smem > smem_max:
            return None

        ws_setup = {
            'n_comp_warps': n_comp_warps,
            'blockx_total': 32*(n_comp_warps + service_warps),
            'prod_warp': warp_map['producer'],
            'comp_threads': 32*n_comp_warps,
            'b_tile_bytes': b_tile_bytes,
            'c_mtile_smem_stride': 8*setup['tile_m']*n_per_cta,
            'c_mgroup_smem_stride': 64*n_per_cta,
            'c_ntile_smem_stride': 8*setup['tile_n'],
        }

        match tpl:
            case 'dmma-steal-ws':
                grid = (-(-self.n // n_per_cta), 1, 1)
                ws_setup['steal_warp'] = warp_map['stealer']
            case 'dmma-stride-ws':
                stride_iters = params['iters']
                work_blocks = -(-self.n // n_per_cta)
                grid_stride = -(-work_blocks // stride_iters)
                grid = (grid_stride, 1, 1)
                ws_setup |= {
                    'stride_iters': stride_iters,
                    'grid_stride': grid_stride,
                    'work_blocks': work_blocks,
                }

        args |= setup | ws_setup

        # With beta=0 these store straight to global, so C is a plain pointer
        operands = {
            'b_desc': self._b_tensormap(dtype, dsize, n_per_cta, k_pad)
        }

        if self.beta == 0:
            sig = SIG_BDESC_C
        else:
            sig = SIG_BDESC_CDESC
            operands['c_desc'] = self._c_tensormap(dtype, dsize, n_per_cta,
                                                   m_pad)

        meta |= {'grid': grid, 'shared': smem, 'sig': sig,
                 'operands': operands}

        return tpl, args, meta

    def _usable_config(self, kernel_cfg, dtype, cc):
        family = kernel_cfg['family']

        if family == 'sparse':
            if not self._sparse_viable(cc):
                return False
        elif family in {'dense', 'dense-ws'}:
            dense = self._dense_viable(cc)
            if dtype != 'double' or self.n is None or not dense:
                return False

        condition = kernel_cfg.get('conditions')
        if condition is None:
            return True
        else:
            stats = self._matmul_stats(dtype, cc)
            return self._eval_condition(condition, stats)

    def _platform_config(self, dtype, cc):
        # Fall back on the default config when the SM has none of its own
        if cc:
            try:
                return self._get_config(f'sm{cc[0]}{cc[1]}-{dtype}')
            except FileNotFoundError:
                pass

        return self._get_config(f'default-{dtype}')

    def _matmul_stats(self, dtype, cc):
        nnz = np.count_nonzero(self.A)
        return {
            'dtype': dtype,
            'm': self.m,
            'k': self.k,
            'n': self.n,
            'beta': self.beta,
            'beta-zero': self.beta == 0,
            'aligne': self.aligne,
            'nnz': nnz,
            'density': nnz / self.A.size,
            'unique-abs': len(np.unique(np.abs(self.A))),
            'k-used': len(self.bix),
            'cc': list(cc),
        }

    @staticmethod
    def _reg_list(prefix, n):
        regs = ', '.join(f'{prefix}_{i}' for i in range(n))
        return f'{{{regs}}}'

    @staticmethod
    def _pred_emit(instr, *preds, pred_reg=None, indent=8*' '):
        # Handle whether an instruction needs a predicate or not
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

    def _launch_description(self, meta):
        div = meta['block'][0]*meta['width']

        return {'grid': ({'div': div}, 1, 1), 'block': meta['block']}
