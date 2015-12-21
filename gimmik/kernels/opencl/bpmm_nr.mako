## -*- coding: ascii -*-
##
## <%page args="dtype, beta, subterms, products" />
##
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void
gimmik_mm(__global const ${dtype}* restrict b,
          __global ${dtype}* restrict c,
          const int width,
          const int bstride,
          const int cstride)
{
    int index = get_global_id(0);
    if (index < width)
    {
        __global const ${dtype} *b_local = b + index;
        __global ${dtype} *c_local = c + index;

% for subterm in subterms:
        const ${dtype} subterm_${loop.index} = \
    % for term in subterm:
        % if not loop.last:
b_local[${term} * bstride] + \
        % else:
b_local[${term} * bstride];
        % endif
    % endfor
% endfor

% for product in products:
        c_local[${loop.index} * cstride] = \
    % if beta != 0.0:
${repr(beta)}${'f' if (dtype == 'float') else ''} * \
c_local[${loop.index} * cstride] + \
    % endif
    % if not product:
0;
<% continue %>
    % endif
    % for constant, subterm in product.items():
        % if loop.index < len(product) - 1:
            % for s in subterm:
${repr(constant)}${'f' if (dtype == 'float') else ''} * \
subterm_${subterms.index(tuple([s]))} + \
            % endfor
        % else:
            % for s in subterm:
                % if not loop.last:
${repr(constant)}${'f' if (dtype == 'float') else ''} * \
subterm_${subterms.index(tuple([s]))} + \
                % else:
${repr(constant)}${'f' if (dtype == 'float') else ''} * \
subterm_${subterms.index(tuple([s]))};
                % endif
            % endfor
        % endif
    % endfor
% endfor
    }
}
