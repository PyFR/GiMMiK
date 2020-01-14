# -*- coding: utf-8 -*-

attributes(global) function ${funcn}(n,b,c)
    use cudafor
    implicit none	 

    integer(kind=4), device, intent(in) :: n

    ${dtype}, device, intent(in) :: b(${len(mat[:,0])},n)

    ${dtype}, device, intent(out) :: c(${len(mat[0,:])},n)

    integer(kind=4) :: i
    ${dtype} :: dotp

    i = blockDim%x*(blockIdx%x - 1) + threadIdx%x
    
    if(i .le. n) then
    % for j, jx in enumerate(mat.transpose(),start=1):
        dotp = ${' + '.join('{kx}*b({k},i)'.format(k=k, kx=kx)
                            for k, kx in enumerate(jx, start=1) if kx != 0) or 0}
    % if beta == 0:
        c(${j},i) = dotp
    % elif beta == 1:
        c(${j},i) = c(${j},i) + dotp
    % else:
        c(${j},i) = dotp + ${beta}*c(${j},i)
    % endif
    % endfor
    endif
    
    return
end function ${funcn}