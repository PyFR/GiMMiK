# -*- coding: utf-8 -*-

attributes(global) subroutine ${funcn}(n,b,c)
    use cudafor
    implicit none	 

    integer(kind=4), device, intent(in) :: n

    ${dtype}, device, intent(in) :: b(n,${len(mat[:,0])})

    ${dtype}, device, intent(out) :: c(n,${len(mat[0,:])})

    integer(kind=4) :: i
    ${dtype} :: dotp

    i = blockDim%x*(blockIdx%x - 1) + threadIdx%x
    
    if(i .le. n) then
    % for j, jx in enumerate(mat.transpose(),start=1):
        dotp = ${' + '.join('{kx}*b(i,{k})'.format(k=k, kx=kx)
                            for k, kx in enumerate(jx, start=1) if kx != 0) or 0}
    % if beta == 0:
        c(i,${j}) = dotp
    % elif beta == 1:
        c(i,${j}) = c(i,${j}) + dotp
    % else:
        c(i,${j}) = dotp + ${beta}*c(i,${j})
    % endif
    % endfor
    endif
    
    return
end subroutine ${funcn}
