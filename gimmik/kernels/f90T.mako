# -*- coding: utf-8 -*-

function ${funcn}(n,b,c)
    implicit none	 

    integer(kind=4), intent(in) :: n

    ${dtype}, intent(in) :: b(${len(mat[:,0])},n)

    ${dtype}, intent(out) :: c(${len(mat[0,:])},n)

    integer(kind=4) :: i
    ${dtype} :: dotp

    !$omp simd
    do i=1,n
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
    enddo
    
    return
end function ${funcn}