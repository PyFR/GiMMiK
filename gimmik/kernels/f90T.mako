# -*- coding: utf-8 -*-

subroutine ${funcn}(n,b,c)
    implicit none	 

    integer(kind=4), intent(in) :: n

    ${dtype}, intent(in) :: b(n,${len(mat[:,0])})

    ${dtype}, intent(${'out' if beta == 0 else 'inout'}) :: c(n,${len(mat[0,:])})

    integer(kind=4) :: i
    ${dtype} :: dotp

    !$omp simd
    do i=1,n
    % for j, jx in enumerate(mat.T,start=1):
        dotp = ${' + '.join(f'{kx}*b(i,{k})' 
                    for k, kx in enumerate(jx, start=1) if kx != 0) or 0}

    % if beta == 0:
        c(i,${j}) = dotp
    % elif beta == 1:
        c(i,${j}) = c(i,${j},i) + dotp
    % else:
        c(i,${j}) = dotp + ${beta}*c(i,${j})
    % endif
    % endfor
    enddo
    
    return
end subroutine ${funcn}
