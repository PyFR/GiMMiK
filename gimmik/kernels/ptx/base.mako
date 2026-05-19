.version 8.6
.target sm_${cc[0]}${cc[1]}${'a' if cc[0] >= 9 else ''}
.address_size 64
${next.body()}
