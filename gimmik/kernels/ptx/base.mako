.version 8.7
.target sm_${cc[0]}${cc[1]}${"a" if cc[0] >= 9 else ""}
.address_size 64
${next.body()}
