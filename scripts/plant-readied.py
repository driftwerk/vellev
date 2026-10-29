import sys
p='crates/vellev-index/src/clustered.rs'; s=open(p).read()
which=sys.argv[1]
if which=='reciprocal':
    old="let x = _mm256_div_ps(_mm256_sub_ps(value, lowest), step);"
    new="let x = _mm256_mul_ps(_mm256_sub_ps(value, lowest), _mm256_div_ps(_mm256_set1_ps(1.0), step));"
elif which=='nan':
    old="if _mm256_movemask_ps(odd) != 0 || lower == 0.0 || upper == 0.0 {"
    new="if lower == 0.0 || upper == 0.0 {"
elif which=='nearest':
    old="let whole = _mm256_cvttps_epi32(x);"
    new="let whole = _mm256_cvtps_epi32(x);"
assert s.count(old)==1, which; s=s.replace(old,new,1); open(p,'w').write(s)
