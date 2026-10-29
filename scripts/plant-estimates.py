import sys
p='crates/vellev-index/src/clustered.rs'; s=open(p).read()
w=sys.argv[1]
if w=='fma':
    old="let set = _mm256_add_ps(_mm256_mul_ps(lower, o), _mm256_mul_ps(delta, w));"
    new="let set = _mm256_fmadd_ps(lower, o, _mm256_mul_ps(delta, w));"
    s=s.replace('''        #[target_feature(enable = "avx2")]
        pub(super) fn estimate(''','''        #[target_feature(enable = "avx2,fma")]
        pub(super) fn estimate(''',1)
elif w=='order':
    old='''                    Distance::L2 => _mm256_sub_ps(
                        _mm256_add_ps(_mm256_mul_ps(n, n), _mm256_set1_ps(probe.norm_squared)),
                        _mm256_mul_ps(_mm256_mul_ps(two, n), inner),
                    ),'''
    new='''                    Distance::L2 => _mm256_add_ps(
                        _mm256_mul_ps(n, n),
                        _mm256_sub_ps(_mm256_set1_ps(probe.norm_squared), _mm256_mul_ps(_mm256_mul_ps(two, n), inner)),
                    ),'''
elif w=='reciprocal':
    old='''                let inner = _mm256_div_ps(
                    _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(two, set), sum), scale),
                    a,
                );'''
    new='''                let inner = _mm256_mul_ps(
                    _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(two, set), sum), scale),
                    _mm256_div_ps(_mm256_set1_ps(1.0), a),
                );'''
assert s.count(old)==1, w; s=s.replace(old,new,1); open(p,'w').write(s)
