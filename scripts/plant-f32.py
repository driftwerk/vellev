import sys
p='crates/vellev-index/src/vector.rs'; s=open(p).read()
which=sys.argv[1]
if which=='fma':
    old='''            let (sums, whole) = lanes(a, b, |x, y| _mm256_mul_ps(x, y));
            let tail: f32 = a[whole..]'''
    new='''            let (sums, whole) = fused(a, b);
            let tail: f32 = a[whole..]'''
    assert s.count(old)==1; s=s.replace(old,new,1)
    s=s.replace('''        #[target_feature(enable = "avx2")]
        pub(super) fn dot(''','''        #[target_feature(enable = "avx2,fma")]
        fn fused(a: &[f32], b: &[f32]) -> ([f32; LANES], usize) {
            let whole = a.len().min(b.len()) / LANES * LANES;
            let (mut low, mut high) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            let mut at = 0;
            while at < whole {
                unsafe {
                    low = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(at)), _mm256_loadu_ps(b.as_ptr().add(at)), low);
                    high = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(at + 8)), _mm256_loadu_ps(b.as_ptr().add(at + 8)), high);
                }
                at += LANES;
            }
            let mut sums = [0.0f32; LANES];
            unsafe {
                _mm256_storeu_ps(sums.as_mut_ptr(), low);
                _mm256_storeu_ps(sums.as_mut_ptr().add(8), high);
            }
            (sums, whole)
        }

        #[target_feature(enable = "avx2,fma")]
        pub(super) fn dot(''',1)
elif which=='order':
    old='''            (sums, whole)
        }

        #[target_feature(enable = "avx2")]
        pub(super) fn dot('''
    new='''            let mut folded = [0.0f32; LANES];
            for lane in 0..8 {
                folded[lane] = sums[lane] + sums[lane + 8];
            }
            (folded, whole)
        }

        #[target_feature(enable = "avx2")]
        pub(super) fn dot('''
    assert s.count(old)==1; s=s.replace(old,new,1)
open(p,'w').write(s)
