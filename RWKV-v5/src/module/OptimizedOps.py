### ---
# Optimized operations, which is commonly used in the model
### ---

from .CoreDependencies import TCompileMax, JITFunction

# Modified LERP operation, which is used to compute the 
# time mixing and channel mixing components. 
# 
# This is slightly different from the standard LERP operation
# due to the presence of the start_mul parameter.
@TCompileMax
@JITFunction
def modified_lerp(start_mul, start, weight):
    return start_mul * start + weight * (1 - start)
