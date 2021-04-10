# Arguments.py

# Arguments
_discount_factor = 1.0
_alpha = 0.01
# _epsilon should not be very small, or Sarsa will choose the first safe path, which is more like a deterministic policy
# _epsilon should also not be very large, or the result will be hard to converge
_epsilon = 0.4
# At least iterate 300000 rounds, in order to guarantee the stability of the algorithm
basic_rounds = 300000
