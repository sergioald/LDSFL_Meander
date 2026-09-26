# Finite-window SEMIANA recurrence design

The legacy cached kernels evaluate a separate finite interior sum at each
1-based station. Keep their `jtoll` (`L`), `cA`, `cB`, exponential table,
routing, and endpoint terms. Only update the interior sum recursively.

For upstream SEMIANA1, let `J_j = min(j + L - 1, N)`,
`w = exp(-lambda * deltas)`, and

`I_j = sum(cA[q] * w**(q-j) for q = j+1, ..., J_j-1)`.

Scan stations from `N-1` down to `1`, starting with `I_N = 0`:

`I_j = w*I_(j+1) + [j+1 < J_j]*w*cA[j+1]
       - [L >= 3 and j+L-1 < N]*w**(L-1)*cA[j+L-1]`.

The last term removes the previous window's oldest interior point once it
becomes the current endpoint. When `J_j = N`, the physical boundary shortens
the window and no nonexistent point is subtracted. The full convolution stays

`cB[j]*(exp(-lambda*deltas) + lambda*deltas - 1) + I_j
 - cB[J_j]*(1 + lambda*deltas - exp(lambda*deltas))
   * exp(lambda*deltas*(j-J_j))`.

For downstream SEMIANA2, let `K_j = max(j - L + 1, 1)`,
`v = exp(lambda * deltas)`, and

`I_j = sum(cA[q] * v**(j-q) for q = K_j+1, ..., j-1)`.

Scan stations from `2` up to `N`, starting with `I_1 = 0`:

`I_j = v*I_(j-1) + [j-1 > K_j]*v*cA[j-1]
       - [L >= 3 and j-L+1 > 1]*v**(L-1)*cA[j-L+1]`.

The last term explicitly removes the point leaving the finite window. At the
physical boundary `K_j = 1`, no point is removed. The full convolution stays

`cB[K_j]*(-1 + lambda*deltas + exp(-lambda*deltas))
 * exp(lambda*deltas*(j-K_j)) + I_j
 - cB[j]*(1 + lambda*deltas - exp(lambda*deltas))`.

The upstream scan uses `abs(w) < 1` for positive-real-part eigenvalues; the
downstream scan uses `abs(v) < 1` for negative-real-part eigenvalues. Both
retain finite-window subtraction. The `L <= 2` interior is empty. All local
and sign/mode factors remain in the existing mode assembly. The alternative
is serial only; the public Numba wrapper selects it for the validated serial,
non-fastmath SL0/SL1 paths and keeps the legacy Numba response as an internal
reference.
