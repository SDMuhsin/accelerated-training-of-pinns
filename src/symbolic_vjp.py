"""
Symbolic VJP Engine — Tracing-based reverse-mode AD for PDE residuals.

Traces forward PDE computation via TracedVar operator overloading,
records operations on a tape, replays in reverse applying VJP rules,
and emits optimized PyTorch backward functions.
"""

import torch

# Physics constant (must match lid_benchmark.py)
_NU_LAMINAR = 0.001  # U_lid / Re = 1.0 / 1000.0


# =============================================================================
# TracedVar — thin wrapper that records operations on a tape
# =============================================================================
class TracedVar:
    """Variable that records operations instead of computing them."""

    _counter = 0

    def __init__(self, name, tape, is_const=False):
        self.name = name
        self.tape = tape
        self.is_const = is_const

    @classmethod
    def _fresh(cls, tape):
        n = f"_t{cls._counter}"
        cls._counter += 1
        return cls(n, tape)

    # --- matmul: D @ x ---
    def __matmul__(self, other):
        r = TracedVar._fresh(self.tape)
        self.tape.append(('matmul', self, other, r))
        return r

    # --- element-wise multiply ---
    def __mul__(self, other):
        if isinstance(other, TracedVar):
            r = TracedVar._fresh(self.tape)
            if self.is_const:
                self.tape.append(('const_mul', self, other, r))
            elif other.is_const:
                self.tape.append(('const_mul', other, self, r))
            else:
                self.tape.append(('mul', self, other, r))
            return r
        # scalar * traced
        r = TracedVar._fresh(self.tape)
        self.tape.append(('smul', other, self, r))  # (scalar, traced, result)
        return r

    def __rmul__(self, other):
        # other * self where other is scalar/float
        if isinstance(other, TracedVar):
            return other.__mul__(self)
        r = TracedVar._fresh(self.tape)
        self.tape.append(('smul', other, self, r))
        return r

    # --- addition ---
    def __add__(self, other):
        if isinstance(other, TracedVar):
            r = TracedVar._fresh(self.tape)
            self.tape.append(('add', self, other, r))
            return r
        # traced + scalar/constant
        r = TracedVar._fresh(self.tape)
        self.tape.append(('cadd', self, other, r))
        return r

    def __radd__(self, other):
        if isinstance(other, TracedVar):
            return other.__add__(self)
        # scalar + traced  →  same as traced + scalar
        r = TracedVar._fresh(self.tape)
        self.tape.append(('cadd', self, other, r))
        return r

    # --- subtraction ---
    def __sub__(self, other):
        if isinstance(other, TracedVar):
            r = TracedVar._fresh(self.tape)
            self.tape.append(('sub', self, other, r))
            return r
        # traced - scalar → cadd with negative
        r = TracedVar._fresh(self.tape)
        self.tape.append(('cadd', self, -other, r))
        return r

    def __rsub__(self, other):
        # other - self → neg(self) + other
        neg = self.__neg__()
        return neg.__add__(other) if not isinstance(other, TracedVar) else other.__sub__(self)

    def __neg__(self):
        r = TracedVar._fresh(self.tape)
        self.tape.append(('smul', -1.0, self, r))
        return r

    # --- power (only x**2 supported) ---
    def __pow__(self, exp):
        assert exp == 2, f"Only x**2 supported, got x**{exp}"
        r = TracedVar._fresh(self.tape)
        self.tape.append(('square', self, r))
        return r

    # --- torch function protocol for torch.sqrt, torch.sparse.mm, etc. ---
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.sqrt:
            x = args[0]
            r = TracedVar._fresh(x.tape)
            x.tape.append(('sqrt', x, r))
            return r
        if func is torch.sparse.mm:
            D, x = args[0], args[1]
            # D is a real sparse tensor (constant), x is TracedVar
            if isinstance(x, TracedVar):
                r = TracedVar._fresh(x.tape)
                x.tape.append(('sparse_matmul', D, x, r))
                return r
        raise NotImplementedError(f"TracedVar does not support {func}")


# =============================================================================
# Tracing helpers
# =============================================================================
def trace_pde_forward(compute_fn, N_all, tape, sparse=False, constants=None,
                      input_names=None):
    """Trace compute_pde_terms through TracedVar to build tape.

    Args:
        compute_fn: PDE forward function (pred, g) -> tuple of TracedVars
        N_all: grid size (can be None for tracing)
        tape: list to record operations
        sparse: if True, use sparse matmul
        constants: list of constant names for grid_data (default: ['Dx', 'Dy', 'Cs_d_sq'])
        input_names: list of input variable names (default: ['u', 'v', 'p'])

    Returns (output_vars, input_vars) where:
      - output_vars = tuple of TracedVars (one per PDE equation)
      - input_vars = dict mapping name -> TracedVar
    """
    if constants is None:
        constants = ['Dx', 'Dy', 'Cs_d_sq']
    if input_names is None:
        input_names = ['u', 'v', 'p']

    TracedVar._counter = 0

    # Create traced input variables dynamically
    input_vars = {}
    for name in input_names:
        input_vars[name] = TracedVar(name, tape)

    # Create a fake pred object that supports slicing
    class FakePred:
        def __getitem__(self, key):
            if isinstance(key, tuple) and len(key) == 2:
                _, col = key
                for i, name in enumerate(input_names):
                    if col == slice(i, i + 1):
                        return input_vars[name]
            raise KeyError(f"Unsupported slice: {key}")

    # Create traced grid_data with constant TracedVars for matrices
    g = {'N_all': N_all}
    for name in constants:
        g[name] = TracedVar(name, tape, is_const=True)

    pred = FakePred()
    outputs = compute_fn(pred, g)

    return outputs, input_vars


# =============================================================================
# VJP Rules
# =============================================================================
def _vjp_matmul(adj, A, x, _out):
    """y = A @ x  →  adj_x = A^T @ adj  (A is constant)"""
    return [(x, f"g['DxT'] @ {adj}" if 'Dx' in A else f"g['DyT'] @ {adj}")]

def _vjp_sparse_matmul(adj, D, x, _out):
    """y = sparse.mm(D, x)  →  adj_x = sparse.mm(D.t(), adj)"""
    return [(x, f"torch.sparse.mm(g['{D}'].t(), {adj})")]

def _vjp_mul(adj, a, b, _out):
    """y = a * b  →  adj_a = adj*b, adj_b = adj*a"""
    return [(a, f"{adj} * {b}"), (b, f"{adj} * {a}")]

def _vjp_const_mul(adj, C, x, _out):
    """y = C * x where C is constant  →  adj_x = C * adj"""
    return [(x, f"g['{C}'] * {adj}" if C != 'Cs_d_sq' else f"g['Cs_d_sq'] * {adj}")]

def _vjp_add(adj, a, b, _out):
    """y = a + b  →  adj_a = adj, adj_b = adj"""
    return [(a, adj), (b, adj)]

def _vjp_sub(adj, a, b, _out):
    """y = a - b  →  adj_a = adj, adj_b = -adj"""
    return [(a, adj), (b, f"-{adj}")]

def _vjp_smul(adj, c, x, _out):
    """y = c * x (scalar)  →  adj_x = c * adj"""
    return [(x, f"{c} * {adj}")]

def _vjp_square(adj, x, _out):
    """y = x**2  →  adj_x = 2*x*adj"""
    return [(x, f"2 * {x} * {adj}")]

def _vjp_sqrt(adj, x, _out):
    """y = sqrt(x)  →  adj_x = adj / (2*y)"""
    return [(x, f"{adj} / (2 * {_out})")]

def _vjp_cadd(adj, x, _c, _out):
    """y = x + constant  →  adj_x = adj"""
    return [(x, adj)]


# =============================================================================
# Backward pass — symbolic reverse-mode AD
# =============================================================================
def symbolic_backward(tape, output_vars, seed_names):
    """Walk tape in reverse, accumulate adjoint expressions as code strings.

    Args:
        tape: list of (op, *inputs, output) recorded during tracing
        output_vars: list of TracedVar outputs (continuity, mom_u, mom_v)
        seed_names: list of seed variable names ('dc', 'dmu', 'dmv')

    Returns:
        adj: dict mapping TracedVar.name → list of code-string contributions
    """
    # Map op names to VJP functions
    VJP = {
        'matmul': _build_matmul_vjp,
        'sparse_matmul': _build_sparse_matmul_vjp,
        'mul': _build_mul_vjp,
        'const_mul': _build_const_mul_vjp,
        'add': _build_add_vjp,
        'sub': _build_sub_vjp,
        'smul': _build_smul_vjp,
        'square': _build_square_vjp,
        'sqrt': _build_sqrt_vjp,
        'cadd': _build_cadd_vjp,
    }

    # adj maps var_name → list of string expressions to sum
    adj = {}
    for var, seed in zip(output_vars, seed_names):
        adj[var.name] = [seed]

    for entry in reversed(tape):
        op = entry[0]
        if op in ('matmul', 'sparse_matmul'):
            A, x, out = entry[1], entry[2], entry[3]
        elif op in ('mul', 'const_mul', 'add', 'sub'):
            a, b, out = entry[1], entry[2], entry[3]
        elif op == 'smul':
            c, x, out = entry[1], entry[2], entry[3]
        elif op == 'square':
            x, out = entry[1], entry[2]
        elif op == 'sqrt':
            x, out = entry[1], entry[2]
        elif op == 'cadd':
            x, c, out = entry[1], entry[2], entry[3]
        else:
            raise ValueError(f"Unknown op: {op}")

        if out.name not in adj:
            continue  # dead node

        # Get the adjoint expression for this output
        adj_name = f"adj_{out.name}"

        # Apply VJP rule
        contributions = VJP[op](adj_name, entry)

        for target_var, expr in contributions:
            if target_var.is_const:
                continue  # no gradient for constants
            if target_var.name not in adj:
                adj[target_var.name] = [expr]
            else:
                adj[target_var.name].append(expr)

    return adj


def _build_matmul_vjp(adj_name, entry):
    """matmul: y = A @ x → adj_x = A^T @ adj"""
    _, A, x, out = entry
    # Use precomputed transpose: 'Dx' → 'DxT', 'Dxx' → 'DxxT', etc.
    a_name = A.name
    t_name = a_name + 'T'
    return [(x, f"g['{t_name}'] @ {adj_name}")]


def _build_sparse_matmul_vjp(adj_name, entry):
    """sparse_matmul: y = sparse.mm(D, x) → adj_x = sparse.mm(D.t(), adj)"""
    _, D, x, out = entry
    # D is a real sparse tensor stored in g
    d_name = D if isinstance(D, str) else D.name  # 'Dx' or 'Dy'
    return [(x, f"torch.sparse.mm(g['{d_name}'].t(), {adj_name})")]


def _build_mul_vjp(adj_name, entry):
    """mul: y = a * b → adj_a = adj*b, adj_b = adj*a"""
    _, a, b, out = entry
    return [(a, f"{adj_name} * {b.name}"), (b, f"{adj_name} * {a.name}")]


def _build_const_mul_vjp(adj_name, entry):
    """const_mul: y = C * x (C constant tensor) → adj_x = C * adj"""
    _, C, x, out = entry
    return [(x, f"g['{C.name}'] * {adj_name}")]


def _build_add_vjp(adj_name, entry):
    """add: y = a + b → adj_a = adj, adj_b = adj"""
    _, a, b, out = entry
    return [(a, adj_name), (b, adj_name)]


def _build_sub_vjp(adj_name, entry):
    """sub: y = a - b → adj_a = adj, adj_b = -adj"""
    _, a, b, out = entry
    return [(a, adj_name), (b, f"-{adj_name}")]


def _build_smul_vjp(adj_name, entry):
    """smul: y = c * x (scalar) → adj_x = c * adj"""
    _, c, x, out = entry
    return [(x, f"{c} * {adj_name}")]


def _build_square_vjp(adj_name, entry):
    """square: y = x**2 → adj_x = 2*x*adj"""
    _, x, out = entry
    return [(x, f"2 * {x.name} * {adj_name}")]


def _build_sqrt_vjp(adj_name, entry):
    """sqrt: y = sqrt(x) → adj_x = adj / (2*y)"""
    _, x, out = entry
    return [(x, f"{adj_name} / (2 * {out.name})")]


def _build_cadd_vjp(adj_name, entry):
    """cadd: y = x + constant → adj_x = adj"""
    _, x, c, out = entry
    return [(x, adj_name)]


# =============================================================================
# Code Emitter — generates optimized PyTorch backward function
# =============================================================================
def emit_backward(tape, output_vars, seed_names, input_vars, sparse=False,
                  func_name=None, input_names=None, backend="torch",
                  external_seeds=False):
    """Generate a backward function from symbolic adjoint expressions.

    Args:
        func_name: name for the generated function (default: 'generated_analytical_grad')
        input_names: list of input variable names (default: ['u', 'v', 'p'])
        backend: "torch" (default) emits PyTorch code; "jax" emits jax.numpy code.
            The jax backend does not support sparse=True (raises NotImplementedError).
        external_seeds: if True, the emitted function accepts the adjoint seeds
            (one per output residual, named via ``seed_names``) as extra
            positional arguments instead of computing them internally as
            ``residual * 2/M * interior_mask``. This lets an outer loss (e.g.
            PhysicsNeMo's PointwiseLossNorm) own the seeding and hand
            ``∂loss/∂residual_k`` into SAGE directly.

    Returns (source_code, compiled_fn).
    """
    if func_name is None:
        func_name = 'generated_analytical_grad'
    if input_names is None:
        input_names = ['u', 'v', 'p']
    if backend not in ("torch", "jax"):
        raise ValueError(f"backend must be 'torch' or 'jax', got {backend!r}")
    if backend == "jax" and sparse:
        raise NotImplementedError(
            "sparse+jax backend is not supported (SK-PINN is torch-only)")

    adj = symbolic_backward(tape, output_vars, seed_names)

    # Find constant TracedVars used in add/sub/mul/smul/cadd ops (not matmul/
    # const_mul which already reference g['name'] directly in the emitted code).
    const_vars_in_add = set()
    for entry in tape:
        op = entry[0]
        if op in ('add', 'sub'):
            a, b = entry[1], entry[2]
            if isinstance(a, TracedVar) and a.is_const:
                const_vars_in_add.add(a.name)
            if isinstance(b, TracedVar) and b.is_const:
                const_vars_in_add.add(b.name)
        elif op == 'mul':
            a, b = entry[1], entry[2]
            if isinstance(a, TracedVar) and a.is_const:
                const_vars_in_add.add(a.name)
            if isinstance(b, TracedVar) and b.is_const:
                const_vars_in_add.add(b.name)
        elif op == 'smul':
            # smul emits 'out = c * x.name'. If x is a const TracedVar, x.name
            # must be bound in local scope.
            x = entry[2]
            if isinstance(x, TracedVar) and x.is_const:
                const_vars_in_add.add(x.name)
        elif op == 'cadd':
            x = entry[1]
            if isinstance(x, TracedVar) and x.is_const:
                const_vars_in_add.add(x.name)

    lines_v2 = []
    if external_seeds:
        # Caller supplies ∂loss/∂residual_k as positional args; the emitted
        # code uses them verbatim as adjoint seeds. No internal scale/mask.
        sig_seeds = ", " + ", ".join(seed_names)
        lines_v2.append(f"def {func_name}(pred_det, g{sig_seeds}):")
    else:
        lines_v2.append(f"def {func_name}(pred_det, g):")
    if backend == "torch":
        lines_v2.append("    import torch")
    else:
        lines_v2.append("    import jax.numpy as jnp")
    for i, name in enumerate(input_names):
        lines_v2.append(f"    {name} = pred_det[:g['N_all'], {i}:{i+1}]")
    # Extract constant TracedVars that appear in add/sub/mul ops
    for cname in sorted(const_vars_in_add):
        lines_v2.append(f"    {cname} = g['{cname}']")
    lines_v2.append("")

    # Emit ALL forward ops
    for entry in tape:
        line = _emit_forward_op(entry, sparse, backend=backend)
        if line:
            lines_v2.append(f"    {line}")

    if not external_seeds:
        lines_v2.append("")
        lines_v2.append("    # Loss scaling (adjoint seeds)")
        lines_v2.append("    M = g['M']")
        lines_v2.append("    scale = 2.0 / M")
        lines_v2.append("    mask = g['interior_mask']")

        # Map output var names to seed names
        out_names = [ov.name for ov in output_vars]
        for oname, sname in zip(out_names, seed_names):
            lines_v2.append(f"    {sname} = {oname} * scale * mask")

    lines_v2.append("")
    lines_v2.append("    # Backward pass — adjoint accumulation")

    # Emit adjoint computations in reverse tape order
    emitted_adj = set()
    for entry in reversed(tape):
        out = entry[-1]
        if out.name not in adj:
            continue
        adj_name = f"adj_{out.name}"
        if adj_name in emitted_adj:
            continue
        emitted_adj.add(adj_name)

        exprs = adj[out.name]
        if len(exprs) == 1:
            lines_v2.append(f"    {adj_name} = {exprs[0]}")
        else:
            lines_v2.append(f"    {adj_name} = {exprs[0]}")
            for e in exprs[1:]:
                lines_v2.append(f"    {adj_name} = {adj_name} + {e}")

    # Accumulate final gradients for input variables
    lines_v2.append("")
    comment_vars = ", ".join(input_names)
    lines_v2.append(f"    # Accumulate final gradients for {comment_vars}")
    zeros_like_fn = "torch.zeros_like" if backend == "torch" else "jnp.zeros_like"
    for var_name in input_names:
        if var_name in adj:
            exprs = adj[var_name]
            adj_name = f"adj_{var_name}"
            lines_v2.append(f"    {adj_name} = {exprs[0]}")
            for e in exprs[1:]:
                lines_v2.append(f"    {adj_name} = {adj_name} + {e}")
        else:
            lines_v2.append(f"    adj_{var_name} = {zeros_like_fn}({var_name})")

    lines_v2.append("")
    cat_args = ", ".join(f"adj_{name}" for name in input_names)
    if backend == "torch":
        lines_v2.append(f"    return torch.cat([{cat_args}], dim=1)")
    else:
        lines_v2.append(f"    return jnp.concatenate([{cat_args}], axis=1)")

    source = "\n".join(lines_v2)

    # Compile
    if backend == "torch":
        namespace = {'torch': torch}
    else:
        import jax.numpy as jnp  # local import so torch-only runs stay jax-free
        namespace = {'jnp': jnp}
    exec(source, namespace)
    fn = namespace[func_name]

    return source, fn


def _emit_forward_op(entry, sparse=False, backend="torch"):
    """Convert a tape entry to a forward computation line."""
    op = entry[0]

    if op == 'matmul':
        A, x, out = entry[1], entry[2], entry[3]
        return f"{out.name} = g['{A.name}'] @ {x.name}"

    elif op == 'sparse_matmul':
        if backend != "torch":
            raise NotImplementedError("sparse_matmul is only supported for backend='torch'")
        D, x, out = entry[1], entry[2], entry[3]
        d_name = D if isinstance(D, str) else D.name
        return f"{out.name} = torch.sparse.mm(g['{d_name}'], {x.name})"

    elif op == 'mul':
        a, b, out = entry[1], entry[2], entry[3]
        return f"{out.name} = {a.name} * {b.name}"

    elif op == 'const_mul':
        C, x, out = entry[1], entry[2], entry[3]
        return f"{out.name} = g['{C.name}'] * {x.name}"

    elif op == 'add':
        a, b, out = entry[1], entry[2], entry[3]
        return f"{out.name} = {a.name} + {b.name}"

    elif op == 'sub':
        a, b, out = entry[1], entry[2], entry[3]
        return f"{out.name} = {a.name} - {b.name}"

    elif op == 'smul':
        c, x, out = entry[1], entry[2], entry[3]
        return f"{out.name} = {c} * {x.name}"

    elif op == 'square':
        x, out = entry[1], entry[2]
        return f"{out.name} = {x.name} ** 2"

    elif op == 'sqrt':
        x, out = entry[1], entry[2]
        if backend == "torch":
            return f"{out.name} = torch.sqrt({x.name})"
        else:
            return f"{out.name} = jnp.sqrt({x.name})"

    elif op == 'cadd':
        x, c, out = entry[1], entry[2], entry[3]
        return f"{out.name} = {x.name} + {c}"

    return None


# =============================================================================
# Public API
# =============================================================================
def generate_backward(sparse=False, problem='cavity', backend="torch",
                      kronecker=False):
    """Generate an optimized backward function for the PDE residuals.

    Args:
        sparse: if True, generate sparse variant for SK-PINN
        problem: 'cavity' (NS+Smagorinsky), 'kovasznay' (constant-viscosity NS),
            or 'elasticity' (Navier-Cauchy).
        backend: 'torch' (default) or 'jax'.
        kronecker: if True and backend='jax', replace dense D^T matmuls
            in the backward with Kronecker-structured ops (BFSA).

    Returns:
        (source_code, backward_fn) tuple
    """
    if problem == 'elasticity':
        from src.lid_benchmark import compute_pde_elasticity, compute_pde_elasticity_sparse
        compute_fn = compute_pde_elasticity_sparse if sparse else compute_pde_elasticity
        # Parametric family F3: lam_e, mu_e threaded via g dict as TracedVar
        # scalar constants, so the emitted backward reads them from g at runtime
        # rather than baking in the (lam_e=1, mu_e=0.5) defaults.
        if sparse:
            constants = ['Dx', 'Dy', 'fx', 'fy', 'lam_e', 'mu_e']
        else:
            constants = ['Dxx', 'Dyy', 'Dxy', 'fx', 'fy', 'lam_e', 'mu_e']
        input_names = ['ux', 'uy']
        seed_names = ['deq_x', 'deq_y']
        func_name = 'generated_elasticity_grad'
    elif problem == 'kovasznay':
        from src.lid_benchmark import compute_pde_kovasznay
        compute_fn = compute_pde_kovasznay
        # Parametric family F2: nu_kov threaded via g dict as TracedVar constant.
        constants = ['Dx', 'Dy', 'nu_kov']
        input_names = ['u', 'v', 'p']
        seed_names = ['dc', 'dmu', 'dmv']
        func_name = 'generated_kovasznay_grad'
    else:
        from src.lid_benchmark import compute_pde_terms, compute_pde_terms_sparse
        compute_fn = compute_pde_terms_sparse if sparse else compute_pde_terms
        # Parametric family F1: nu_lam threaded via g dict as TracedVar constant.
        constants = ['Dx', 'Dy', 'Cs_d_sq', 'nu_lam']
        input_names = ['u', 'v', 'p']
        seed_names = ['dc', 'dmu', 'dmv']
        func_name = 'generated_analytical_grad'

    tape = []
    outputs, inputs = trace_pde_forward(compute_fn, None, tape, sparse=sparse,
                                         constants=constants,
                                         input_names=input_names)

    source, fn = emit_backward(
        tape, list(outputs), seed_names, inputs, sparse=sparse,
        func_name=func_name, input_names=input_names, backend=backend
    )

    if kronecker and backend == "jax":
        source, fn = _kronecker_postprocess(source, func_name, problem)

    return source, fn


def _kronecker_postprocess(source, func_name, problem):
    """Replace dense D^T matmuls with Kronecker-structured ops (BFSA).

    For Chebyshev spectral operators on a Ny×Nx grid:
        DxT = I ⊗ D1d^T  →  V @ D1d  (reshape, 50×50 matmul, reshape)
        DyT = D1d^T ⊗ I  →  D1dT @ V  (reshape, 50×50 matmul, reshape)
    Similarly for DxxT, DyyT, DxyT (elasticity).
    """
    import re
    import jax.numpy as jnp

    backward_replacements = [
        ("DxxT", "_kron_dxxt"),
        ("DyyT", "_kron_dyyt"),
        ("DxyT", "_kron_dxyt"),
        ("DxT", "_kron_dxt"),
        ("DyT", "_kron_dyt"),
    ]
    forward_replacements = [
        ("Dxx", "_kron_dxx"),
        ("Dyy", "_kron_dyy"),
        ("Dxy", "_kron_dxy"),
        ("Dx", "_kron_dx"),
        ("Dy", "_kron_dy"),
    ]
    ordered_replacements = backward_replacements[:]
    if problem != 'elasticity':
        ordered_replacements += forward_replacements
    new_source = source
    for key, fn_name in ordered_replacements:
        pattern = rf"g\['{key}'\] @ (\w+)"
        new_source = re.sub(pattern, rf"{fn_name}(\1, g)", new_source)

    helper_defs = '''
def _kron_dxt(adj, g):
    """(I ⊗ Dx_1d^T) @ adj via Kronecker: V @ Dx_1d."""
    Ng = g['N_grid']
    V = adj.reshape(Ng, Ng)
    return (V @ g['D1d_x']).reshape(Ng * Ng, 1)

def _kron_dyt(adj, g):
    """(Dy_1d^T ⊗ I) @ adj via Kronecker: Dy_1dT @ V."""
    Ng = g['N_grid']
    V = adj.reshape(Ng, Ng)
    return (g['D1dT_y'] @ V).reshape(Ng * Ng, 1)

def _kron_dxxt(adj, g):
    """(I ⊗ (Dx_1d²)^T) @ adj via Kronecker: V @ Dx_1d_sq."""
    Ng = g['N_grid']
    V = adj.reshape(Ng, Ng)
    return (V @ g['D1d_sq_x']).reshape(Ng * Ng, 1)

def _kron_dyyt(adj, g):
    """((Dy_1d²)^T ⊗ I) @ adj via Kronecker: Dy_1dT_sq @ V."""
    Ng = g['N_grid']
    V = adj.reshape(Ng, Ng)
    return (g['D1dT_sq_y'] @ V).reshape(Ng * Ng, 1)

def _kron_dxyt(adj, g):
    """(Dy_1d^T ⊗ Dx_1d^T) @ adj via Kronecker: Dy_1dT @ V @ Dx_1d."""
    Ng = g['N_grid']
    V = adj.reshape(Ng, Ng)
    return (g['D1dT_y'] @ V @ g['D1d_x']).reshape(Ng * Ng, 1)

def _kron_dx(v, g):
    """(I ⊗ Dx_1d) @ v via Kronecker: V @ Dx_1d^T."""
    Ng = g['N_grid']
    V = v.reshape(Ng, Ng)
    return (V @ g['D1dT_x']).reshape(Ng * Ng, 1)

def _kron_dy(v, g):
    """(Dy_1d ⊗ I) @ v via Kronecker: Dy_1d @ V."""
    Ng = g['N_grid']
    V = v.reshape(Ng, Ng)
    return (g['D1d_y'] @ V).reshape(Ng * Ng, 1)

def _kron_dxx(v, g):
    """(I ⊗ Dx_1d²) @ v via Kronecker: V @ (Dx_1d^T)²."""
    Ng = g['N_grid']
    V = v.reshape(Ng, Ng)
    return (V @ g['D1dT_sq_x']).reshape(Ng * Ng, 1)

def _kron_dyy(v, g):
    """(Dy_1d² ⊗ I) @ v via Kronecker: Dy_1d² @ V."""
    Ng = g['N_grid']
    V = v.reshape(Ng, Ng)
    return (g['D1d_sq_y'] @ V).reshape(Ng * Ng, 1)

def _kron_dxy(v, g):
    """(Dy_1d ⊗ Dx_1d) @ v via Kronecker: Dy_1d @ V @ Dx_1d^T."""
    Ng = g['N_grid']
    V = v.reshape(Ng, Ng)
    return (g['D1d_y'] @ V @ g['D1dT_x']).reshape(Ng * Ng, 1)
'''
    namespace = {'jnp': jnp}
    exec(helper_defs, namespace)

    full_source = helper_defs + "\n" + new_source
    exec(new_source, namespace)
    fn = namespace[func_name]
    return full_source, fn


# =============================================================================
# Verification
# =============================================================================
def verify_against_analytical(device='cuda'):
    """Level 2 verification: compare generated vs hand-derived backward."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.lid_benchmark import (build_grid_data, compute_pde_terms,
                                    compute_analytical_grad)

    print("=" * 60)
    print("Verification: Generated vs Hand-Derived Backward")
    print("=" * 60)

    # Generate backward
    print("\n[1] Tracing forward computation...")
    tape = []
    outputs, inputs = trace_pde_forward(compute_pde_terms, None, tape, sparse=False)
    print(f"    Tape entries: {len(tape)}")

    print("\n[2] Generating backward code...")
    source, gen_fn = emit_backward(tape, list(outputs), ['dc', 'dmu', 'dmv'], inputs, sparse=False)

    print("\n--- Generated Code ---")
    print(source)
    print("--- End Generated Code ---")

    # Test at N=10
    for N in [10, 50]:
        print(f"\n[3] Testing at N={N} ({N*N} points)...")
        g = build_grid_data(N, device)
        pred = torch.randn(g['N_all'], 3, device=device)

        # Hand-derived
        hand_grad = compute_analytical_grad(pred, g)

        # Generated
        gen_grad = gen_fn(pred, g)

        diff = (gen_grad - hand_grad).abs().max().item()
        print(f"    max |diff| vs hand-derived: {diff:.2e}")
        status = "PASS" if diff < 1e-4 else "FAIL"
        print(f"    Status: {status}")

    # Also verify against autograd
    print(f"\n[4] Verifying against torch.autograd...")
    N = 10
    g = build_grid_data(N, device)
    pred_auto = torch.randn(g['N_all'], 3, device=device, requires_grad=True)
    c, mu, mv = compute_pde_terms(pred_auto, g)
    M = g['M']
    scale = 2.0 / M
    mask = g['interior_mask']
    loss = ((c**2 + mu**2 + mv**2) * mask).sum() / M
    auto_grad = torch.autograd.grad(loss, pred_auto)[0]

    gen_grad = gen_fn(pred_auto.detach(), g)
    diff = (gen_grad - auto_grad).abs().max().item()
    print(f"    max |diff| vs autograd: {diff:.2e}")
    status = "PASS" if diff < 1e-4 else "FAIL"
    print(f"    Status: {status}")


# =============================================================================
# SK-CERT — SAGE-Kronecker Certification (C1, Phase 5)
# =============================================================================
# Computes a scalar certificate B = C_lambda * R(theta; lambda) from the
# Kronecker factor K_sens(x) that SAGE maintains natively as part of its
# analytical VJP. The minimum eigenvalue of G_lambda = sum_x K_sens^T K_sens
# plays the role of the squared discrete inf-sup constant under a Korn-type
# inequality specific to F3 linear elasticity. See
# `llmdocs/research/research_log/04_design.md` § SELECTION and
# `llmdocs/research/research_log/contract_target_pin.md` for the target pin.
# =============================================================================

def _K_sens_elasticity(lam_e, mu_e, device=None, dtype=None):
    """Return the constant-coefficient 2x2 residual-to-output sensitivity
    matrix K_sens for 2D linear elasticity (Navier-Cauchy).
    """
    dtype = dtype or torch.float64
    # Coefficient matrix derived from the leading-order Navier-Cauchy form.
    # Row i of K_sens reads off the linear combination of output second-
    # derivative contributions that enter residual component R_i.
    # For constant-coefficient elasticity this is x-independent.
    a_diag = (lam_e + 2.0 * mu_e) + mu_e
    a_off = (lam_e + mu_e)
    return torch.tensor([[a_diag, a_off], [a_off, a_diag]],
                        device=device, dtype=dtype)


def compute_sk_cert(eq_x, eq_y, lam_e, mu_e, interior_mask, r=2):
    """SK-CERT bound for F3 linear elasticity.

    Implements the algorithm from research_log/04_design.md § SURVIVOR
    DESCRIPTIONS § C1 SK-CERT, steps (1)-(5), under the grid-consistent
    interpretation of G_lambda (the per-point Kronecker factor
    K_sens^T K_sens, equivalent to (1/N) * sum_x K_sens(x)^T K_sens(x);
    paired with RMS residual). This is the dimension-consistent form
    matching the classical Korn / Babuska-Aziz framework, where the
    stability constant is a property of the continuum operator and is
    grid-independent in the limit.

    See research_log/contract_interpretations.md
    § RULING-2026-04-18-A for the SG-4 sub-agent ruling that selected
    this reading over the algorithm-literal sum-G_lambda form (which
    produces an O(1/sqrt(N)) bound that vanishes under grid refinement
    — not a valid a-posteriori estimator). The literal sum-G_lambda
    quantity is still computed and returned as a diagnostic for
    reviewer transparency.

    For constant-coefficient elasticity K_sens(x) is x-independent, so
    the per-point factor IS the grid-average factor (no spatial averaging
    needed). For x-dependent operators (F1 cavity NS, F2 Kovasznay), the
    grid-consistent form is G_lambda = (1/N) sum_x K_sens(x)^T K_sens(x).

    Args:
        eq_x, eq_y: residual fields at collocation points, shape (N, 1) each.
        lam_e, mu_e: first Lame parameter and shear modulus.
        interior_mask: (N, 1) tensor with 1.0 on interior, 0.0 on boundary.
        r: rank for the top-r eigendecomposition (here K=2, so r in {1,2};
            the spec uses lambda_min^{(r)} = r-th-smallest eigenvalue).

    Returns:
        Dict of diagnostics with keys: B, C_lambda, R_rms, B_local,
        C_lambda_local, lambda_min_literal, lambda_min_local, eigvals_G,
        K_sens, N_eff, lam_e, mu_e, r.
    """
    assert eq_x.shape == eq_y.shape, "residual component shapes must match"
    device = eq_x.device
    # RMS residual on interior (matches contract eval protocol).
    R_sq = (eq_x ** 2 + eq_y ** 2) * interior_mask
    N_eff = float(interior_mask.sum().item())
    if N_eff <= 0:
        raise ValueError("interior_mask must have at least one interior point")
    R_rms = float(torch.sqrt(R_sq.sum() / N_eff).item())

    # K_sens: constant 2x2 for F3 (parametric in (lam_e, mu_e) per instance).
    K = _K_sens_elasticity(float(lam_e), float(mu_e), device=device)
    Ktk = K.T @ K  # 2x2, symmetric PSD

    # Reading-1 (algorithm-literal): G_lambda = N_eff * K^T K
    G_literal = N_eff * Ktk
    eigvals_lit = torch.linalg.eigvalsh(G_literal).sort().values  # ascending
    r_eff = int(max(1, min(r, K.shape[0])))
    lam_min_lit = float(eigvals_lit[r_eff - 1].item())  # r-th smallest
    C_lit = 1.0 / (lam_min_lit ** 0.5) if lam_min_lit > 0 else float("inf")
    B_lit = C_lit * R_rms

    # Reading-2 (grid-consistent): G_lambda = K^T K (no N factor),
    # equivalently lambda_min^{(r)}(N^{-1} G_literal).
    eigvals_loc = torch.linalg.eigvalsh(Ktk).sort().values
    lam_min_loc = float(eigvals_loc[r_eff - 1].item())
    C_loc = 1.0 / (lam_min_loc ** 0.5) if lam_min_loc > 0 else float("inf")
    B_loc = C_loc * R_rms

    return {
        "B": B_lit,
        "C_lambda": C_lit,
        "R_rms": R_rms,
        "B_local": B_loc,
        "C_lambda_local": C_loc,
        "lambda_min_literal": lam_min_lit,
        "lambda_min_local": lam_min_loc,
        "eigvals_G_literal": [float(v) for v in eigvals_lit.tolist()],
        "eigvals_G_local": [float(v) for v in eigvals_loc.tolist()],
        "K_sens": [[float(v) for v in row] for row in K.tolist()],
        "N_eff": int(N_eff),
        "lam_e": float(lam_e),
        "mu_e": float(mu_e),
        "r": r_eff,
    }


# =============================================================================
# KT-LAP — Kronecker-Trajectory Laplace (C3′, Phase 5)
# =============================================================================
# Trajectory-ensembled Gauss-Newton Hessian posterior on PINN parameters.
# Σ^{-1} ≈ γI + Σ_t β_t J_t(θ_t)^T J_t(θ_t) with J_t the residual-to-parameter
# Jacobian emitted natively by SAGE's generated backward. Top-r Ritz pairs
# (V, Λ) of the trajectory-ensembled Hessian, then Σ ≈ γI + V Λ^{-1} V^T
# per 04_design.md § C3′ step (4). See contract_target_pin.md § T4 for the
# permitted hyperparameter levers and 05_results.md § SMOKE-TEST EXECUTION
# PLAN for the smoke configuration.
# =============================================================================
import math


def flatten_params(model):
    """Flatten model parameters into a 1-D tensor of shape (P,)."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def unflatten_into_model(model, flat):
    """Copy a flat (P,) tensor back into model.parameters() in-place."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        with torch.no_grad():
            p.copy_(flat[offset:offset + n].reshape(p.shape))
        offset += n


def param_layer_slices(model):
    """Return a list of (start, end) slices into the flat parameter vector, one
    per nn.Parameter tensor. Used for layer-wise block diagonal (K-FAC analogue)
    restrictions of the GN Hessian HVP in P2 ablation paths.
    """
    slices = []
    offset = 0
    for p in model.parameters():
        n = p.numel()
        slices.append((offset, offset + n))
        offset += n
    return slices


def _split_flat_to_params(flat, params):
    """Split flat (P,) into a list of tensors matching params' shapes."""
    out = []
    offset = 0
    for p in params:
        n = p.numel()
        out.append(flat[offset:offset + n].reshape(p.shape))
        offset += n
    return out


def gn_hvp_pinn(model, loss_closure, v_flat, create_graph=False):
    """Hessian-vector product of a PINN residual-squared loss via Pearlmutter's
    double-back trick. At a trained PINN the residuals are small so the full
    Hessian coincides with the Gauss-Newton approximation up to O(||r||) terms.

    Args:
        model: PINN whose nn.Parameters carry the current θ.
        loss_closure: zero-arg callable returning the (scalar) PINN loss that
            depends on model.parameters() via a fresh autograd graph. Must use
            create_graph-compatible operations (standard torch ops).
        v_flat: (P,) tensor, same device/dtype as params.
        create_graph: if True the HVP itself is differentiable (unused here).

    Returns:
        (P,) tensor H v.
    """
    params = list(model.parameters())
    loss = loss_closure()
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grads = torch.cat([g.reshape(-1) for g in grads])
    gv = (flat_grads * v_flat).sum()
    hvp = torch.autograd.grad(gv, params, retain_graph=False,
                              create_graph=create_graph)
    return torch.cat([h.reshape(-1) for h in hvp])


def gn_hvp_pinn_layerwise(model, loss_closure, v_flat):
    """Block-diagonal HVP: only the diagonal-block couplings within each
    nn.Parameter tensor are kept. Implements the layer-wise-K-FAC-analogue
    ablation for P2 (B_no_SAGE): cross-layer Hessian entries are zeroed.

    Computed by projecting v_flat onto each parameter slice in isolation and
    reading back only that slice of the HVP result; cross-layer contributions
    are discarded. This is strictly STRONGER than K-FAC's within-layer
    Kronecker factorisation, so a ≥5× coverage-gap degradation under this
    approximation is a conservative lower bound on the degradation K-FAC
    would show.
    """
    slices = param_layer_slices(model)
    out = torch.zeros_like(v_flat)
    for (s, e) in slices:
        v_l = torch.zeros_like(v_flat)
        v_l[s:e] = v_flat[s:e]
        hvp_l = gn_hvp_pinn(model, loss_closure, v_l)
        out[s:e] = hvp_l[s:e]
    return out


def lanczos_topk(hvp_fn, P, k, max_iters=100, device='cuda',
                 dtype=torch.float32, tol=1e-10, seed=0):
    """Lanczos iteration with full reorthogonalisation (twice-modified Gram-
    Schmidt) for a symmetric PSD operator exposed only through an HVP oracle.

    Args:
        hvp_fn: callable v (P,) -> A v (P,); A assumed symmetric.
        P: ambient dimension.
        k: number of Ritz pairs to return (largest eigenvalues).
        max_iters: Lanczos iterations; returned Ritz rank ≤ min(max_iters, P).
        device, dtype: storage configuration for Krylov basis Q.
        tol: early-termination threshold on β_j.
        seed: RNG seed for the initial Krylov vector (reproducibility).

    Returns:
        V: (P, k_eff) orthonormal columns (Ritz vectors), k_eff = min(k, m_used).
        Lam: (k_eff,) Ritz values (DESCENDING order — largest first).
    """
    m = min(max_iters, P)
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    q = torch.randn(P, device=device, dtype=dtype, generator=g)
    q = q / q.norm()

    Q = torch.zeros(P, m + 1, device=device, dtype=dtype)
    Q[:, 0] = q
    alphas = torch.zeros(m, device=device, dtype=dtype)
    betas = torch.zeros(m, device=device, dtype=dtype)

    beta = torch.zeros((), device=device, dtype=dtype)
    q_prev = torch.zeros(P, device=device, dtype=dtype)

    m_used = m
    for j in range(m):
        w = hvp_fn(Q[:, j])
        w = w - beta * q_prev
        alpha = (w * Q[:, j]).sum()
        alphas[j] = alpha
        w = w - alpha * Q[:, j]

        # Full reorthogonalisation (twice for numerical robustness).
        for _ in range(2):
            coeffs = Q[:, :j + 1].T @ w
            w = w - Q[:, :j + 1] @ coeffs

        beta = w.norm()
        betas[j] = beta
        if float(beta.item()) < tol:
            m_used = j + 1
            break
        q_prev = Q[:, j].clone()
        Q[:, j + 1] = w / beta

    # Tridiagonal T (symmetric).
    T = torch.zeros(m_used, m_used, device=device, dtype=dtype)
    T[torch.arange(m_used), torch.arange(m_used)] = alphas[:m_used]
    if m_used >= 2:
        idx_upper = torch.arange(m_used - 1)
        T[idx_upper, idx_upper + 1] = betas[:m_used - 1]
        T[idx_upper + 1, idx_upper] = betas[:m_used - 1]

    # Eigendecompose T (small, dense).
    evals, evecs = torch.linalg.eigh(T)
    # Descending.
    sort_desc = torch.argsort(evals, descending=True)
    evals_d = evals[sort_desc]
    evecs_d = evecs[:, sort_desc]

    k_eff = int(min(k, m_used))
    top_vecs = evecs_d[:, :k_eff]
    top_vals = evals_d[:k_eff]

    V = Q[:, :m_used] @ top_vecs   # (P, k_eff) Ritz vectors
    return V, top_vals


def kt_laplace_sample(theta_star, V, Lam, gamma, n_sample, device=None,
                      dtype=None, generator=None):
    """Draw posterior samples θ^{(s)} ~ N(θ*, Σ) with
        Σ = γ I_P + V diag(1/Λ) V^T
    per 04_design.md § C3′ step (5). V must be orthonormal columns; Λ must be
    non-negative Ritz eigenvalues of the trajectory-ensembled Hessian. For the
    low-Ritz-rank regime (r ≪ P), the √γ-scaled isotropic component dominates
    the parameter-perturbation norm; the low-rank component inflates variance
    in the Ritz-span directions.

    Returns:
        samples: (n_sample, P).
    """
    device = device or theta_star.device
    dtype = dtype or theta_star.dtype
    P = theta_star.numel()
    r = V.shape[1]
    sqrt_gamma = math.sqrt(float(gamma))
    inv_sqrt_lam = 1.0 / torch.sqrt(torch.clamp(Lam, min=1e-30))

    xi = torch.randn(n_sample, P, device=device, dtype=dtype,
                     generator=generator)
    z = torch.randn(n_sample, r, device=device, dtype=dtype,
                    generator=generator)

    iso = sqrt_gamma * xi                          # (n, P)
    low = (z * inv_sqrt_lam.unsqueeze(0)) @ V.T     # (n, P)
    return theta_star.unsqueeze(0) + iso + low


if __name__ == '__main__':
    verify_against_analytical()
