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
def trace_pde_forward(compute_fn, N_all, tape, sparse=False, constants=None):
    """Trace compute_pde_terms through TracedVar to build tape.

    Args:
        compute_fn: PDE forward function (pred, g) -> (continuity, mom_u, mom_v)
        N_all: grid size (can be None for tracing)
        tape: list to record operations
        sparse: if True, use sparse matmul
        constants: list of constant names for grid_data (default: ['Dx', 'Dy', 'Cs_d_sq'])

    Returns (output_vars, input_vars) where:
      - output_vars = (continuity, mom_u, mom_v) TracedVars
      - input_vars = {'u': TracedVar, 'v': TracedVar, 'p': TracedVar}
    """
    if constants is None:
        constants = ['Dx', 'Dy', 'Cs_d_sq']

    TracedVar._counter = 0

    # Create traced input variables
    u = TracedVar('u', tape)
    v = TracedVar('v', tape)
    p = TracedVar('p', tape)

    # Create a fake pred object that supports slicing
    class FakePred:
        def __getitem__(self, key):
            if isinstance(key, tuple) and len(key) == 2:
                _, col = key
                if col == slice(0, 1):
                    return u
                elif col == slice(1, 2):
                    return v
                elif col == slice(2, 3):
                    return p
            raise KeyError(f"Unsupported slice: {key}")

    # Create traced grid_data with constant TracedVars for matrices
    g = {'N_all': N_all}
    for name in constants:
        g[name] = TracedVar(name, tape, is_const=True)

    pred = FakePred()
    continuity, mom_u, mom_v = compute_fn(pred, g)

    return (continuity, mom_u, mom_v), {'u': u, 'v': v, 'p': p}


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
    # Determine which matrix (Dx or Dy) and use precomputed transpose
    a_name = A.name  # 'Dx' or 'Dy'
    t_name = a_name.replace('Dx', 'DxT').replace('Dy', 'DyT')
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
def emit_backward(tape, output_vars, seed_names, input_vars, sparse=False, func_name=None):
    """Generate a PyTorch backward function from symbolic adjoint expressions.

    Args:
        func_name: name for the generated function (default: 'generated_analytical_grad')

    Returns (source_code, compiled_fn).
    """
    if func_name is None:
        func_name = 'generated_analytical_grad'

    adj = symbolic_backward(tape, output_vars, seed_names)

    lines_v2 = []
    lines_v2.append(f"def {func_name}(pred_det, g):")
    lines_v2.append("    import torch")
    lines_v2.append(f"    u = pred_det[:g['N_all'], 0:1]")
    lines_v2.append(f"    v = pred_det[:g['N_all'], 1:2]")
    lines_v2.append(f"    p = pred_det[:g['N_all'], 2:3]")
    lines_v2.append("")

    # Emit ALL forward ops
    for entry in tape:
        line = _emit_forward_op(entry, sparse)
        if line:
            lines_v2.append(f"    {line}")

    lines_v2.append("")
    lines_v2.append("    # Loss scaling (adjoint seeds)")
    lines_v2.append("    M = g['M']")
    lines_v2.append("    scale = 2.0 / M")
    lines_v2.append("    mask = g['interior_mask']")

    # Map output var names to PDE residual names
    out_names = [ov.name for ov in output_vars]
    pde_names = ['continuity', 'mom_u', 'mom_v']
    for oname, pname, sname in zip(out_names, pde_names, seed_names):
        lines_v2.append(f"    {sname} = {oname} * scale * mask")

    lines_v2.append("")
    lines_v2.append("    # Backward pass — adjoint accumulation")

    # Emit adjoint computations in reverse tape order
    # First, compute all adj_ variables for tape outputs
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

    # Also emit adj for u, v, p
    lines_v2.append("")
    lines_v2.append("    # Accumulate final gradients for u, v, p")
    for var_name in ['u', 'v', 'p']:
        if var_name in adj:
            exprs = adj[var_name]
            adj_name = f"adj_{var_name}"
            lines_v2.append(f"    {adj_name} = {exprs[0]}")
            for e in exprs[1:]:
                lines_v2.append(f"    {adj_name} = {adj_name} + {e}")
        else:
            lines_v2.append(f"    adj_{var_name} = torch.zeros_like({var_name})")

    lines_v2.append("")
    lines_v2.append("    return torch.cat([adj_u, adj_v, adj_p], dim=1)")

    source = "\n".join(lines_v2)

    # Compile
    namespace = {'torch': torch}
    exec(source, namespace)
    fn = namespace[func_name]

    return source, fn


def _emit_forward_op(entry, sparse=False):
    """Convert a tape entry to a forward computation line."""
    op = entry[0]

    if op == 'matmul':
        A, x, out = entry[1], entry[2], entry[3]
        return f"{out.name} = g['{A.name}'] @ {x.name}"

    elif op == 'sparse_matmul':
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
        return f"{out.name} = torch.sqrt({x.name})"

    elif op == 'cadd':
        x, c, out = entry[1], entry[2], entry[3]
        return f"{out.name} = {x.name} + {c}"

    return None


# =============================================================================
# Public API
# =============================================================================
def generate_backward(sparse=False, problem='cavity'):
    """Generate an optimized backward function for the PDE residuals.

    Args:
        sparse: if True, generate sparse variant for SK-PINN
        problem: 'cavity' (NS+Smagorinsky) or 'kovasznay' (constant-viscosity NS)

    Returns:
        (source_code, backward_fn) tuple
    """
    if problem == 'kovasznay':
        from src.lid_benchmark import compute_pde_kovasznay
        compute_fn = compute_pde_kovasznay
        constants = ['Dx', 'Dy']
        func_name = 'generated_kovasznay_grad'
    else:
        from src.lid_benchmark import compute_pde_terms, compute_pde_terms_sparse
        compute_fn = compute_pde_terms_sparse if sparse else compute_pde_terms
        constants = ['Dx', 'Dy', 'Cs_d_sq']
        func_name = 'generated_analytical_grad'

    tape = []
    outputs, inputs = trace_pde_forward(compute_fn, None, tape, sparse=sparse,
                                         constants=constants)

    source, fn = emit_backward(
        tape, list(outputs), ['dc', 'dmu', 'dmv'], inputs, sparse=sparse,
        func_name=func_name
    )

    return source, fn


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


if __name__ == '__main__':
    verify_against_analytical()
