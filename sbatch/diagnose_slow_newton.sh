#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=0-00:30:00
#SBATCH --output=./logs/diagnose-%N-%j.out
#SBATCH --error=./logs/diagnose-%N-%j.err

echo "========================================"
echo "DIAGNOSTIC: Newton Iteration Slowdown"
echo "========================================"
echo "Started: $(date)"
echo ""

# Try WITHOUT scipy-stack first
echo "=== TEST 1: Without scipy-stack (virtualenv only) ==="
source ./env/bin/activate
echo "Python: $(which python3)"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__} from {numpy.__file__}')"
python3 -c "import scipy; print(f'SciPy: {scipy.__version__} from {scipy.__file__}')"
python3 -c "import numpy; numpy.show_config()" 2>/dev/null | grep -A2 "blas\|lapack" || echo "(config hidden)"
echo ""

export PYTHONPATH="$PYTHONPATH:$(pwd)"

echo "Running deep4 on nonlinear-poisson (should be <5s if fix works)..."
time python3 -c "
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/experiment_dt_elm_pinn')
from tasks import TaskRegistry
from models import ModelRegistry

task = TaskRegistry.get('nonlinear-poisson')(file_path='tasks/matlab_pde_tasks/nonlinear_poisson_exponential/2_2236.mat')
model = ModelRegistry.get('dt-elm-pinn-deep4')(task)
result = model.train(verbose=True)
print(f'Time: {result.train_time:.2f}s, L2: {result.l2_error:.4e}')
"
echo ""

# Now try WITH scipy-stack
echo "=== TEST 2: With scipy-stack loaded ==="
deactivate 2>/dev/null
module load scipy-stack
module load arrow
source ./env/bin/activate
echo "Python: $(which python3)"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__} from {numpy.__file__}')"
python3 -c "import scipy; print(f'SciPy: {scipy.__version__} from {scipy.__file__}')"
python3 -c "import numpy; numpy.show_config()" 2>/dev/null | grep -A2 "blas\|lapack" || echo "(config hidden)"
echo ""

echo "Running deep4 on nonlinear-poisson..."
time python3 -c "
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/experiment_dt_elm_pinn')
from tasks import TaskRegistry
from models import ModelRegistry

task = TaskRegistry.get('nonlinear-poisson')(file_path='tasks/matlab_pde_tasks/nonlinear_poisson_exponential/2_2236.mat')
model = ModelRegistry.get('dt-elm-pinn-deep4')(task)
result = model.train(verbose=True)
print(f'Time: {result.train_time:.2f}s, L2: {result.l2_error:.4e}')
"
echo ""

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
