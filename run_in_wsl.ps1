# PowerShell helper to run commands inside the WSL2 FEniCSx environment from Windows
param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$CommandArgs
)

$WSL_DIR = "/mnt/g/My Drive/MSSiSc/SuSe26/11.00153 Modern Simulation Software Development/03 - Projects/Project 01 - FEM/modalSolver"

if ($CommandArgs.Count -eq 0) {
    Write-Host "Usage: .\run_in_wsl.ps1 [script.py or command]" -ForegroundColor Yellow
    Write-Host "Example: .\run_in_wsl.ps1 scripts/run_scenario_study.py"
    Write-Host "Example: .\run_in_wsl.ps1 pytest tests/test_verification.py"
    exit 1
}

Write-Host "Running inside WSL2 conda environment..." -ForegroundColor Green

# Join all arguments into a single command string
$cmd_str = $CommandArgs -join " "

# If the first argument is a python file, run it with python automatically
if ($CommandArgs[0].EndsWith(".py")) {
    $full_cmd = "PYTHONPATH=. python $cmd_str"
} else {
    $full_cmd = "PYTHONPATH=. $cmd_str"
}

# Run the command inside WSL2 conda environment
wsl bash -c ". ~/miniforge3/etc/profile.d/conda.sh || . ~/miniconda3/etc/profile.d/conda.sh || . ~/anaconda3/etc/profile.d/conda.sh; conda activate fenicsx-env && cd '$WSL_DIR' && $full_cmd"
