# PowerShell version of run_grpo.sh for Windows

$env:ACCELERATE_LOG_LEVEL = 'info'

# Run the GRPO training
accelerate launch `
  --config_file "C:\Users\MSI\Desktop\ARENA\configs\grpo_cpu.yaml" `
  "C:\Users\MSI\Desktop\ARENA\src\grpo.py" `
  --config "C:\Users\MSI\Desktop\ARENA\configs\grpo.yaml"
