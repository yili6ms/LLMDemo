"""Hyperparameter sweep script for TinyGPT."""

import itertools
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import subprocess
import sys
from datetime import datetime


def generate_sweep_configs(base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate sweep configurations from hyperparameter grid."""
    
    # Define hyperparameter grid
    sweep_params = {
        'd_model': [64, 128, 256],
        'n_layers': [2, 4, 6],
        'learning_rate': [0.0001, 0.0005, 0.001, 0.002],
        'dropout': [0.0, 0.1, 0.2],
    }
    
    # Generate all combinations
    configs = []
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    
    for combination in itertools.product(*param_values):
        config = base_config.copy()
        
        # Update with sweep parameters
        for param, value in zip(param_names, combination):
            config[param] = value
            
        # Ensure n_heads divides d_model
        while config['d_model'] % config.get('n_heads', 4) != 0:
            config['n_heads'] = config.get('n_heads', 4) - 1
            if config['n_heads'] < 1:
                config['n_heads'] = 1
                break
        
        # Adjust batch size based on model size for memory constraints
        if config['d_model'] >= 256:
            config['batch_size'] = 4
        elif config['d_model'] >= 128:
            config['batch_size'] = 6
        else:
            config['batch_size'] = 8
            
        # Shorter training for sweep
        config['max_steps'] = 2000
        config['eval_interval'] = 200
        config['log_interval'] = 50
        
        configs.append(config)
    
    return configs


def run_experiment(config: Dict[str, Any], experiment_id: str) -> Dict[str, Any]:
    """Run a single experiment with given config."""
    
    # Create experiment directory
    exp_dir = Path(f"experiments/{experiment_id}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = exp_dir / "config.yaml"
    config['checkpoint_path'] = str(exp_dir / "best.pt")
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Running experiment {experiment_id}...")
    print(f"Config: d_model={config['d_model']}, n_layers={config['n_layers']}, "
          f"lr={config['learning_rate']}, dropout={config['dropout']}")
    
    try:
        # Run training
        result = subprocess.run([
            sys.executable, "train.py", "--config", str(config_path)
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode != 0:
            print(f"Experiment {experiment_id} failed:")
            print(result.stderr)
            return {
                'experiment_id': experiment_id,
                'status': 'failed',
                'error': result.stderr,
                **config
            }
        
        # Parse final validation loss from output
        output_lines = result.stdout.strip().split('\n')
        final_loss = None
        final_ppl = None
        
        for line in reversed(output_lines):
            if 'Final - Loss:' in line:
                try:
                    parts = line.split('Loss: ')[1].split(',')[0]
                    final_loss = float(parts)
                    ppl_parts = line.split('PPL: ')[1]
                    final_ppl = float(ppl_parts)
                    break
                except (IndexError, ValueError):
                    continue
        
        if final_loss is None:
            # Try to get last validation loss
            for line in reversed(output_lines):
                if 'Validation - Loss:' in line:
                    try:
                        parts = line.split('Loss: ')[1].split(',')[0]
                        final_loss = float(parts)
                        ppl_parts = line.split('PPL: ')[1]
                        final_ppl = float(ppl_parts)
                        break
                    except (IndexError, ValueError):
                        continue
        
        return {
            'experiment_id': experiment_id,
            'status': 'completed',
            'final_loss': final_loss,
            'final_perplexity': final_ppl,
            **config
        }
        
    except subprocess.TimeoutExpired:
        return {
            'experiment_id': experiment_id,
            'status': 'timeout',
            **config
        }
    except Exception as e:
        return {
            'experiment_id': experiment_id,
            'status': 'error',
            'error': str(e),
            **config
        }


def main():
    """Run hyperparameter sweep."""
    
    # Load base configuration
    base_config_path = Path("configs/tiny.yaml")
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Generate sweep configurations
    sweep_configs = generate_sweep_configs(base_config)
    
    print(f"Generated {len(sweep_configs)} sweep configurations")
    
    # Run experiments
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, config in enumerate(sweep_configs):
        experiment_id = f"sweep_{timestamp}_{i:03d}"
        result = run_experiment(config, experiment_id)
        results.append(result)
        
        # Save intermediate results
        results_path = Path(f"experiments/sweep_results_{timestamp}.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Analyze results
    print("\n" + "="*80)
    print("SWEEP RESULTS SUMMARY")
    print("="*80)
    
    successful_results = [r for r in results if r['status'] == 'completed' and r.get('final_loss')]
    
    if successful_results:
        # Sort by final loss
        successful_results.sort(key=lambda x: x['final_loss'])
        
        print(f"Successful experiments: {len(successful_results)}/{len(results)}")
        print("\nTop 5 configurations:")
        
        for i, result in enumerate(successful_results[:5]):
            print(f"\n{i+1}. Loss: {result['final_loss']:.4f}, PPL: {result.get('final_perplexity', 'N/A'):.2f}")
            print(f"   d_model={result['d_model']}, n_layers={result['n_layers']}, "
                  f"lr={result['learning_rate']}, dropout={result['dropout']}")
            print(f"   Experiment: {result['experiment_id']}")
        
        # Generate CSV for further analysis
        csv_path = Path(f"experiments/sweep_results_{timestamp}.csv")
        with open(csv_path, 'w') as f:
            f.write("experiment_id,d_model,n_layers,learning_rate,dropout,final_loss,final_perplexity,status\n")
            for result in results:
                f.write(f"{result['experiment_id']},{result.get('d_model', '')},{result.get('n_layers', '')},"
                       f"{result.get('learning_rate', '')},{result.get('dropout', '')},{result.get('final_loss', '')},"
                       f"{result.get('final_perplexity', '')},{result['status']}\n")
        
        print(f"\nDetailed results saved to: {results_path}")
        print(f"CSV results saved to: {csv_path}")
        
    else:
        print("No successful experiments found!")
        failed_count = len([r for r in results if r['status'] == 'failed'])
        timeout_count = len([r for r in results if r['status'] == 'timeout'])
        error_count = len([r for r in results if r['status'] == 'error'])
        
        print(f"Failed: {failed_count}, Timeout: {timeout_count}, Error: {error_count}")


if __name__ == "__main__":
    main()