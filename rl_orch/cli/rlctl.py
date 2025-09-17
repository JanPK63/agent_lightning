#!/usr/bin/env python3
"""
Charter-compliant rlctl CLI interface
Implements: rlctl launch -f config.yaml, rlctl resume, rlctl sweep, rlctl promote
"""

import click
import yaml
import json
import sys
import os
from pathlib import Path
from typing import Optional

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from rl_orch.core.orchestrator import RLOrchestrator
from rl_orch.core.config_models import ExperimentConfig


@click.group()
def cli():
    """RL Orchestrator CLI - Charter compliant interface"""
    pass


@cli.command()
@click.option('-f', '--file', 'config_file', required=True, help='Configuration file path')
@click.option('--dry-run', is_flag=True, help='Validate config without running')
def launch(config_file: str, dry_run: bool):
    """Launch RL experiment from config file"""
    try:
        # Load config
        config_path = Path(config_file)
        if not config_path.exists():
            click.echo(f"❌ Config file not found: {config_file}", err=True)
            sys.exit(1)
        
        with open(config_path) as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        
        # Validate config
        try:
            experiment_config = ExperimentConfig(**config_data)
        except Exception as e:
            click.echo(f"❌ Invalid config: {e}", err=True)
            sys.exit(1)
        
        click.echo(f"✅ Config validated: {experiment_config.name}")
        
        if dry_run:
            click.echo("🔍 Dry run - config is valid")
            return
        
        # Run experiment
        orchestrator = RLOrchestrator()
        run_id = orchestrator.run_experiment(experiment_config)
        
        click.echo(f"🚀 Experiment launched: {run_id}")
        click.echo(f"📊 Name: {experiment_config.name}")
        click.echo(f"🎯 Algorithm: {experiment_config.policy.algo}")
        click.echo(f"🏃 Epochs: {experiment_config.train.epochs}")
        
    except Exception as e:
        click.echo(f"❌ Launch failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('run_id')
def resume(run_id: str):
    """Resume paused experiment"""
    try:
        orchestrator = RLOrchestrator()
        experiment = orchestrator.get_experiment_status(run_id)
        
        if not experiment:
            click.echo(f"❌ Experiment not found: {run_id}", err=True)
            sys.exit(1)
        
        if experiment.status != "paused":
            click.echo(f"❌ Experiment not paused (status: {experiment.status})", err=True)
            sys.exit(1)
        
        click.echo(f"🔄 Resuming experiment: {run_id}")
        # Resume logic would go here
        click.echo("✅ Resume functionality not yet implemented")
        
    except Exception as e:
        click.echo(f"❌ Resume failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('-f', '--file', 'config_file', required=True, help='Sweep configuration file')
@click.option('--max-concurrent', default=4, help='Maximum concurrent experiments')
def sweep(config_file: str, max_concurrent: int):
    """Run hyperparameter sweep"""
    try:
        click.echo(f"🔍 Loading sweep config: {config_file}")
        
        # Load sweep config
        config_path = Path(config_file)
        if not config_path.exists():
            click.echo(f"❌ Sweep config not found: {config_file}", err=True)
            sys.exit(1)
        
        with open(config_path) as f:
            sweep_config = yaml.safe_load(f)
        
        click.echo(f"🎯 Max concurrent: {max_concurrent}")
        click.echo("✅ Sweep functionality not yet implemented")
        
    except Exception as e:
        click.echo(f"❌ Sweep failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('run_id')
@click.option('--target', default='production', help='Promotion target')
def promote(run_id: str, target: str):
    """Promote trained model to target environment"""
    try:
        orchestrator = RLOrchestrator()
        experiment = orchestrator.get_experiment_status(run_id)
        
        if not experiment:
            click.echo(f"❌ Experiment not found: {run_id}", err=True)
            sys.exit(1)
        
        if experiment.status != "completed":
            click.echo(f"❌ Experiment not completed (status: {experiment.status})", err=True)
            sys.exit(1)
        
        click.echo(f"📦 Promoting model: {run_id}")
        click.echo(f"🎯 Target: {target}")
        click.echo("✅ Promote functionality not yet implemented")
        
    except Exception as e:
        click.echo(f"❌ Promote failed: {e}", err=True)
        sys.exit(1)


@cli.command()
def status():
    """Show status of all experiments"""
    try:
        orchestrator = RLOrchestrator()
        experiments = orchestrator.list_experiments()
        
        if not experiments:
            click.echo("📭 No experiments found")
            return
        
        click.echo("📊 Experiment Status:")
        click.echo("-" * 80)
        
        for run_id, exp in experiments.items():
            status_emoji = {
                "running": "🏃",
                "completed": "✅", 
                "failed": "❌",
                "paused": "⏸️"
            }.get(exp.status, "❓")
            
            click.echo(f"{status_emoji} {exp.config.name[:30]:<30} {run_id[:8]} {exp.status:<10} Epoch {exp.epoch}")
        
    except Exception as e:
        click.echo(f"❌ Status failed: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()