#!/usr/bin/env python3
"""
Generate dependency graphs for OpenWave documentation.
Creates both module-level and class-level dependency visualizations.
"""

import os
import sys
import subprocess
from pathlib import Path

def generate_pydeps_graph():
    """Generate comprehensive dependency graph using pydeps."""
    project_root = Path(__file__).parent.parent
    output_dir = Path(__file__).parent / '_static' / 'deps'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate main dependency graph
    cmd = [
        'pydeps',
        'openwave',
        '--max-bacon', '4',
        '--cluster',
        '--min-cluster-size', '2',
        '--keep-target-cluster',
        '--rankdir', 'TB',
        '--show-deps',
        '-o', str(output_dir / 'openwave_deps.svg'),
        '--no-show',
        '--pylib-all'
    ]
    
    try:
        subprocess.run(cmd, cwd=project_root, check=True)
        print(f"Generated dependency graph: {output_dir / 'openwave_deps.svg'}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating dependency graph: {e}")
        return False
    
    # Generate core module dependencies
    core_modules = [
        'constants', 'config', 'equations', 'quantum_space', 
        'quantum_wave', 'forces', 'heat', 'matter', 'motion', 'photon'
    ]
    
    for module in core_modules:
        cmd = [
            'pydeps',
            f'openwave.core.{module}',
            '--max-bacon', '2',
            '--rankdir', 'LR',
            '-o', str(output_dir / f'{module}_deps.svg'),
            '--no-show'
        ]
        
        try:
            subprocess.run(cmd, cwd=project_root, check=True)
            print(f"Generated {module} dependency graph")
        except subprocess.CalledProcessError:
            print(f"Skipping {module} (may not exist yet)")
    
    return True

def generate_import_map():
    """Generate a detailed import map as JSON."""
    import json
    import ast
    from collections import defaultdict
    
    project_root = Path(__file__).parent.parent
    openwave_dir = project_root / 'openwave'
    
    import_map = defaultdict(list)
    
    for py_file in openwave_dir.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
            
        module_name = str(py_file.relative_to(project_root))[:-3].replace('/', '.')
        
        try:
            with open(py_file, 'r') as f:
                tree = ast.parse(f.read())
                
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_map[module_name].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        import_map[module_name].append(node.module)
        except Exception as e:
            print(f"Error parsing {py_file}: {e}")
    
    output_file = Path(__file__).parent / '_static' / 'import_map.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(dict(import_map), f, indent=2, sort_keys=True)
    
    print(f"Generated import map: {output_file}")
    return True

if __name__ == '__main__':
    print("Generating dependency visualizations...")
    
    # Check if pydeps is installed
    try:
        subprocess.run(['pydeps', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing pydeps...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pydeps'], check=True)
    
    success = True
    success &= generate_pydeps_graph()
    success &= generate_import_map()
    
    if success:
        print("\nDependency generation complete!")
        print("View graphs in docs/_static/deps/")
    else:
        print("\nSome dependency graphs could not be generated")
        sys.exit(1)