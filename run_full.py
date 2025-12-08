"""
Complete setup and test script for Generative Agents project.

This script automates the entire workflow from a fresh clone:
1. Checks prerequisites (Python, Ollama, dependencies)
2. Cleans existing agents and results
3. Creates settings.py from example
4. Installs Python dependencies
5. Verifies Ollama models are available
6. Creates agents from CSV
7. Runs agent testing
8. Displays results

Usage:
    python run_full.py
"""

import os
import sys
import shutil
import subprocess
import json
import time
from pathlib import Path
from typing import Tuple, List
from tqdm import tqdm

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_step(step_num: int, total: int, description: str):
    """Print a step description."""
    print(f"{Colors.OKCYAN}[{step_num}/{total}] {description}{Colors.ENDC}")


def print_success(message: str):
    """Print a success message."""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_error(message: str):
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def print_info(message: str):
    """Print an info message."""
    print(f"{Colors.OKBLUE}ℹ {message}{Colors.ENDC}")


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is 3.7 or higher."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        return True, f"{version.major}.{version.minor}.{version.micro}"
    return False, f"{version.major}.{version.minor}.{version.micro}"


def check_ollama_running() -> Tuple[bool, str]:
    """Check if Ollama is running and accessible."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return True, "Ollama is running"
        return False, f"Ollama returned status code {response.status_code}"
    except ImportError:
        # Try with subprocess instead
        try:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/tags"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return True, "Ollama is running"
            return False, "Ollama is not accessible"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False, "Cannot check Ollama (curl not available or timeout)"
    except Exception as e:
        return False, f"Error checking Ollama: {str(e)}"


def check_ollama_models() -> Tuple[bool, List[str], List[str]]:
    """Check if required Ollama models are installed."""
    required_models = ["llama3.1:latest", "nomic-embed-text"]
    installed_models = []
    missing_models = []
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            installed = [model.get("name", "") for model in data.get("models", [])]
            
            for model in required_models:
                # Check for exact match or version match
                found = False
                for inst in installed:
                    if model in inst or inst.startswith(model.split(":")[0]):
                        found = True
                        installed_models.append(inst)
                        break
                if not found:
                    missing_models.append(model)
            
            return len(missing_models) == 0, installed_models, missing_models
    except Exception as e:
        return False, [], required_models
    
    return False, [], required_models


def run_command(cmd: List[str], description: str, check: bool = True, timeout: int = 300, 
                capture_output: bool = True) -> Tuple[bool, str]:
    """Run a shell command and return success status and output.
    
    Args:
        cmd: Command to run
        description: Description of what the command does
        check: Whether to raise exception on non-zero exit code
        timeout: Timeout in seconds
        capture_output: If False, output is streamed to console (for tqdm compatibility)
    """
    try:
        if capture_output:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=check,
                timeout=timeout
            )
            return True, result.stdout + result.stderr
        else:
            # Stream output directly to console (for tqdm progress bars)
            # Still wait for process to complete and check return code
            result = subprocess.run(
                cmd,
                check=check,
                timeout=timeout
            )
            # Return success based on return code
            return result.returncode == 0, ""
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout} seconds"
    except subprocess.CalledProcessError as e:
        if capture_output:
            return False, f"Command failed with exit code {e.returncode}: {e.stderr}"
        else:
            return False, f"Command failed with exit code {e.returncode}"
    except FileNotFoundError:
        return False, f"Command not found: {cmd[0]}"
    except Exception as e:
        return False, f"Error running command: {str(e)}"


def clean_existing_data():
    """Remove existing agents and results."""
    print_step(1, 8, "Cleaning existing agents and results...")
    
    base_dir = Path(__file__).parent
    agents_dir = base_dir / "agents" / "starbucks_agents"
    results_dir = base_dir / "results"
    
    cleaned = []
    
    # Progress bar for cleaning
    with tqdm(total=2, desc="Cleaning", unit="item", leave=False) as pbar:
        # Remove agents
        if agents_dir.exists():
            try:
                pbar.set_description("Removing agents directory")
                shutil.rmtree(agents_dir)
                cleaned.append(f"Removed {agents_dir}")
                pbar.update(1)
            except Exception as e:
                print_error(f"Failed to remove agents directory: {e}")
                return False
        else:
            pbar.update(1)
        
        # Remove results
        if results_dir.exists():
            pbar.set_description("Removing result files")
            result_files = list(results_dir.glob("*.csv"))
            for result_file in result_files:
                try:
                    result_file.unlink()
                    cleaned.append(f"Removed {result_file}")
                except Exception as e:
                    print_warning(f"Failed to remove {result_file}: {e}")
            pbar.update(1)
        else:
            pbar.update(1)
    
    if cleaned:
        for item in cleaned:
            print_success(item)
    else:
        print_info("No existing data to clean")
    
    return True


def create_settings_file():
    """Create settings.py from example-settings.py if it doesn't exist."""
    print_step(2, 8, "Creating settings.py from example...")
    
    base_dir = Path(__file__).parent
    example_settings = base_dir / "simulation_engine" / "example-settings.py"
    settings_file = base_dir / "simulation_engine" / "settings.py"
    
    with tqdm(total=1, desc="Creating settings", unit="file", leave=False) as pbar:
        if settings_file.exists():
            print_info("settings.py already exists, skipping creation")
            pbar.update(1)
            return True
        
        if not example_settings.exists():
            print_error(f"Example settings file not found: {example_settings}")
            return False
        
        try:
            pbar.set_description("Copying settings file")
            shutil.copy(example_settings, settings_file)
            pbar.update(1)
            print_success(f"Created {settings_file}")
            return True
        except Exception as e:
            print_error(f"Failed to create settings.py: {e}")
            return False


def install_dependencies():
    """Install Python dependencies from requirements.txt."""
    print_step(3, 8, "Installing Python dependencies...")
    
    base_dir = Path(__file__).parent
    requirements = base_dir / "requirements.txt"
    
    if not requirements.exists():
        print_error(f"requirements.txt not found: {requirements}")
        return False
    
    # Show progress bar while installing
    with tqdm(total=1, desc="Installing packages", unit="step", leave=False) as pbar:
        pbar.set_description("Running pip install")
        success, output = run_command(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
            "Installing dependencies"
        )
        pbar.update(1)
    
    if success:
        print_success("Dependencies installed successfully")
        return True
    else:
        print_error(f"Failed to install dependencies: {output}")
        return False


def verify_ollama_setup():
    """Verify Ollama is running and required models are installed."""
    print_step(4, 8, "Verifying Ollama setup...")
    
    # Progress bar for verification steps
    total_steps = 2
    with tqdm(total=total_steps, desc="Verifying Ollama", unit="step", leave=False) as pbar:
        # Check if Ollama is running
        pbar.set_description("Checking Ollama status")
        is_running, msg = check_ollama_running()
        if not is_running:
            print_error(f"Ollama is not running: {msg}")
            print_info("Please start Ollama and ensure it's accessible at http://localhost:11434")
            return False
        
        print_success("Ollama is running")
        pbar.update(1)
        
        # Check if models are installed
        pbar.set_description("Checking models")
        models_ok, installed, missing = check_ollama_models()
        if not models_ok:
            print_warning(f"Missing required models: {', '.join(missing)}")
            print_info("Attempting to pull missing models...")
            
            # Update total for model pulling
            pbar.total = total_steps + len(missing)
            
            for model in missing:
                pbar.set_description(f"Pulling {model}")
                print_info(f"Pulling {model}...")
                success, output = run_command(
                    ["ollama", "pull", model],
                    f"Pulling {model}",
                    check=False
                )
                if success:
                    print_success(f"Successfully pulled {model}")
                else:
                    print_error(f"Failed to pull {model}: {output}")
                    return False
                pbar.update(1)
        else:
            print_success(f"All required models are installed: {', '.join(installed)}")
            pbar.update(1)
    
    return True


def create_agents():
    """Create agents from CSV file."""
    print_step(5, 8, "Creating agents from CSV...")
    
    base_dir = Path(__file__).parent
    os.chdir(base_dir)
    
    csv_path = base_dir / "data" / "satisfaction.csv"
    if not csv_path.exists():
        print_error(f"CSV file not found: {csv_path}")
        return False
    
    # Read CSV to get total number of agents to create
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        total_agents = len(df)
    except Exception:
        total_agents = 122  # Default estimate
    
    # Don't capture output so tqdm progress bars from the script are visible
    print_info("Starting agent creation (this may take several minutes)...")
    success, output = run_command(
        [sys.executable, "create_agents_from_csv.py"],
        "Creating agents",
        check=False,
        capture_output=False,  # Let tqdm progress bars show
        timeout=3600  # 1 hour timeout for agent creation
    )
    
    # Wait a moment for any final file writes to complete
    import time
    time.sleep(1)
    
    # Check if agents were created (even if script had errors)
    agents_dir = base_dir / "agents" / "starbucks_agents"
    if agents_dir.exists():
        agent_count = len(list(agents_dir.glob("agent_*")))
        expected_count = 122  # From CSV
        
        if agent_count == expected_count:
            print_success(f"Created {agent_count} agents")
            print_info("Agent creation step completed successfully")
            return True
        elif agent_count > 0:
            print_warning(f"Only created {agent_count}/{expected_count} agents (some may have failed)")
            print_warning("This may be due to LLM errors. Check the output above for details.")
            # Ask user if they want to continue with partial agents
            response = input(f"Continue with {agent_count} agents? (yes/no): ").strip().lower()
            if response in ['yes', 'y']:
                print_info("Continuing with partial agent set")
                return True
            else:
                print_error("Aborting due to incomplete agent creation")
                return False
        else:
            print_error("No agents were created")
            if not success:
                print_error(f"Script returned non-zero exit code")
            return False
    else:
        print_error("Agents directory was not created")
        if not success:
            print_error(f"Script returned non-zero exit code")
        return False


def run_tests():
    """Run agent testing script."""
    print_step(6, 8, "Running agent tests...")
    
    base_dir = Path(__file__).parent
    os.chdir(base_dir)
    
    test_script = base_dir / "agent_testing" / "test_agents.py"
    if not test_script.exists():
        print_error(f"Test script not found: {test_script}")
        return False
    
    # Don't capture output so tqdm progress bars from the script are visible
    # Use longer timeout for tests (30 minutes)
    print_info("Starting agent tests (this may take 20-30 minutes)...")
    success, output = run_command(
        [sys.executable, str(test_script)],
        "Running tests",
        check=False,
        timeout=1800,  # 30 minute timeout for tests
        capture_output=False  # Let tqdm progress bars show
    )
    
    # Wait a moment for any final file writes to complete
    import time
    time.sleep(1)
    
    # Check if results were generated (even if script had errors)
    results_file = base_dir / "results" / "agent_predictions_dataframe.csv"
    if not results_file.exists():
        # Try agent_testing directory
        results_file = base_dir / "agent_testing" / "agent_predictions_dataframe.csv"
    
    if results_file.exists():
        print_success("Tests completed and results file generated")
        print_info(f"Results saved to: {results_file}")
        return True
    else:
        if success:
            print_warning("Tests completed but results file not found")
        else:
            print_error(f"Tests failed (exit code non-zero) and no results file generated")
        return False


def display_results():
    """Display test results."""
    print_step(7, 8, "Displaying results...")
    
    base_dir = Path(__file__).parent
    # Check both possible locations for results
    results_file = base_dir / "results" / "agent_predictions_dataframe.csv"
    if not results_file.exists():
        # Try agent_testing directory (where test_agents.py saves it)
        results_file = base_dir / "agent_testing" / "agent_predictions_dataframe.csv"
    
    if not results_file.exists():
        print_warning("Results file not found in results/ or agent_testing/")
        return False
    
    with tqdm(total=1, desc="Loading results", unit="file", leave=False) as pbar:
        try:
            import pandas as pd
            pbar.set_description("Reading results file")
            df = pd.read_csv(results_file)
            pbar.update(1)
            
            print_header("TEST RESULTS")
            print(df.to_string(index=False))
            print()
            
            print_info(f"Results saved to: {results_file}")
            # Check both locations for detailed log
            detailed_log = base_dir / "results" / "agent_predictions_detailed_log.csv"
            if not detailed_log.exists():
                detailed_log = base_dir / "agent_testing" / "agent_predictions_detailed_log.csv"
            if detailed_log.exists():
                print_info(f"Detailed log saved to: {detailed_log}")
            
            return True
        except Exception as e:
            print_error(f"Failed to display results: {e}")
            return False


def show_confirmation_prompt() -> bool:
    """Show confirmation prompt with all steps that will be executed."""
    print_header("GENERATIVE AGENTS - COMPLETE SETUP & TEST")
    
    print(f"{Colors.BOLD}This script will perform the following steps:{Colors.ENDC}\n")
    
    steps = [
        ("1. Clean Existing Data", 
         "Remove any existing agents in agents/starbucks_agents/ and results in results/"),
        ("2. Create Settings File", 
         "Create simulation_engine/settings.py from example-settings.py"),
        ("3. Install Dependencies", 
         "Install Python packages from requirements.txt using pip"),
        ("4. Verify Ollama Setup", 
         "Check if Ollama is running and pull required models (llama3.1:latest, nomic-embed-text)"),
        ("5. Create Agents", 
         "Create 122 agents from data/satisfaction.csv (this may take several minutes)"),
        ("6. Run Agent Tests", 
         "Test all agents on held-out questions and calculate accuracy metrics"),
        ("7. Display Results", 
         "Show accuracy, precision, and recall metrics for each test question"),
    ]
    
    for step_name, description in steps:
        print(f"  {Colors.OKCYAN}{step_name}{Colors.ENDC}")
        print(f"    → {description}")
        print()
    
    print(f"{Colors.WARNING}Note: This process may take 30-60 minutes depending on your system.{Colors.ENDC}")
    print(f"{Colors.WARNING}Make sure Ollama is running before proceeding.{Colors.ENDC}\n")
    
    while True:
        response = input(f"{Colors.BOLD}Do you want to proceed? (yes/no): {Colors.ENDC}").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print(f"{Colors.WARNING}Please enter 'yes' or 'no'{Colors.ENDC}")


def main():
    """Main function to run the complete setup and test workflow."""
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Show confirmation prompt
    if not show_confirmation_prompt():
        print_info("Setup cancelled by user")
        return 0
    
    print_header("STARTING SETUP & TEST PROCESS")
    
    # Check prerequisites
    print_step(0, 8, "Checking prerequisites...")
    
    # Check Python version
    py_ok, py_version = check_python_version()
    if py_ok:
        print_success(f"Python version: {py_version}")
    else:
        print_error(f"Python 3.7+ required, found: {py_version}")
        return 1
    
    # Run all steps with overall progress bar
    steps = [
        ("Cleaning existing data", clean_existing_data),
        ("Creating settings file", create_settings_file),
        ("Installing dependencies", install_dependencies),
        ("Verifying Ollama setup", verify_ollama_setup),
        ("Creating agents", create_agents),
        ("Running tests", run_tests),
        ("Displaying results", display_results),
    ]
    
    total_steps = len(steps)
    with tqdm(total=total_steps, desc="Overall progress", unit="step", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as main_pbar:
        for i, (step_name, step_func) in enumerate(steps, start=1):
            try:
                main_pbar.set_description(f"Step {i}/{total_steps}: {step_name}")
                print()  # Add blank line for readability
                success = step_func()
                
                if not success:
                    print_error(f"Step {i} ({step_name}) failed. Aborting.")
                    return 1
                
                # Explicitly confirm step completion
                print_success(f"Step {i} ({step_name}) completed successfully")
                main_pbar.update(1)
                
                # Small delay to ensure any file operations complete
                import time
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                print_warning("\nProcess interrupted by user")
                return 1
            except Exception as e:
                print_error(f"Unexpected error in step {i} ({step_name}): {e}")
                import traceback
                traceback.print_exc()
                return 1
    
    print_header("SETUP & TEST COMPLETE")
    print_success("All steps completed successfully!")
    print_info("You can now interact with agents or run additional tests.")
    print_info("See summary.md for more information on using the project.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

