#!/usr/bin/env python3
"""
Main runner script for the ODE Solver and Beautiful Graphs project
"""

import sys
import os
import subprocess

# This will automatically find the correct python from your (venv)
python_executable = sys.executable

def show_menu():
    """Display the main menu"""
    print("���� ODE Solver & Beautiful Graphs Toolkit")
    print("=" * 50)
    print("Choose what you'd like to run:")
    print()
    print("1. 🎨 Beautiful Graphs Demo")
    print("2. 📊 Equation Grapher (Interactive)")
    print("3. 🧮 Advanced ODE Solver (Interactive)")
    print("4. 📈 Advanced Equation Grapher")
    print("5. 🏠 Room Thermal Model Simulation (Newton's Cooling)")
    print("6. 🎯 Quick Examples")
    print("7. 📚 ODE Examples Library")
    print("8. 🔬 Advanced Control Strategies")
    print("9. 🏢 Multi-Zone Building Control")
    print("10. ✅ ODE Accuracy Tests")
    print()
    print("0. Exit")
    print("=" * 50)

def run_script(script_name):
    """Run a specific script using a reliable method"""
    try:
        script_path = os.path.join('scripts', script_name)
        if os.path.exists(script_path):
            print(f"\n🚀 Running {script_name}...")
            print("-" * 30)
            # This method is much more reliable than the old one
            subprocess.run([python_executable, script_path], check=True)
        else:
            print(f"❌ Script {script_name} not found!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Script {script_name} failed with an error.")
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")

def main():
    """Main function"""
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (0-10): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                run_script('beautiful_graphs.py')
            elif choice == '2':
                run_script('equation_grapher.py')
            elif choice == '3':
                run_script('ode_solver.py')
            elif choice == '4':
                run_script('advanced_grapher.py')
            elif choice == '5':
                run_script('room_thermal_model_pid.py')
            elif choice == '6':
                run_script('quick_examples.py')
            elif choice == '7':
                run_script('ode_examples_library.py')
            elif choice == '8':
                run_script('advanced_control_strategies.py')
            elif choice == '9':
                run_script('multi_zone_coordination.py')
            elif choice == '10':
                run_script('advanced_ode_examples.py')
            else:
                print("❌ Invalid choice! Please enter 0-10.")
            
            input("\nPress Enter to continue...")
            print("\n" * 2)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()