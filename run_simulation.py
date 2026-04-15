#!/usr/bin/env python3
"""
Main entry point for the Smart Classroom Simulation
Run this file to execute the complete simulation pipeline
"""

import os
import sys
import argparse
from simulation.classroom_sim import run_simulation
from simulation.ml_integration import test_model_prediction

def main():
    parser = argparse.ArgumentParser(description='Smart Classroom Simulation')
    parser.add_argument('--hours', type=int, default=2, help='Simulation duration in hours')
    parser.add_argument('--students', type=int, default=30, help='Number of students')
    parser.add_argument('--room-size', type=int, default=100, help='Room size in square meters')
    parser.add_argument('--start-hour', type=int, default=9, help='Starting classroom hour (0-23)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--test', action='store_true', help='Run ML model test only')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SMART CLASSROOM SIMULATION FRAMEWORK")
    print("=" * 60)
    print(f"Duration: {args.hours} hours")
    print(f"Students: {args.students}")
    print(f"Room size: {args.room_size} m²")
    print(f"Start hour: {args.start_hour:02d}:00")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    if args.test:
        print("\nTesting ML Model Predictions...")
        test_model_prediction()
    else:
        print("\nStarting Simulation...\n")
        run_simulation(
            hours=args.hours,
            num_students=args.students,
            room_size=args.room_size,
            start_hour=args.start_hour,
            random_seed=args.seed,
        )
    
    print("\n" + "=" * 60)
    print("Simulation Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()