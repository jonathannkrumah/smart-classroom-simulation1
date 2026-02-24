# Core SimPy simulation
import simpy
import pandas as pd
import numpy as np
from ml_integration import predict_environment

class ClassroomSimulation:
    def __init__(self, env, num_students=30, room_size=100):
        self.env = env
        self.num_students = num_students
        self.room_size = room_size
        
        # Initial environmental conditions
        self.temperature = 22.0  # °C
        self.co2 = 450  # ppm
        self.humidity = 50  # %
        self.light = 400  # lux
        self.noise = 45  # dB
        
        # Start simulation processes
        self.env.process(self.simulate_occupancy())
        self.env.process(self.simulate_environment_changes())
        self.env.process(self.monitor_and_intervene())
    
    def simulate_occupancy(self):
        """Simulates CO2 increase from student respiration"""
        while True:
            # CO2 increases with occupancy
            co2_production = self.num_students * 0.005  # per minute
            self.co2 += co2_production
            
            # Natural CO2 decay (ventilation)
            self.co2 -= 0.02 * (self.co2 - 450)  # Base decay
            
            yield self.env.timeout(1)  # Update every simulated minute
    
    def simulate_environment_changes(self):
        """Simulates temperature, light, and noise changes"""
        while True:
            # Temperature rise from body heat and lighting
            heat_gain = (self.num_students * 0.1) + (self.light / 1000)
            self.temperature += heat_gain * 0.01
            
            # Natural cooling
            self.temperature -= 0.05 * (self.temperature - 22)
            
            # Random noise fluctuations
            self.noise = 45 + np.random.normal(0, 5)
            
            yield self.env.timeout(1)
    
    def monitor_and_intervene(self):
        """Uses ML model to predict and trigger interventions"""
        while True:
            # Create feature vector for ML model
            features = {
                'temperature': self.temperature,
                'co2': self.co2,
                'humidity': self.humidity,
                'light': self.light,
                'noise': self.noise
            }
            
            # Get prediction from ML model
            prediction, confidence = predict_environment(features)
            
            # Trigger interventions if non-conducive
            if prediction == "non-conducive":
                self.trigger_intervention(features)
            
            # Log current state
            self.log_state(prediction, confidence)
            
            yield self.env.timeout(5)  # Check every 5 simulated minutes
    
    def trigger_intervention(self, features):
        """Triggers simulated IoT interventions"""
        if features['co2'] > 700:
            print(f"[{self.env.now}min] Triggering ventilation - CO2: {features['co2']:.0f}ppm")
            self.co2 -= 50  # Simulated ventilation effect
        
        if features['temperature'] > 26:
            print(f"[{self.env.now}min] Triggering cooling - Temp: {features['temperature']:.1f}°C")
            self.temperature -= 2
        
        if features['light'] < 300:
            print(f"[{self.env.now}min] Increasing lighting - Light: {features['light']:.0f}lux")
            self.light += 100
    
    def log_state(self, prediction, confidence):
        """Logs the current state for visualization"""
        print(f"[{self.env.now}min] Temp: {self.temperature:.1f}°C, "
              f"CO2: {self.co2:.0f}ppm, "
              f"Prediction: {prediction} ({confidence:.1%})")

def run_simulation(hours=8, num_students=30):
    """Runs the classroom simulation"""
    env = simpy.Environment()
    classroom = ClassroomSimulation(env, num_students)
    
    print(f"Starting simulation for {hours} hours with {num_students} students...")
    env.run(until=hours * 60)  # Convert hours to minutes

if __name__ == "__main__":
    run_simulation(hours=2, num_students=25)