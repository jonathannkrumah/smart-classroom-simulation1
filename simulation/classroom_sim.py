"""
Core SimPy simulation for smart classroom environment
"""

import simpy
import numpy as np
from datetime import datetime
from simulation.ml_integration import predict_environment

class ClassroomSimulation:
    """
    A discrete-event simulation of a classroom environment
    using SimPy for event-based modeling
    """
    
    def __init__(self, env, num_students=30, room_size=100):
        self.env = env
        self.num_students = num_students
        self.room_size = room_size
        
        # Initial environmental conditions
        self.temperature = 22.0  # °C
        self.co2 = 450  # ppm (baseline outdoor level)
        self.humidity = 50  # %
        self.light = 450 + np.random.normal(0, 50)  # lux
        self.noise = 45  # dB
        
        # Logging
        self.log = []
        
        # Start simulation processes
        self.env.process(self.simulate_occupancy())
        self.env.process(self.simulate_environment_changes())
        self.env.process(self.monitor_and_intervene())
    
    def simulate_occupancy(self):
        """
        Simulates the effect of student occupancy on CO2 levels
        Based on respiration rates: ~0.005 ppm per student per minute
        """
        while True:
            # CO2 increases with occupancy
            # Each student produces CO2
            co2_production = self.num_students * 0.008  # per minute
            
            # Natural CO2 decay from ventilation
            # Simple model: decay proportional to difference from baseline
            baseline_co2 = 450
            decay_rate = 0.02
            co2_decay = decay_rate * (self.co2 - baseline_co2)
            
            # Update CO2
            self.co2 += co2_production - co2_decay
            
            yield self.env.timeout(1)  # Update every simulated minute
    
    def simulate_environment_changes(self):
        """
        Simulates changes in temperature, light, and noise
        """
        while True:
            # Temperature changes
            # Heat from students and equipment
            heat_gain = (self.num_students * 0.1) + (self.light / 1000)
            self.temperature += heat_gain * 0.01
            
            # Natural cooling (simplified)
            self.temperature -= 0.05 * (self.temperature - 22)
            
            # Humidity changes (simplified)
            # Increases with occupancy, decreases with ventilation
            self.humidity += (self.num_students * 0.02) - 0.1
            
            # Noise level - varies with activity
            base_noise = 45
            activity_noise = np.random.normal(0, 8)  # Random variation
            self.noise = max(35, min(80, base_noise + activity_noise))
            
            # Light - natural variation (simulating time of day)
            hour = (self.env.now / 60) % 24
            if 6 <= hour <= 18:  # Daytime
                self.light = 400 + 200 * np.sin((hour - 6) * np.pi / 12)
            else:  # Night
                self.light = 100
            
            yield self.env.timeout(5)  # Update every 5 minutes
    
    def monitor_and_intervene(self):
        """
        Uses ML model to predict learning conditions
        and trigger simulated interventions
        """
        while True:
            # Current environmental state
            features = {
                'temperature': self.temperature,
                'co2': self.co2,
                'humidity': self.humidity,
                'light': self.light,
                'noise': self.noise
            }
            
            # Get ML model prediction
            try:
                prediction, confidence = predict_environment(features)
            except:
                # Fallback if model not loaded
                prediction = "conducive" if self.co2 < 1000 else "non-conducive"
                confidence = 0.85
            
            # Log current state
            self.log_state(features, prediction, confidence)
            
            # Trigger interventions if needed
            if prediction == "non-conducive" or confidence < 0.6:
                self.trigger_intervention(features)
            
            yield self.env.timeout(10)  # Check every 10 minutes
    
    def trigger_intervention(self, features):
        """
        Simulates IoT actuator responses to poor conditions
        """
        interventions = []
        
        # CO2 intervention (ventilation)
        if features['co2'] > 800:
            self.co2 -= 100
            interventions.append(f"Ventilation ON (CO2: {features['co2']:.0f}ppm)")
        
        # Temperature intervention (cooling)
        if features['temperature'] > 26:
            self.temperature -= 1.5
            interventions.append(f"Cooling ON (Temp: {features['temperature']:.1f}°C)")
        elif features['temperature'] < 18:
            self.temperature += 1.5
            interventions.append(f"Heating ON (Temp: {features['temperature']:.1f}°C)")
        
        # Light intervention
        if features['light'] < 250:
            self.light += 150
            interventions.append(f"Lights ON (Light: {features['light']:.0f}lux)")
        elif features['light'] > 800:
            self.light -= 150
            interventions.append(f"Blinds adjusted (Light: {features['light']:.0f}lux)")
        
        # Noise intervention
        if features['noise'] > 65:
            interventions.append(f"Warning: High noise level ({features['noise']:.0f}dB)")
        
        if interventions:
            timestamp = self.env.now
            print(f"\n[{timestamp}min] INTERVENTIONS TRIGGERED:")
            for i in interventions:
                print(f"  • {i}")
    
    def log_state(self, features, prediction, confidence):
        """
        Logs the current environmental state
        """
        timestamp = self.env.now
        log_entry = {
            'time': timestamp,
            'temperature': features['temperature'],
            'co2': features['co2'],
            'humidity': features['humidity'],
            'light': features['light'],
            'noise': features['noise'],
            'prediction': prediction,
            'confidence': confidence
        }
        self.log.append(log_entry)
        
        # Print periodic updates (every 30 minutes)
        if timestamp % 30 == 0:
            status_icon = "✅" if prediction == "conducive" else "⚠️"
            print(f"[{timestamp:3d}min] {status_icon} Temp:{features['temperature']:5.1f}°C "
                  f"CO2:{features['co2']:6.0f}ppm Light:{features['light']:5.0f}lux "
                  f"Noise:{features['noise']:5.1f}dB | {prediction} ({confidence:.1%})")

def run_simulation(hours=2, num_students=30):
    """
    Runs the classroom simulation
    """
    # Create simulation environment
    env = simpy.Environment()
    
    # Create classroom instance
    classroom = ClassroomSimulation(env, num_students=num_students)
    
    print(f"\n🏫 SIMULATION STARTED")
    print(f"   Duration: {hours} hours ({hours*60} minutes)")
    print(f"   Students: {num_students}")
    print(f"   Initial CO2: {classroom.co2}ppm")
    print(f"   Initial Temp: {classroom.temperature}°C")
    print("-" * 70)
    
    # Run simulation
    env.run(until=hours * 60)
    
    print("-" * 70)
    print(f"\n📊 SIMULATION SUMMARY")
    print(f"   Final CO2: {classroom.co2:.0f}ppm")
    print(f"   Final Temp: {classroom.temperature:.1f}°C")
    print(f"   Total logs: {len(classroom.log)}")
    
    # Calculate statistics
    conducive_count = sum(1 for entry in classroom.log if entry['prediction'] == 'conducive')
    if classroom.log:
        conducive_percent = (conducive_count / len(classroom.log)) * 100
        print(f"   Time conducive: {conducive_percent:.1f}%")
    
    return classroom.log