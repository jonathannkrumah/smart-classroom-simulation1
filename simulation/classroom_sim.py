"""
Core SimPy simulation for smart classroom environment
"""

import simpy
import numpy as np
from datetime import datetime
from simulation.ml_integration import predict_environment


DEFAULT_SIM_CONFIG = {
    'co2_production_per_student': 0.008,
    'co2_decay_rate': 0.02,
    'baseline_co2': 450,
    'heat_gain_per_student': 0.1,
    'light_heat_factor': 1 / 1000,
    'temperature_heat_scale': 0.01,
    'temperature_cooling_coeff': 0.05,
    'temperature_baseline': 22,
    'humidity_gain_per_student': 0.02,
    'humidity_vent_loss': 0.1,
    'light_day_base': 400,
    'light_day_amp': 200,
    'light_night_level': 100,
    'light_min': 100,
    'light_max': 1000,
    'occupancy_update_minutes': 1,
    'environment_update_minutes': 5,
    'monitor_interval_minutes': 10,
}

# Three-zone thresholds (attention-first policy)
ATTENTION_THRESHOLDS = {
    'temperature': {'low': 21.0, 'high': 25.0},
    'humidity': {'low': 40.0, 'high': 60.0},
    'co2': {'high': 800.0},
    'light': {'low': 500.0, 'high': 650.0},
}

COMFORT_THRESHOLDS = {
    'temperature': {'low': 20.0, 'high': 27.0},
    'humidity': {'low': 30.0, 'high': 60.0},
    'co2': {'high': 1000.0},
    'light': {'low': 300.0, 'high': 500.0},
}


def evaluate_features_zone(features):
    """Classify each factor and overall environment into optimal/acceptable/non-conducive zones."""
    factor_zones = {}
    acceptable_factors = []
    non_conducive_factors = []

    for factor in ('temperature', 'humidity', 'co2', 'light'):
        value = float(features[factor])
        att = ATTENTION_THRESHOLDS[factor]
        comfort = COMFORT_THRESHOLDS[factor]

        att_low = att.get('low', -np.inf)
        att_high = att.get('high', np.inf)
        comfort_low = comfort.get('low', -np.inf)
        comfort_high = comfort.get('high', np.inf)

        in_attention = att_low <= value <= att_high
        in_comfort = comfort_low <= value <= comfort_high

        if in_attention:
            zone = 'optimal'
        elif in_comfort:
            zone = 'acceptable'
        else:
            zone = 'non-conducive'

        factor_zones[factor] = zone

        if zone == 'acceptable':
            if factor == 'co2':
                acceptable_factors.append(
                    f"CO₂ above attention target ({value:.0f}ppm > {att_high:.0f}ppm)"
                )
            elif value < att_low:
                acceptable_factors.append(
                    f"{factor.capitalize()} below attention range ({value:.1f} < {att_low:.1f})"
                )
            else:
                acceptable_factors.append(
                    f"{factor.capitalize()} above attention range ({value:.1f} > {att_high:.1f})"
                )

        if zone == 'non-conducive':
            if factor == 'co2':
                non_conducive_factors.append(
                    f"CO₂ above comfort limit ({value:.0f}ppm > {comfort_high:.0f}ppm)"
                )
            elif value < comfort_low:
                non_conducive_factors.append(
                    f"{factor.capitalize()} below comfort limit ({value:.1f} < {comfort_low:.1f})"
                )
            else:
                non_conducive_factors.append(
                    f"{factor.capitalize()} above comfort limit ({value:.1f} > {comfort_high:.1f})"
                )

    if non_conducive_factors:
        overall_zone = 'non-conducive'
    elif acceptable_factors:
        overall_zone = 'acceptable'
    else:
        overall_zone = 'optimal'

    return {
        'overall_zone': overall_zone,
        'factor_zones': factor_zones,
        'acceptable_factors': acceptable_factors,
        'non_conducive_factors': non_conducive_factors,
    }


def fuse_model_zone_status(model_prediction, zone_state, confidence, low_confidence=0.6):
    """
    Unified final decision policy.

    Returns dict with:
      - final_status: conducive / acceptable / non-conducive
      - disagreement: True when model and zone imply different risk levels
      - rationale: concise reason string
    """
    overall_zone = zone_state.get('overall_zone', 'optimal')
    model_non_conducive = str(model_prediction) == 'non-conducive'

    # Hard safety gate from threshold policy
    if overall_zone == 'non-conducive':
        return {
            'final_status': 'non-conducive',
            'disagreement': not model_non_conducive,
            'rationale': 'Zone safety override (outside comfort limits)',
        }

    # Model risk signal can elevate to non-conducive
    if model_non_conducive and float(confidence) >= low_confidence:
        return {
            'final_status': 'non-conducive',
            'disagreement': overall_zone != 'non-conducive',
            'rationale': 'Model risk alert (high-confidence non-conducive)',
        }

    if overall_zone == 'acceptable':
        return {
            'final_status': 'acceptable',
            'disagreement': model_non_conducive,
            'rationale': 'Within comfort, outside attention range',
        }

    return {
        'final_status': 'conducive',
        'disagreement': model_non_conducive,
        'rationale': 'Model and zone both favorable',
    }

class ClassroomSimulation:
    """
    A discrete-event simulation of a classroom environment
    using SimPy for event-based modeling
    """
    
    def __init__(
        self,
        env,
        num_students=30,
        room_size=100,
        start_hour=9,
        initial_conditions=None,
        sim_config=None,
        random_seed=None,
    ):
        self.env = env
        self.num_students = num_students
        self.room_size = room_size
        self.start_hour = start_hour

        self.sim_config = {**DEFAULT_SIM_CONFIG, **(sim_config or {})}
        if random_seed is not None:
            np.random.seed(int(random_seed))

        initial = initial_conditions or {}
        
        # Initial environmental conditions
        self.temperature = float(initial.get('temperature', self.sim_config['temperature_baseline']))  # °C
        self.co2 = float(initial.get('co2', self.sim_config['baseline_co2']))  # ppm
        self.humidity = float(initial.get('humidity', 50))  # %
        self.light = float(initial.get('light', 450 + np.random.normal(0, 50)))  # lux
        self.artificial_light_offset = 0.0
        self.acceptable_zone_streak = 0
        
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
            co2_production = self.num_students * self.sim_config['co2_production_per_student']  # per minute
            
            # Natural CO2 decay from ventilation
            # Simple model: decay proportional to difference from baseline
            baseline_co2 = self.sim_config['baseline_co2']
            decay_rate = self.sim_config['co2_decay_rate']
            co2_decay = decay_rate * (self.co2 - baseline_co2)
            
            # Update CO2
            self.co2 += co2_production - co2_decay
            
            yield self.env.timeout(self.sim_config['occupancy_update_minutes'])
    
    def simulate_environment_changes(self):
        """
        Simulates changes in temperature, light, and noise
        """
        while True:
            # Temperature changes
            # Heat from students and equipment
            heat_gain = (self.num_students * self.sim_config['heat_gain_per_student']) + (
                self.light * self.sim_config['light_heat_factor']
            )
            self.temperature += heat_gain * self.sim_config['temperature_heat_scale']
            
            # Natural cooling (simplified)
            self.temperature -= self.sim_config['temperature_cooling_coeff'] * (
                self.temperature - self.sim_config['temperature_baseline']
            )
            
            # Humidity changes (simplified)
            # Increases with occupancy, decreases with ventilation
            self.humidity += (self.num_students * self.sim_config['humidity_gain_per_student']) - self.sim_config['humidity_vent_loss']
            
            # Light - natural variation (simulating time of day)
            hour = (self.start_hour + (self.env.now / 60)) % 24
            if 6 <= hour <= 18:  # Daytime
                natural_light = self.sim_config['light_day_base'] + self.sim_config['light_day_amp'] * np.sin((hour - 6) * np.pi / 12)
            else:  # Night
                natural_light = self.sim_config['light_night_level']

            # Apply persistent artificial lighting interventions
            self.light = max(
                self.sim_config['light_min'],
                min(self.sim_config['light_max'], natural_light + self.artificial_light_offset)
            )
            
            yield self.env.timeout(self.sim_config['environment_update_minutes'])
    
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
                'occupancy': self.num_students,
                'occupancy_count': self.num_students,
            }
            
            # Get ML model prediction
            try:
                prediction, confidence = predict_environment(
                    features,
                    context={
                        'current_minute': self.env.now,
                        'start_hour': self.start_hour,
                        'room_size': self.room_size,
                    }
                )
            except:
                # Fallback if model not loaded
                prediction = "conducive" if self.co2 < 1000 else "non-conducive"
                confidence = 0.85

            zone_state = self.evaluate_environment_zone(features)
            overall_zone = zone_state['overall_zone']
            fused = fuse_model_zone_status(prediction, zone_state, confidence)
            final_status = fused['final_status']
            decision_rationale = fused['rationale']
            disagreement = fused['disagreement']

            if overall_zone == 'acceptable':
                self.acceptable_zone_streak += 1
            else:
                self.acceptable_zone_streak = 0

            interventions = []
            trigger_reason = ""
            
            # Trigger interventions using three-zone policy
            if overall_zone == 'non-conducive':
                trigger_reason = "Immediate: non-conducive zone"
                interventions = self.trigger_intervention(features)
            elif overall_zone == 'acceptable':
                acceptable_factor_count = len(zone_state['acceptable_factors'])
                if self.acceptable_zone_streak >= 2:
                    trigger_reason = "Delayed: acceptable zone persisted"
                    interventions = self.trigger_intervention(features)
                elif confidence < 0.6:
                    trigger_reason = "Precaution: acceptable zone + low confidence"
                    interventions = self.trigger_intervention(features)
                elif acceptable_factor_count >= 2:
                    trigger_reason = "Precaution: multiple acceptable drifts"
                    interventions = self.trigger_intervention(features)

            # Log current state
            self.log_state(
                features,
                prediction,
                confidence,
                zone_state,
                interventions,
                trigger_reason,
                final_status=final_status,
                disagreement=disagreement,
                decision_rationale=decision_rationale,
            )
            
            yield self.env.timeout(self.sim_config['monitor_interval_minutes'])

    def evaluate_environment_zone(self, features):
        return evaluate_features_zone(features)
    
    def trigger_intervention(self, features):
        """
        Simulates IoT actuator responses to poor conditions
        """
        interventions = []
        
        # CO2 intervention (ventilation)
        if features['co2'] > COMFORT_THRESHOLDS['co2']['high']:
            self.co2 -= 150
            interventions.append(f"Ventilation HIGH (CO2: {features['co2']:.0f}ppm)")
        elif features['co2'] > ATTENTION_THRESHOLDS['co2']['high']:
            self.co2 -= 100
            interventions.append(f"Ventilation LOW (CO2: {features['co2']:.0f}ppm)")
        
        # Temperature intervention (cooling)
        if features['temperature'] > COMFORT_THRESHOLDS['temperature']['high']:
            self.temperature -= 1.5
            interventions.append(f"Cooling HIGH (Temp: {features['temperature']:.1f}°C)")
        elif features['temperature'] > ATTENTION_THRESHOLDS['temperature']['high']:
            self.temperature -= 0.6
            interventions.append(f"Cooling LOW (Temp: {features['temperature']:.1f}°C)")
        elif features['temperature'] < COMFORT_THRESHOLDS['temperature']['low']:
            self.temperature += 1.5
            interventions.append(f"Heating HIGH (Temp: {features['temperature']:.1f}°C)")
        elif features['temperature'] < ATTENTION_THRESHOLDS['temperature']['low']:
            self.temperature += 0.6
            interventions.append(f"Heating LOW (Temp: {features['temperature']:.1f}°C)")

        # Humidity intervention
        if features['humidity'] > COMFORT_THRESHOLDS['humidity']['high']:
            self.humidity -= 4
            interventions.append(f"Dehumidifier ON (Humidity: {features['humidity']:.1f}%)")
        elif features['humidity'] < ATTENTION_THRESHOLDS['humidity']['low']:
            self.humidity += 4
            interventions.append(f"Humidifier ON (Humidity: {features['humidity']:.1f}%)")
        
        # Light intervention
        if features['light'] < ATTENTION_THRESHOLDS['light']['low']:
            boost = 180 if features['light'] < COMFORT_THRESHOLDS['light']['low'] else 80
            self.artificial_light_offset = min(500, self.artificial_light_offset + boost)
            interventions.append(
                f"Lights ON (Light: {features['light']:.0f}lux, boost: +{self.artificial_light_offset:.0f})"
            )
        elif features['light'] > ATTENTION_THRESHOLDS['light']['high']:
            reduction = 120 if features['light'] > COMFORT_THRESHOLDS['light']['high'] else 60
            self.artificial_light_offset = max(-300, self.artificial_light_offset - reduction)
            interventions.append(
                f"Blinds adjusted (Light: {features['light']:.0f}lux, boost: +{self.artificial_light_offset:.0f})"
            )
        
        if interventions:
            timestamp = self.env.now
            print(f"\n[{timestamp}min] INTERVENTIONS TRIGGERED:")
            for i in interventions:
                print(f"  • {i}")

        return interventions
    
    def log_state(
        self,
        features,
        prediction,
        confidence,
        zone_state=None,
        interventions=None,
        trigger_reason="",
        final_status=None,
        disagreement=False,
        decision_rationale="",
    ):
        """
        Logs the current environmental state
        """
        timestamp = self.env.now
        zone_state = zone_state or {
            'overall_zone': 'optimal',
            'acceptable_factors': [],
            'non_conducive_factors': [],
        }
        non_conducive_factors = zone_state.get('non_conducive_factors', [])
        acceptable_factors = zone_state.get('acceptable_factors', [])
        interventions = interventions or []
        log_entry = {
            'time': timestamp,
            'temperature': features['temperature'],
            'co2': features['co2'],
            'humidity': features['humidity'],
            'light': features['light'],
            'prediction': prediction,
            'model_prediction': prediction,
            'confidence': confidence,
            'final_status': final_status or prediction,
            'model_zone_disagreement': bool(disagreement),
            'decision_rationale': decision_rationale,
            'overall_zone': zone_state.get('overall_zone', 'optimal'),
            'acceptable_factors': '; '.join(acceptable_factors),
            'non_conducive_factors': '; '.join(non_conducive_factors),
            'interventions_triggered': '; '.join(interventions),
            'intervention_count': len(interventions),
            'zone_trigger_reason': trigger_reason,
        }
        self.log.append(log_entry)
        
        # Print periodic updates (every 30 minutes)
        if timestamp % 30 == 0:
            final_status = final_status or prediction
            status_icon = "✅" if final_status == "conducive" else "⚠️"
            overall_zone = zone_state.get('overall_zone', 'optimal')
            factor_note = f" | cause: {', '.join(non_conducive_factors)}" if non_conducive_factors else ""
            print(f"[{timestamp:3d}min] {status_icon} Temp:{features['temperature']:5.1f}°C "
                  f"CO2:{features['co2']:6.0f}ppm Hum:{features['humidity']:5.1f}% "
                  f"Light:{features['light']:5.0f}lux "
                f"| model:{prediction} final:{final_status} ({confidence:.1%}) | zone:{overall_zone}{factor_note}")

def run_simulation(
    hours=2,
    num_students=30,
    start_hour=9,
    room_size=100,
    initial_conditions=None,
    sim_config=None,
    random_seed=None,
):
    """
    Runs the classroom simulation
    """
    # Create simulation environment
    env = simpy.Environment()
    
    # Create classroom instance
    classroom = ClassroomSimulation(
        env,
        num_students=num_students,
        room_size=room_size,
        start_hour=start_hour,
        initial_conditions=initial_conditions,
        sim_config=sim_config,
        random_seed=random_seed,
    )
    
    print(f"\n🏫 SIMULATION STARTED")
    print(f"   Duration: {hours} hours ({hours*60} minutes)")
    print(f"   Students: {num_students}")
    print(f"   Room size: {room_size} m²")
    print(f"   Start hour: {start_hour:02d}:00")
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
    conducive_count = sum(
        1 for entry in classroom.log
        if entry.get('final_status', entry.get('prediction')) == 'conducive'
    )
    if classroom.log:
        conducive_percent = (conducive_count / len(classroom.log)) * 100
        print(f"   Time conducive: {conducive_percent:.1f}%")
    
    return classroom.log