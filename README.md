# smart-classroom-simulation1

B. Run the Core Simulation
cd smart-classroom-sim
python3 simulation/classroom_sim.py

C. Launch the Interactive Dashboard
streamlit run simulation/dashboard.py

Then open your browser to http://localhost:8501

D. Run Complete Pipeline
python run_simulation.py

E. Hardware-in-the-Loop Verification

Mock stream (no device required):
python validation/hardware_test.py --mock --duration 180 --interval 1

Live serial stream (device connected):
python validation/hardware_test.py --port /dev/ttyUSB0 --baud 115200 --duration 300

Optional: send actuator command strings back to hardware:
python validation/hardware_test.py --port /dev/ttyUSB0 --send-actuation

