# smart-classroom-simulation1

B. Run the Core Simulation
cd smart-classroom-sim
python3 simulation/classroom_sim.py

C. Launch the Interactive Dashboard
streamlit run simulation/dashboard.py

Then open your browser to http://localhost:8501

Dashboard HIL / Live Testbed (Windows Arduino setup):
- Serial Port: COM7
- Baud Rate: 9700
- If you accidentally type COMP7, the dashboard and HIL script normalize it to COM7.

D. Run Complete Pipeline
python run_simulation.py

E. Hardware-in-the-Loop Verification

Mock stream (no device required):
python validation/hardware_test.py --mock --duration 180 --interval 1

Live serial stream (device connected):
python validation/hardware_test.py --port COM7 --baud 9700 --duration 300

Optional: send actuator command strings back to hardware:
python validation/hardware_test.py --port COM7 --baud 9700 --send-actuation

