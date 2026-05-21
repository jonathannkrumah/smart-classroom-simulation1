# smart-classroom-simulation1

## Control Logic

This simulator uses a dual-zone decision framework:

- Comfort zone is the hard safety layer. Any comfort violation triggers intervention immediately.
- Attention zone is the soft optimization layer. If conditions stay within comfort but the model predicts non-conducive behavior, the system waits for persistence before acting.
- A cooldown-based anti-flapping mechanism reduces repeated actuator toggling.
- The simulation logs agreement score, attention-drift streak, and total actuator activations for evaluation.

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

If your device is on COM6:
python validation/hardware_test.py --port COM6 --baud 9600 --duration 300

If no data is read, try the common Arduino baud:
python validation/hardware_test.py --port COM6 --baud 9600 --duration 300

Note: `hardware_test.py` now applies refined calibration by default (same as dashboard).
Use raw values only when needed:
python validation/hardware_test.py --port COM6 --baud 9700 --duration 300 --no-calibration

Optional: send actuator command strings back to hardware:
python validation/hardware_test.py --port COM7 --baud 9700 --send-actuation

F. Feed Testbed Data Into Simulation Logic

Use the Arduino request-response bridge (Python sends timestamp, Arduino returns one CSV row):
python validation/testbed_simulation_bridge_v2.py --port COM7 --baud 9600 --duration 300 --interval 1 --forward-live --reset-live-feed

Output CSV (Excel-ready) default:
validation/testbed_simulation_bridge_v2.csv

Optional live feed CSV for dashboard:
validation/live_bridge_feed.csv

