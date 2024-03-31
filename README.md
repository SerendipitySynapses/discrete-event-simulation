# discrete_event_simulation-
This Python script simulates the operations of a healthcare system using discrete event simulation techniques.

## Overview
This project simulates the operations of a clinic managed by a series of doctors and medical students over multiple shifts. The simulation models the treatment process, patient queueing and prioritization, doctor shifts, and breaks to provide insights into clinic efficiency, patient wait times, and resource utilization. 

## Simulation Details
The simulation is structured around the following key components:
- **Patient Arrival**: Simulated based on an exponential distribution with a mean of 21 minutes.
- **Priority System**: Patients are initially triaged into three priority levels, affecting their treatment order and times.
- **Service Times**: Defined by triangular and uniform distributions, varying by the patient's priority and the phase of treatment.
- **Doctor Shifts**: Two sets of doctors work in 8-hour shifts,ensuring 24/7 clinic operation
**Scheduled break and Workload**: Doctors take scheduled breaks, which can be delayed depending on their workload, ensuring 24/7 clinic operation.


## Objectives
- Implemented simulation model of clinic operations with a focus on emergency and regular patient treatment processes.
- Implementation of patient prioritization and queuing mechanisms based on the severity of conditions.
- Evaluation specific operational key performance indicators (KPIs) include average system time for different priority patients, percentage of priority 3 patients not waiting,average patient’s time spent, average queue lengths, area under queue length curve,doctor utilization, and other defined metrics required by clinic management over a 20-day simulation period with at least 100 repetitions to guide management decisions.
- Provide static and dynamic descriptions of the system, including a flowchart of all mentioned events.
- Structure the future event list (FEL) for the simulation, specifying initial events and their representation.
- Provide an Excel output detailing steps, current events, state variables, and cumulative statistics.
- Analysis of the impact of removing doctor rest times on clinic efficiency and compare the current system with an alternative using final-year medical students instead of professional doctors based on the average system time and other evaluation metrics.
-perform sensitivity analysis to identify the most impactful factors on patient wait times and overall system efficiency in a healthcare simulation model.
- Perform cold and warm analysis to determine long-term average patient system time.
- Suggest improvements to the clinic's operation without the need for simulation of these policies.

## Results
- **The sensitivity analysis** aimed to identify which factors most significantly impact patient wait times and treatment efficiency in a healthcare setting. By varying key inputs within our simulation model, we assessed their influence on the overall system performance. The parameters analyzed included the number of staff (both professional doctors and medical students), patient arrival rates, and service times.
- **The cold and warm analysis** for the project is illustrated in the two graphs, highlighting the system's behavior during the initial phase (cold) and after reaching equilibrium (warm). The "Number of Finishing Customers" graph shows a steep initial change and then stabilizes, suggesting a significant shift from the cold start to a steady operational state (warm) after a brief period. The "Aggregate Waiting Time" graph exhibits a similar pattern with a sharp decrease before leveling out. These patterns indicate that while the system takes time to warm up and reach its full operational efficiency, once it does, it maintains a consistent performance level. The warm-up period is crucial for accurate performance assessment, excluding initial volatility due to cold starts.
- The simulation showed that using three medical students (average treatment time of 9.9 minutes per patient) is more efficient than two experienced doctors (14.9 minutes per patient), suggesting staffing adjustments could reduce wait times and improve patient satisfaction. However, considerations such as care quality and the need for supervision must be balanced. Recommendations will focus on strategic staffing to optimize operations and ensure quality care.

### Running the Simulation
1. Clone this repository to your local machine.
2. Install required dependencies: `pip install -r requirements.txt`.
3. Execute the simulation script: `python simulation.py`.
   
## Conclusion
This project aims to provide a comprehensive analysis of a clinic's operation, focusing on patient wait times, staff utilization, and overall system performance. Through detailed simulation and analysis, it seeks to offer actionable insights and recommendations to improve clinic operations and patient satisfaction.
