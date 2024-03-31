import random
import math
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl


# Simulation of hospital
# Doctors don't rest
#
# Input Distributions:
#     1- Inter-arrival time: Exponential with mean = 21 minutes
#     2- Service time for patient with Priority 3: triangular(15,60,105)
#     3- Service time for patient with Priority 1:uniform(10,30)
#     4- Service time for patient with Priority 2 was with priority 3:triangular(10,18,26)
#     5- Service time for patient with Priority 2 was with priority 1:uniform(9,20)
#     statistics:
#     Average Time Spent in the system
#     Server Utilization

def starting_state():
    """
    Initializes the simulation state and data.

    Returns:
        state (dict): Contains the initial state of the simulation, including queue lengths and server status.
        data (dict): Stores simulation-wide data such as patient information and cumulative statistics.
        future_event_list (list): A list of events scheduled to occur, initialized with the first patient arrival and a rest event.
    """
    # State variables to track the lengths of queues for each priority and the server's status
    state = {
        'priority 1 Queue Length': 0,
        'priority 2 Queue Length': 0,
        'priority 3 Queue Length': 0,
        'server Status': 2  # Assuming 2 indicates an idle server ready to serve
    }
    
    # Data storage for various metrics and patient queue tracking
    data = {
        'patients': {},  # Stores individual patient data
        'Last Time Queue1 Length Changed': 0,
        'Last Time Queue2 Length Changed': 0,
        'Last Time Queue3 Length Changed': 0,
        'max1': 0,  # Maximum length reached by Queue 1
        'max2': 0,  # Maximum length reached by Queue 2
        'max3': 0,  # Maximum length reached by Queue 3
        'Queue1': {},
        'Queue2': {},
        'Queue3': {},
        'Cumulative Stats': {  # Stores cumulative statistics for analysis
            'Server Busy Time': 0,
            'priority1': 0,
            'priority2': 0,
            'priority3': 0,
            'Area Under Queue1 Length Curve': 0,
            'Area Under Queue2 Length Curve': 0,
            'Area Under Queue3 Length Curve': 0,
            'priority1 time spent': 0,
            'priority2 time spent': 0,
            'priority3 no expect': 0,
            'number of patients' : 0,
            'time spent': 0
        }
    }

    # Initializing the future event list with the first patient arrival and a predetermined rest event
    future_event_list = [
        {'Event Type': 'Arrival_1', 'Event Time': 0, 'Patient': 'C11', 'restflag': 0, 'k': 0},
        {'Event Type': 'rest_time', 'Event Time': 300, 'Patient': None, 'restflag': 0, 'k': 0}
    ]
    return state, data, future_event_list

def exponential(lambd):
    rn = random.random()
    return -(1 / lambd) * math.log(rn)
def uniform(a, b):
    rn = random.random()
    return a + ((b - a) * rn)
def triangular(a,b,c):
    rn=random.random()
    f=(c-a)/(b-a)
    if 0<rn<f:
        return a + math.sqrt(rn*(b-a)*(c-a))
    elif f<=rn<1:
        return b - math.sqrt((1-rn)*(b-a)*(b-c))

def fel_maker(future_event_list, event_type, clock,patient):

    event_time = 0
    if event_type == 'Arrival_1':
        event_time = clock + exponential(1/20)
    elif event_type == 'Arrival_3':
        event_time = clock + exponential(1/20)

    elif event_type == 'Arrival_2':
        if patient[1]=='1':
            event_time = clock + uniform(10,30)
        elif patient[1]=='3':
            event_time = clock + triangular(15,60,105)

    elif event_type == 'End of Service':
        if patient[1] == '1':
            event_time = clock + uniform(9,20)
        elif patient[1] == '3':
            event_time = clock + triangular(10,18,26)
    new_event = {'Event Type': event_type, 'Event Time': event_time,'Patient': patient}
    future_event_list.append(new_event)

def arrival_1(future_event_list, state, clock,patient,data):
    data['Cumulative Stats']['priority1'] += 1
    if clock >= 7200:
        data['Cumulative Stats']['number of patients'] +=1
    data['patients'][patient] = dict()
    data['patients'][patient]['Arrival Time'] = clock
    if state['server Status']==0:  #busy
        data['Cumulative Stats']['Area Under Queue1 Length Curve'] += \
            state['priority 1 Queue Length'] * (clock - data['Last Time Queue1 Length Changed'])
        state['priority 1 Queue Length'] += 1
        data['Last Time Queue1 Length Changed'] = clock
        data["Queue1"][patient] = clock
        if data['max1']<=state['priority 1 Queue Length']:
            data['max1']=state['priority 1 Queue Length']
    elif state['server Status'] > 0: #free
        state['server Status']-=1
        data['patients'][patient]['Part1 Service Begins'] = clock
        fel_maker(future_event_list, 'Arrival_2', clock, patient)
    n = random.random()
    if n >= 0.6:
        n = 3
    else:
        n = 1
    next_patient = 'C' + str(n) + str(int((patient[2:])) + 1)
    if next_patient[1]=='1':
        fel_maker(future_event_list, 'Arrival_1', clock, next_patient)
    if next_patient[1]=='3':
        fel_maker(future_event_list, 'Arrival_3', clock, next_patient)

def arrival_3(future_event_list, state, clock,patient,data):
    data['Cumulative Stats']['priority3'] += 1
    if clock >= 7200:
        data['Cumulative Stats']['number of patients'] += 1
    data['patients'][patient] = dict()
    data['patients'][patient]['Arrival Time'] = clock
    if state['server Status'] == 0:  # busy
        data['Cumulative Stats']['Area Under Queue3 Length Curve'] += \
            state['priority 3 Queue Length'] * (clock - data['Last Time Queue3 Length Changed'])
        state['priority 3 Queue Length'] += 1
        data['Last Time Queue3 Length Changed'] = clock
        data["Queue3"][patient]=clock
        if data['max3'] < state['priority 3 Queue Length']:
            data['max3'] = state['priority 3 Queue Length']
    elif state['server Status'] > 0:  # free
        state['server Status'] -= 1
        data['patients'][patient]['Part1 Service Begins'] = clock
        fel_maker(future_event_list, 'Arrival_2', clock, patient)
    n = random.random()
    if n >= 0.6:
        n = 3
    else:
        n = 1
    next_patient = 'C' + str(n) + str(int((patient[2:])) + 1)
    if next_patient[1] == '1':
        fel_maker(future_event_list, 'Arrival_1', clock, next_patient)
    if next_patient[1] == '3':
        fel_maker(future_event_list, 'Arrival_3', clock, next_patient)


def arrival_2(future_event_list, state, clock, patient,data):
    data['Cumulative Stats']['priority2'] += 1
    data['patients'][patient]['End of part 1 service']=clock
    if clock >= 7200:
        data['Cumulative Stats']['Server Busy Time'] += (clock- data['patients'][patient]['Part1 Service Begins'])
    data['patients'][patient]['Arrival2 Time']=clock
    if state['priority 3 Queue Length']>0:
        data['Cumulative Stats']['Area Under Queue2 Length Curve'] += \
            state['priority 2 Queue Length'] * (clock - data['Last Time Queue2 Length Changed'])
        state['priority 2 Queue Length'] += 1
        data['Cumulative Stats']['Area Under Queue3 Length Curve'] += \
            state['priority 3 Queue Length'] * (clock - data['Last Time Queue3 Length Changed'])
        state['priority 3 Queue Length']-=1
        data['Last Time Queue3 Length Changed'] = clock
        data['Last Time Queue2 Length Changed'] = clock
        data['Queue2'][patient]=clock
        if data['max2'] <= state['priority 2 Queue Length']:
            data['max2'] = state['priority 2 Queue Length']
        if data['max3'] < state['priority 3 Queue Length']:
            data['max3'] = state['priority 3 Queue Length']
        first_customer_in_queue = min(data['Queue3'], key=data['Queue3'].get)
        data['patients'][first_customer_in_queue]['Part1 Service Begins']=clock
        fel_maker(future_event_list, 'Arrival_2', clock,first_customer_in_queue )
        data['Queue3'].pop(first_customer_in_queue, None)
    elif state['priority 3 Queue Length']==0:
        if state['priority 2 Queue Length']==0:
            data['patients'][patient]['Part2 Service Begins'] = clock
            fel_maker(future_event_list, 'End of Service', clock,patient)
        elif state['priority 2 Queue Length']>0:
            first_customer_in_queue = min(data['Queue2'], key=data['Queue2'].get)
            data['patients'][first_customer_in_queue]['Part2 Service Begins'] = clock
            fel_maker(future_event_list, 'End of Service', clock,first_customer_in_queue)
            data['Queue2'].pop(first_customer_in_queue, None)
            data['Queue2'][patient] = clock

def end_of_service(future_event_list, state, clock, patient,data):
    data['patients'][patient]['End of part 2 service'] = clock
    if clock>=7200:
        data['Cumulative Stats']['Server Busy Time'] += (clock - data['patients'][patient]['Part2 Service Begins'])
    data['Cumulative Stats']['priority2 time spent'] += (clock - data['patients'][patient]['Arrival2 Time'])
    if clock>=7200:
        data['Cumulative Stats']['time spent']+=\
        (data['patients'][patient]['End of part 2 service'] -data['patients'][patient]['Arrival Time'])
    if patient[1]=='1':
        data['Cumulative Stats']['priority1 time spent']+= (clock-data['patients'][patient]['Arrival Time'])
    if patient[1]=='3':
        if data['patients'][patient]['Part1 Service Begins']-data['patients'][patient]['Arrival Time']==0:
            if data['patients'][patient]['End of part 1 service']-data['patients'][patient]['Part2 Service Begins']==0:
                data['Cumulative Stats']['priority3 no expect']+=1

    if state['priority 3 Queue Length']>0:
        data['Cumulative Stats']['Area Under Queue3 Length Curve'] += \
            state['priority 3 Queue Length'] * (clock - data['Last Time Queue3 Length Changed'])
        state['priority 3 Queue Length'] -= 1
        data['Last Time Queue3 Length Changed'] = clock
        if data['max3'] < state['priority 3 Queue Length']:
            data['max3'] = state['priority 3 Queue Length']
        first_customer_in_queue = min(data['Queue3'], key=data['Queue3'].get)
        data['patients'][first_customer_in_queue]['Part1 Service Begins'] = clock
        fel_maker(future_event_list, 'Arrival_2', clock, first_customer_in_queue)
        data['Queue3'].pop(first_customer_in_queue, None)
    if state['priority 3 Queue Length'] == 0:
        if state['priority 2 Queue Length']>0:
            data['Cumulative Stats']['Area Under Queue2 Length Curve'] += \
                state['priority 2 Queue Length'] * (clock - data['Last Time Queue2 Length Changed'])
            state['priority 2 Queue Length']-=1
            data['Last Time Queue2 Length Changed'] = clock
            if data['max2'] < state['priority 2 Queue Length']:
                data['max2'] = state['priority 2 Queue Length']
            first_customer_in_queue = min(data['Queue2'], key=data['Queue2'].get)
            data['patients'][first_customer_in_queue]['Part2 Service Begins'] = clock
            fel_maker(future_event_list, 'End of Service', clock,first_customer_in_queue)
            data['Queue2'].pop(first_customer_in_queue, None)
        else:
            if state['priority 1 Queue Length']>0:
                data['Cumulative Stats']['Area Under Queue1 Length Curve'] += \
                     state['priority 1 Queue Length'] * (clock - data['Last Time Queue1 Length Changed'])
                data['Last Time Queue1 Length Changed'] = clock
                state['priority 1 Queue Length']-=1
                if data['max1'] <= state['priority 1 Queue Length']:
                    data['max1'] = state['priority 1 Queue Length']
                first_customer_in_queue = min(data['Queue1'], key=data['Queue1'].get)
                data['patients'][first_customer_in_queue]['Part1 Service Begins'] = clock
                fel_maker(future_event_list, 'Arrival_2', clock,first_customer_in_queue)
                data['Queue1'].pop(first_customer_in_queue, None)
            else:
                state['server Status'] += 1
def create_row(step, current_event, state, data, future_event_list):
    # Sort future events by their scheduled time for easier readability
    sorted_fel = sorted(future_event_list, key=lambda x: x['Event Time'])
    # Start building a row with step number, event time, type, and the involved patient
    row = [step, current_event['Event Time'], current_event['Event Type'], current_event['Patient']]
    # Append current state values (e.g., queue lengths, server status) to the row
    row.extend(list(state.values()))
    # Append cumulative statistics (e.g., average wait times, service times) to the row
    row.extend(list(data['Cumulative Stats'].values()))
    # Append future event details for transparency on upcoming simulation events
    for event in sorted_fel:
        row.extend([event['Event Time'], event['Event Type'], event.get('Patient', '')])
    return row
def justify(table):
    # Find the maximum row length to standardize the length of all rows
    row_max_len = max(len(row) for row in table)
    # Extend each row with empty strings to ensure uniform length
    for row in table:
        row.extend([""] * (row_max_len - len(row)))
def create_main_header(state, data):
    # Create the main header row with fixed and dynamic components
    header = ['Step', 'Clock', 'Event Type', 'Patient']
    # Include headers for current state information (queue lengths, server status)
    header.extend(list(state.keys()))
    # Include headers for cumulative statistics
    header.extend(list(data['Cumulative Stats'].keys()))
    return header
def create_excel(table, header):
    # Prepare additional headers for future events based on the longest row
    prepare_future_event_headers(table, header)
    # Convert the table into a pandas DataFrame for easier Excel writing
    df = pd.DataFrame(table, columns=header)
    # Use XlsxWriter engine for enhanced Excel formatting capabilities
    writer = pd.ExcelWriter('output.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Output', index=False)
    # Apply formatting to the Excel sheet for readability
    apply_excel_formatting(writer, df)
    writer.close()

def prepare_future_event_headers(table, header):
    # Calculate the number of future event columns to add based on the longest row
    row_len = len(table[0])
    header_len = len(header)
    num_future_events = (row_len - header_len) // 3
    for i in range(1, num_future_events + 1):
        header.extend([f'Future Event Time {i}', f'Future Event Type {i}', f'Future Event Patient {i}'])

def apply_excel_formatting(writer, df):
    workbook = writer.book
    worksheet = writer.sheets['Output']
    # Define formatting options
    header_formatter = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'bold': True, 'font_name': 'Times New Roman'})
    cell_formatter = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'font_name': 'Times New Roman'})
    # Apply column widths based on content
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, header_formatter)
        column_len = max(df[value].astype(str).map(len).max(), len(value))
        worksheet.set_column(col_num, col_num, column_len, cell_formatter)


def simulation(simulation_time):
    table = []  # Table to accumulate data for Excel output
    step = 1
    state, data, future_event_list = starting_state()  # Initialize the simulation
    clock = 0
    future_event_list.append({'Event Type': 'End of Simulation', 'Event Time': simulation_time, 'Patient': None})

    while clock < simulation_time:
        sorted_fel = sorted(future_event_list, key=lambda x: x['Event Time'])
        current_event = sorted_fel[0]
        clock = current_event['Event Time']
        patient = current_event.get('Patient', None)  # Safely get patient ID or None
        
        # Process the current event based on its type
        if current_event['Event Type'] == 'Arrival_1':
            arrival_1(future_event_list, state, clock, patient, data)
        elif current_event['Event Type'] == 'Arrival_3':
            arrival_3(future_event_list, state, clock, patient, data)
        elif current_event['Event Type'] == 'Arrival_2':
            arrival_2(future_event_list, state, clock, patient, data)
        elif current_event['Event Type'] == 'End of Service':
            end_of_service(future_event_list, state, clock, patient, data)
        
        future_event_list.remove(current_event)  # Remove processed event
        table.append(create_row(step, current_event, state, data, future_event_list))  # Add event data to table
        step += 1
    
    # After simulation loop: Generate Excel report
    excel_main_header = create_main_header(state, data)
    justify(table)  # Ensure all rows in table are of equal length
    create_excel(table, excel_main_header)  # Create the Excel file

    return data  # Optional: return simulation data for further analysis 
simulation(28800)  # Run simulation for 20 Days >> minutes

workbook_path='/Users/Serendipity/output.xlsx'
def write_simulation_results_to_excel(workbook_path, simulation_results, num_of_replications, simulation_time):
    workbook = xlsxwriter.Workbook(workbook_path)
    worksheet = workbook.add_worksheet()

    metrics = list(simulation_results.keys())  # Assuming this contains the main metric names

    # Assuming metrics might contain nested dictionaries
    for row, metric in enumerate(metrics):
        worksheet.write(row, 0, metric)  # Write metric names as header in the first column

    for rep in range(1, num_of_replications + 1):
        simulation_result = simulation(simulation_time)  # Run simulation
        for row, metric in enumerate(metrics):
            value = simulation_result[metric]
            # Check if the value is a simple data type or a nested dictionary
            if isinstance(value, (int, float, str)):
                value_to_write = value
            elif isinstance(value, dict):
                # For simplicity, let's just write the length of the dictionary if it's a nested structure
                # You may want to extract specific values instead
                value_to_write = len(value)
            else:
                value_to_write = "Unsupported Type"
            worksheet.write(row, rep, value_to_write)

    workbook.close()

# Assuming 'simulation' is your simulation function
simulation_results = simulation(240000)
write_simulation_results_to_excel('simulation_results.xlsx', simulation_results, 10, 240000)



# 'Warm-up analysis - Time-Frame Approach
num_of_replications = 100  # Total number of simulation replications
num_of_days = 20  # Total number of days the simulation runs
frame_length = 480  # Length of each time frame in minutes (e.g., an 8-hour shift)
window_size = 9  # Window size for moving average calculation
tick_spacing = 5  # Spacing between ticks on the plots

# Setting matplotlib parameters for font
mpl.rc('font', family='Times New Roman')
mpl.rc('font', size=12)

# Creating a figure and axes for plotting with 2 rows and 1 column
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

# Dictionaries to hold finishing patient counts and waiting time averages across simulations
finishing_patients_frame_count = dict()
waiting_time_frame_average = dict()

# Function to calculate moving average of a list with a specified window size
def moving_average(input_list, m):
    output_list = []
    n = len(input_list)
    # Iterates through the list to calculate moving average for each point
    for i in range(n):
        output_list.append(sum(input_list[max(i - m // 2, 2 * i - n + 1, 0):min(i + m // 2 + 1, 2 * i + 1, n)]) / (
                min(i + m // 2, 2 * i, n - 1) - max(i - m // 2, 2 * i - n + 1, 0) + 1))
    return output_list

# Function to calculate the number of patients finishing within a time frame
def calculate_number_patients(start_time, end_time, patients_data):
    number_of_patients = 0
    # Loops through each patient to check if they finished within the time frame
    for patient in patients_data:
        if start_time < patients_data[patient]['End of part 2 service'] <= end_time:
            number_of_patients += 1
        elif patients_data[patient]['End of part 2 service'] > end_time:
            break
    return number_of_patients

# Function to calculate total waiting time of patients within a time frame
def calculate_aggregate_waiting_time(start_time, end_time, patients_data):
    cumulative_waiting_time = 0
    # Loops through each patient to aggregate their waiting time within the time frame
    for patient in patients_data:
        if start_time <= patients_data[patient]['Arrival Time'] < end_time:
            if start_time <= patients_data[patient]['End of part 2 service'] < end_time:
                cumulative_waiting_time += patients_data[patient]['End of part 2 service'] - \
                                           patients_data[patient]['Arrival Time']
            elif patients_data[patient]['End of part 2 service'] >= end_time:
                cumulative_waiting_time += end_time - \
                                           patients_data[patient]['Arrival Time']
        elif start_time > patients_data[patient]['Arrival Time']:
           if start_time <= patients_data[patient]['End of part 2 service'] < end_time:
                 cumulative_waiting_time += patients_data[patient]['End of part 2 service'] - \
                                   start_time
           elif patients_data[patient]['End of part 2 service'] >= end_time:
                cumulative_waiting_time += end_time-start_time
        elif patients_data[patient]['Arrival Time'] > end_time:
              break
    return cumulative_waiting_time

# Total simulation time calculated from the number of days
simulation_time = num_of_days * 1440  # Convert days to minutes
# Calculating the number of frames based on simulation time and frame length
num_of_frames = simulation_time // frame_length - 2
# x-axis values for the plot
x = [i for i in range(1, num_of_frames + 1)]

# Main loop to run simulations and collect data
for replication in range(1, num_of_replications + 1):
    # Run the simulation and collect data
    simulation_data = simulation(num_of_days * 1440)
    patients_data = simulation_data['patients']
    # Initialize lists to hold data for this replication
    finishing_patients_frame_count[replication] = []
    waiting_time_frame_average[replication] = []
    # Loop through each frame to calculate data
    for time in range(0, num_of_frames * frame_length, frame_length):
        finishing_patients_frame_count[replication].append(
            calculate_number_patients(time, time + frame_length, patients_data))
        waiting_time_frame_average[replication].append(
            calculate_aggregate_waiting_time(time, time + frame_length, patients_data))

# Averaging results across replications and applying moving average
finishing_patients_replication_average = []
waiting_time_replication_average = []
for i in range(num_of_frames):
    average_finishing_patients = 0
    average_stay_time = 0
    for replication in range(1, num_of_replications + 1):
        average_finishing_patients += finishing_patients_frame_count[replication][i] * (1 / num_of_replications)
        average_stay_time += waiting_time_frame_average[replication][i] * (1 / num_of_replications)
    finishing_patients_replication_average.append(average_finishing_patients)
    waiting_time_replication_average.append(average_stay_time)

# Applying moving average to the averaged results
moving_finishing_patients_replication_average = moving_average(finishing_patients_replication_average, window_size)
moving_waiting_time_replication_average = moving_average(waiting_time_replication_average, window_size)

# Setting up plot titles, labels, legends, and saving the figure
fig.suptitle(f'Warm-up analysis over {num_of_replications} replications')
# Plot for the number of finishing customers
ax[0].plot(x, finishing_patients_replication_average, 'r', linewidth=5, label="Average across replications")
ax[0].plot(x, moving_finishing_patients_replication_average, 'k', label=f'Moving average (m = {window_size})')
ax[0].set_title('Number of Finishing Customers')
ax[0].set_xlabel('Frame No.')
ax[0].set_ylabel('Number of Finishing Customers')
ax[0].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax[0].legend()
# Plot for the aggregate waiting time
ax[1].plot(x, waiting_time_replication_average, 'r', linewidth=5, label="Average across replications")
ax[1].plot(x, moving_waiting_time_replication_average, 'k', label=f'Moving average (m = {window_size})')
ax[1].set_title('Aggregate Waiting Time')
ax[1].set_xlabel('Frame No.')
ax[1].set_ylabel('Aggregate Waiting Time')
ax[1].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax[1].legend()

# Display the figure and save it to a file
fig.show()
fig.savefig('Warm-up analysis - Time-Frame Approach')
