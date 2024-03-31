import random
import math
import pandas as pd
import numpy as np
import xlsxwriter


# Simulation of hospital
#
# Input Distributions:
#     1- Inter-arrival time: Exponential with mean = 21 minutes
#     2- Service time for patient with Priority 3: triangular(22,40,62)
#     3- Service time for patient with Priority 1:40*B(1,3)+3
#     4- Service time for patient with Priority 2 was with priority 3:triangular(10,12,14)
#     5- Service time for patient with Priority 2 was with priority 1:uniform(8,12)
#     statistics:
#     Average Time Spent in the System by Priority 1 and 2 Patients
#     Percentage of Patients with Priority 3 Who Start Service immediately
#     Maximum and Average Queue Length for Type 1, 2, and 3 Patients
#     Mean of Doctor's Productivity
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
        'r': 0,  # A flag to indicate if the server needs rest
        'k': 0,  # Counter for rest periods
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
            event_time = clock + (40*(random.betavariate(1,3)))+3
        elif patient[1]=='3':
            event_time = clock + triangular(22,40,62)

    elif event_type == 'End of Service':
        if patient[1] == '1':
            event_time = clock + uniform(8,12)
        elif patient[1] == '3':
            event_time = clock + triangular(10,12,14)

    elif event_type == 'rest_time':
        event_time = clock + 60


    elif event_type== 'end_of_rest':
        event_time = clock + 10


    new_event = {'Event Type': event_type, 'Event Time': event_time,'Patient': patient}
    future_event_list.append(new_event)
def arrival_1(future_event_list, state, clock, patient, data):
    # Increment the count of Priority 1 patients
    data['Cumulative Stats']['priority1'] += 1

    # Record the arrival time of the patient
    data['patients'][patient] = {'Arrival Time': clock}

    if state['server Status'] == 0:  # If the server is busy
        # Update the area under the queue length curve for Priority 1 patients
        # This represents the total waiting time of patients in the queue until this moment
        data['Cumulative Stats']['Area Under Queue1 Length Curve'] += \
            state['priority 1 Queue Length'] * (clock - data['Last Time Queue1 Length Changed'])
        # Increment the queue length for Priority 1 patients
        state['priority 1 Queue Length'] += 1
        # Update the time when the queue length last changed
        data['Last Time Queue1 Length Changed'] = clock
        # Add the patient to the queue
        data["Queue1"][patient] = clock
        # Update the maximum queue length for Priority 1, if necessary
        if data['max1'] <= state['priority 1 Queue Length']:
            data['max1'] = state['priority 1 Queue Length']
    elif state['server Status'] > 0:  # If the server is free
        # Decrement the server status to indicate it's now busy with this patient
        state['server Status'] -= 1
        # Record the time when service begins for this patient
        data['patients'][patient]['Part1 Service Begins'] = clock
        # Schedule the next part of service for this patient
        fel_maker(future_event_list, 'Arrival_2', clock, patient)
    # Determine the next patient's priority randomly
    n = random.random()
    next_priority = 3 if n >= 0.6 else 1
    # Create the next patient ID
    next_patient = 'C' + str(next_priority) + str(int(patient[2:]) + 1)
    # Schedule the next patient's arrival
    if next_priority == 1:
        fel_maker(future_event_list, 'Arrival_1', clock, next_patient)
    else:
        fel_maker(future_event_list, 'Arrival_3', clock, next_patient)

def arrival_2(future_event_list, state, clock, patient, data):
    # Increment the counter for patients with priority 2.
    data['Cumulative Stats']['priority2'] += 1
    # Record the clock time as the end of part 1 service for the patient.
    data['patients'][patient]['End of part 1 service'] = clock
    # Update the total server busy time based on the duration of part 1 service for the patient.
    data['Cumulative Stats']['Server Busy Time'] += (clock - data['patients'][patient]['Part1 Service Begins'])
    # Mark the arrival time for part 2 of the service for the patient.
    data['patients'][patient]['Arrival2 Time'] = clock
    
    # Check if a rest period was initiated but not completed due to the immediate need for service.
    if data['r'] == 1:
        # Increment the counter for rest periods.
        data['k'] += 1
        # Reset the rest flag.
        data['r'] = 0
        # Calculate and update the area under the queue 2 length curve.
        data['Cumulative Stats']['Area Under Queue2 Length Curve'] += \
            state['priority 2 Queue Length'] * (clock - data['Last Time Queue2 Length Changed'])
        # Increment the queue length for priority 2.
        state['priority 2 Queue Length'] += 1
        # Update the last time the queue 2 length changed.
        data['Last Time Queue2 Length Changed'] = clock
        # Add the patient to queue 2.
        data["Queue2"][patient] = clock
        # Update the maximum queue length for priority 2 if necessary.
        if data['max2'] <= state['priority 2 Queue Length']:
            data['max2'] = state['priority 2 Queue Length']
        # Schedule the end of the rest period.
        fel_maker(future_event_list, 'end_of_rest', clock, None)
    else:
        # If there are patients with priority 3 waiting.
        if state['priority 3 Queue Length'] > 0:
            # Update the area under the curve for queue lengths of priority 2 and 3.
            data['Cumulative Stats']['Area Under Queue2 Length Curve'] += \
                state['priority 2 Queue Length'] * (clock - data['Last Time Queue2 Length Changed'])
            state['priority 2 Queue Length'] += 1
            data['Cumulative Stats']['Area Under Queue3 Length Curve'] += \
                state['priority 3 Queue Length'] * (clock - data['Last Time Queue3 Length Changed'])
            state['priority 3 Queue Length'] -= 1
            # Update the last time the queue lengths changed.
            data['Last Time Queue3 Length Changed'] = clock
            data['Last Time Queue2 Length Changed'] = clock
            # Add the current patient to queue 2.
            data['Queue2'][patient] = clock
            # Check and update the maximum queue lengths if necessary.
            if data['max2'] <= state['priority 2 Queue Length']:
                data['max2'] = state['priority 2 Queue Length']
            if data['max3'] < state['priority 3 Queue Length']:
                data['max3'] = state['priority 3 Queue Length']
            # Find the first patient in queue 3, begin their service, and remove them from the queue.
            first_customer_in_queue = min(data['Queue3'], key=data['Queue3'].get)
            data['patients'][first_customer_in_queue]['Part1 Service Begins'] = clock
            fel_maker(future_event_list, 'Arrival_2', clock, first_customer_in_queue)
            data['Queue3'].pop(first_customer_in_queue, None)
        # If there are no priority 3 patients waiting.
        elif state['priority 3 Queue Length'] == 0:
            # If no patients are waiting in priority 2 queue, begin part 2 service for the current patient.
            if state['priority 2 Queue Length'] == 0:
                data['patients'][patient]['Part2 Service Begins'] = clock
                fel_maker(future_event_list, 'End of Service', clock, patient)
            # If there are patients waiting in the priority 2 queue.
            elif state['priority 2 Queue Length'] > 0:
                # Find the first patient in queue 2, begin their part 2 service, and remove them from the queue.
                first_customer_in_queue = min(data['Queue2'], key=data['Queue2'].get)
                data['patients'][first_customer_in_queue]['Part2 Service Begins'] = clock
                fel_maker(future_event_list, 'End of Service', clock, first_customer_in_queue)
                data['Queue2'].pop(first_customer_in_queue, None)
                # Add the current patient to queue 2.
                data['Queue2'][patient] = clock

def arrival_3(future_event_list, state, clock, patient, data):
    # Increment the count of priority 3 patients.
    data['Cumulative Stats']['priority3'] += 1
    # Initialize patient data.
    data['patients'][patient] = dict()
    # Record the arrival time of the patient.
    data['patients'][patient]['Arrival Time'] = clock
    
    # Check if the server is busy.
    if state['server Status'] == 0:
        # Update the area under queue 3 length curve.
        data['Cumulative Stats']['Area Under Queue3 Length Curve'] += \
            state['priority 3 Queue Length'] * (clock - data['Last Time Queue3 Length Changed'])
        # Increment the priority 3 queue length.
        state['priority 3 Queue Length'] += 1
        # Update the last time queue 3 length changed.
        data['Last Time Queue3 Length Changed'] = clock
        # Record the arrival time of the patient in the queue.
        data["Queue3"][patient] = clock
        # Update the maximum priority 3 queue length.
        if data['max3'] < state['priority 3 Queue Length']:
            data['max3'] = state['priority 3 Queue Length']
    # If the server is free, start serving the patient immediately.
    elif state['server Status'] > 0:
        # Decrement the server status as it becomes busy.
        state['server Status'] -= 1
        # Record the start time of part 1 service for the patient.
        data['patients'][patient]['Part1 Service Begins'] = clock
        # Schedule the next event for the patient.
        fel_maker(future_event_list, 'Arrival_2', clock, patient)
    
    n = random.random()
    next_priority = 3 if n >= 0.6 else 1
    # Create the next patient ID
    next_patient = 'C' + str(next_priority) + str(int(patient[2:]) + 1)
    # Schedule the next patient's arrival
    if next_priority == 1:
        fel_maker(future_event_list, 'Arrival_1', clock, next_patient)
    else:
        fel_maker(future_event_list, 'Arrival_3', clock, next_patient)

def end_of_service(future_event_list, state, clock, patient, data):
    # Record the end time of part 2 service for the patient.
    data['patients'][patient]['End of part 2 service'] = clock
    # Update the cumulative server busy time.
    data['Cumulative Stats']['Server Busy Time'] += (clock - data['patients'][patient]['Part2 Service Begins'])
    # Update the time spent by the patient in priority 2.
    data['Cumulative Stats']['priority2 time spent'] += (clock - data['patients'][patient]['Arrival2 Time'])
    
    # Check the priority of the patient and update the time spent accordingly.
    if patient[1] == '1':
        data['Cumulative Stats']['priority1 time spent'] += (clock - data['patients'][patient]['Arrival Time'])
    if patient[1] == '3':
        # Check if the patient experienced unexpected delay in priority 3.
        if (data['patients'][patient]['Part1 Service Begins'] - data['patients'][patient]['Arrival Time']) == 0:
            if (data['patients'][patient]['End of part 1 service'] - data['patients'][patient]['Part2 Service Begins']) == 0:
                data['Cumulative Stats']['priority3 no expect'] += 1
    
    # Check if the server needs to rest.
    if data['r'] == 1:
        data['r'] = 0
        data['k'] += 1
        fel_maker(future_event_list, 'end_of_rest', clock, None)
    else:
        # If there are patients waiting in priority 3 queue, schedule their service.
        if state['priority 3 Queue Length'] > 0:
            # Update the area under queue 3 length curve.
            data['Cumulative Stats']['Area Under Queue3 Length Curve'] += \
                state['priority 3 Queue Length'] * (clock - data['Last Time Queue3 Length Changed'])
            # Decrement the priority 3 queue length.
            state['priority 3 Queue Length'] -= 1
            # Update the last time queue 3 length changed.
            data['Last Time Queue3 Length Changed'] = clock
            # Find the first customer in the priority 3 queue.
            first_customer_in_queue = min(data['Queue3'], key=data['Queue3'].get)
            # Mark the beginning of part 1 service for the first customer.
            data['patients'][first_customer_in_queue]['Part1 Service Begins'] = clock
            # Schedule the next event for part 2 service for the first customer.
            fel_maker(future_event_list, 'Arrival_2', clock, first_customer_in_queue)
            # Remove the first customer from the queue.
            data['Queue3'].pop(first_customer_in_queue, None)
        else:
            # If there are no patients in priority 3 queue, check priority 2 queue.
            if state['priority 2 Queue Length'] > 0:
                # Update the area under queue 2 length curve.
                data['Cumulative Stats']['Area Under Queue2 Length Curve'] += \
                    state['priority 2 Queue Length'] * (clock - data['Last Time Queue2 Length Changed'])
                # Decrement the priority 2 queue length.
                state['priority 2 Queue Length'] -= 1
                # Update the last time queue 2 length changed.
                data['Last Time Queue2 Length Changed'] = clock
                # Find the first customer in the priority 2 queue.
                first_customer_in_queue = min(data['Queue2'], key=data['Queue2'].get)
                # Mark the beginning of part 2 service for the first customer.
                data['patients'][first_customer_in_queue]['Part2 Service Begins'] = clock
                # Schedule the end of service event for the first customer.
                fel_maker(future_event_list, 'End of Service', clock, first_customer_in_queue)
                # Remove the first customer from the queue.
                data['Queue2'].pop(first_customer_in_queue, None)
            else:
                # If there are no patients in priority 2 queue, check priority 1 queue.
                if state['priority 1 Queue Length'] > 0:
                    # Update the area under queue 1 length curve.
                    data['Cumulative Stats']['Area Under Queue1 Length Curve'] += \
                         state['priority 1 Queue Length'] * (clock - data['Last Time Queue1 Length Changed'])
                    # Update the last time queue 1 length changed.
                    data['Last Time Queue1 Length Changed'] = clock
                    # Decrement the priority 1 queue length.
                    state['priority 1 Queue Length'] -= 1
                    # Update the maximum priority 1 queue length.
                    if data['max1'] <= state['priority 1 Queue Length']:
                        data['max1'] = state['priority 1 Queue Length']
                    # Find the first customer in the priority 1 queue.
                    first_customer_in_queue = min(data['Queue1'], key=data['Queue1'].get)
                    # Mark the beginning of part 1 service for the first customer.
                    data['patients'][first_customer_in_queue]['Part1 Service Begins'] = clock
                    # Schedule the next event for part 2 service for the first customer.
                    fel_maker(future_event_list, 'Arrival_2', clock, first_customer_in_queue)
                    # Remove the first customer from the queue.
                    data['Queue1'].pop(first_customer_in_queue, None)
                else:
                    # If there are no patients in any queue, increment the server status.
                    state['server Status'] += 1

def rest(future_event_list, state, clock, data):
    # Check if the server is free.
    if state['server Status'] > 0:
        # Increment the count of rest times.
        data['k'] += 1
        # Decrement the server status as it becomes busy during rest.
        state['server Status'] -= 1
        # Schedule the end of rest event.
        fel_maker(future_event_list, 'end_of_rest', clock, None)
    # If the server is busy, set the rest flag.
    elif state['server Status'] == 0:
        data['r'] = 1

def end_of_rest(future_event_list, state, clock, data):
    # If there are patients waiting in priority 3 queue, schedule their service.
    if state['priority 3 Queue Length'] > 0:
        # Update the area under queue 3 length curve.
        data['Cumulative Stats']['Area Under Queue3 Length Curve'] += \
            state['priority 3 Queue Length'] * (clock - data['Last Time Queue3 Length Changed'])
        # Decrement the priority 3 queue length.
        state['priority 3 Queue Length'] -= 1
        # Update the last time queue 3 length changed.
        data['Last Time Queue3 Length Changed'] = clock
        # Find the first customer in the priority 3 queue.
        first_customer_in_queue = min(data['Queue3'], key=data['Queue3'].get)
        # Mark the beginning of part 1 service for the first customer.
        data['patients'][first_customer_in_queue]['Part1 Service Begins'] = clock
        # Schedule the next event for part 2 service for the first customer.
        fel_maker(future_event_list, 'Arrival_2', clock, first_customer_in_queue)
        # Remove the first customer from the queue.
        data['Queue3'].pop(first_customer_in_queue, None)
    else:
        # If there are no patients in priority 3 queue, check priority 2 queue.
        if state['priority 2 Queue Length'] > 0:
            # Update the area under queue 2 length curve.
            data['Cumulative Stats']['Area Under Queue2 Length Curve'] += \
                state['priority 2 Queue Length'] * (clock - data['Last Time Queue2 Length Changed'])
            # Decrement the priority 2 queue length.
            state['priority 2 Queue Length'] -= 1
            # Update the last time queue 2 length changed.
            data['Last Time Queue2 Length Changed'] = clock
            # Find the first customer in the priority 2 queue.
            first_customer_in_queue = min(data['Queue2'], key=data['Queue2'].get)
            # Mark the beginning of part 2 service for the first customer.
            data['patients'][first_customer_in_queue]['Part2 Service Begins'] = clock
            # Schedule the end of service event for the first customer.
            fel_maker(future_event_list, 'End of Service', clock, first_customer_in_queue )
            # Remove the first customer from the queue.
            data['Queue2'].pop(first_customer_in_queue, None)
        else:
            # If there are no patients in priority 2 queue, check priority 1 queue.
            if state['priority 1 Queue Length'] > 0:
                # Update the area under queue 1 length curve.
                data['Cumulative Stats']['Area Under Queue1 Length Curve'] += \
                    state['priority 1 Queue Length'] * (clock - data['Last Time Queue1 Length Changed'])
                # Update the last time queue 1 length changed.
                data['Last Time Queue1 Length Changed'] = clock
                # Decrement the priority 1 queue length.
                state['priority 1 Queue Length'] -= 1
                # Update the maximum priority 1 queue length.
                if data['max1'] < state['priority 1 Queue Length']:
                    data['max1'] = state['priority 1 Queue Length']
                # Find the first customer in the priority 1 queue.
                first_customer_in_queue = min(data['Queue1'], key=data['Queue1'].get)
                # Mark the beginning of part 1 service for the first customer.
                data['patients'][first_customer_in_queue]['Part1 Service Begins'] = clock
                # Schedule the next event for part 2 service for the first customer.
                fel_maker(future_event_list, 'Arrival_2', clock, first_customer_in_queue)
                # Remove the first customer from the queue.
                data['Queue1'].pop(first_customer_in_queue, None)
            else:
                # If there are no patients in any queue, increment the server status.
                state['server Status'] += 1

    # Schedule the next rest time event based on the count of rest times.
    if data['k'] % 2 != 0:
        fel_maker(future_event_list, 'rest_time', clock, None)
    else:
        future_event_list.append({'Event Type': 'rest_time', 'Event Time': (((data['k'] // 2) * 8) + 5) * 60,
                                  'Patient': None, 'restflag': None, 'k': None})

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
        elif current_event['Event Type'] == 'rest_time':
            rest(future_event_list, state, clock, data)
        elif current_event['Event Type'] == 'end_of_rest':
            end_of_rest(future_event_list, state, clock, data)
        
        future_event_list.remove(current_event)  # Remove processed event
        table.append(create_row(step, current_event, state, data, future_event_list))  # Add event data to table
        step += 1
    
    # After simulation loop: Generate Excel report
    excel_main_header = create_main_header(state, data)
    justify(table)  # Ensure all rows in table are of equal length
    create_excel(table, excel_main_header)  # Create the Excel file

    return data  # Optional: return simulation data for further analysis 
simulation(28800)  # Run simulation for 20 Days >> minutes



#Calculate metrics
def calculate_metrics(data):
    # Calculate average time spent in the system by priority 1 patients
    total_time_p1 = sum(data['patients'][p]['End of part 2 service'] - data['patients'][p]['Arrival Time']
                        for p in data['patients'] if p[1] == '1')
    total_p1_patients = sum(1 for p in data['patients'] if p[1] == '1')
    average_time_p1 = total_time_p1 / total_p1_patients if total_p1_patients > 0 else 0

    # Calculate average time spent in the system by priority 2 patients
    total_time_p2 = sum(data['patients'][p]['End of part 2 service'] - data['patients'][p]['Arrival Time']
                        for p in data['patients'] if p[1] == '2')
    total_p2_patients = sum(1 for p in data['patients'] if p[1] == '2')
    average_time_p2 = total_time_p2 / total_p2_patients if total_p2_patients > 0 else 0

    # Calculate percentage of patients with priority 3 who start service immediately
    immediate_service_count = sum(1 for p in data['patients'] if p[1] == '3' and
                                  data['patients'][p]['Part1 Service Begins'] ==
                                  data['patients'][p]['Arrival Time'])
    total_p3_patients = sum(1 for p in data['patients'] if p[1] == '3')
    immediate_service_percentage = (immediate_service_count / total_p3_patients) * 100 if total_p3_patients > 0 else 0

    # Maximum queue length for each priority type
    max_queue_length_p1 = data['max1']
    max_queue_length_p2 = data['max2']
    max_queue_length_p3 = data['max3']

    # Average queue length for each priority type
    average_queue_length_p1 = data['Cumulative Stats']['Area Under Queue1 Length Curve'] / data['Total Time']
    average_queue_length_p2 = data['Cumulative Stats']['Area Under Queue2 Length Curve'] / data['Total Time']
    average_queue_length_p3 = data['Cumulative Stats']['Area Under Queue3 Length Curve'] / data['Total Time']

    # Mean of doctor's productivity
    mean_productivity = data['Cumulative Stats']['Server Busy Time'] / data['Total Time']

    return {
        'Average Time Spent Priority 1': average_time_p1,
        'Average Time Spent Priority 2': average_time_p2,
        'Percentage Priority 3 Immediate Service': immediate_service_percentage,
        'Max Queue Length Priority 1': max_queue_length_p1,
        'Max Queue Length Priority 2': max_queue_length_p2,
        'Max Queue Length Priority 3': max_queue_length_p3,
        'Average Queue Length Priority 1': average_queue_length_p1,
        'Average Queue Length Priority 2': average_queue_length_p2,
        'Average Queue Length Priority 3': average_queue_length_p3,
        'Mean Doctor Productivity': mean_productivity
    }


#Sensetivity_analysis
simulation_runs = 100
simulation_time = 28800  # 8 hours
def perform_sensitivity_analysis(simulation_runs, simulation_time):
    # Placeholder for storing the metric of interest from each simulation run
    server_busy_times = []

    for _ in range(simulation_runs):
        simulation_result = simulation(simulation_time)  # Run the simulation
        # Extract a specific metric - for example, 'Server Busy Time'
        busy_time = simulation_result['Cumulative Stats']['Server Busy Time']
        server_busy_times.append(busy_time / simulation_time)  # Normalize or process as needed

    # Calculate average and standard deviation
    average_busy_time = np.mean(server_busy_times)
    std_dev_busy_time = np.std(server_busy_times)
    return server_busy_times, average_busy_time, std_dev_busy_time

server_busy_times, avg_busy_time, std_busy_time = perform_sensitivity_analysis(simulation_runs, simulation_time)
print(f"Average Server Busy Time: {avg_busy_time}, Std Dev: {std_busy_time}")

def write_confidence_interval_to_excel(server_busy_times, file_path):
    workbook = xlsxwriter.Workbook(file_path)
    worksheet = workbook.add_worksheet("Confidence Interval")
    # Calculate mean, standard deviation, and standard error
    mean_busy_time = np.mean(server_busy_times)
    std_error = np.std(server_busy_times) / np.sqrt(len(server_busy_times))
    # Assuming a 95% confidence interval, Z-score ~ 1.96 for 95% confidence
    z_score = 1.96
    margin_of_error = z_score * std_error
    lower_bound = mean_busy_time - margin_of_error
    upper_bound = mean_busy_time + margin_of_error
    # Write the confidence interval to Excel
    worksheet.write('A1', 'Mean Server Busy Time')
    worksheet.write('A2', mean_busy_time)
    worksheet.write('B1', '95% Confidence Interval Lower Bound')
    worksheet.write('B2', lower_bound)
    worksheet.write('C1', '95% Confidence Interval Upper Bound')
    worksheet.write('C2', upper_bound)
    workbook.close()
    print(f"Confidence Interval written to {file_path}")
# Example of writing the confidence interval to Excel
file_path = 'ci.xlsx'
write_confidence_interval_to_excel(server_busy_times, file_path)


