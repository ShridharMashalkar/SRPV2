import numpy as np
import os
import time
import pandas as pd
from datetime import datetime
import subprocess
import json

start_time = time.time()
Folder_Name = "DA_Codes"
Visualization_FileName = "Visualization.py"
Comparison_FileName = "Comparison.py"
Base_Code_FileName = "base_code.py"

with open('augumentation_techniques.json', 'r') as f:
    data = json.load(f)

techniques = data['augmentation_techniques']
print(techniques)



for name in techniques:
    print('Current file is '+str(name))
    sendVal = "$".join(name)
    folderNameVisualization = "_".join(name)
    try: 
        subprocess.run(["python", Base_Code_FileName,sendVal])
        subprocess.run(["python", Visualization_FileName,folderNameVisualization])
    except Exception as e:
        print("Error:", e)
        pass
    print('******************************************************************************')
    print('\n')

end_time = time.time()
subprocess.run(["python", Comparison_FileName])

total_time = end_time  - start_time

current_datetime = time.strftime("%Y-%m-%d %H:%M:%S")
start_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
end_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))

if not os.path.exists("Computation_Time.txt"):
    with open("Computation_Time.txt", "w") as file:
        file.write("StartTime | EndTime | CurrentDate | CurrentTime | Total Time (seconds)\n")

with open("Computation_Time.txt", "a") as file:
    file.write(f"{start_datetime} | {end_datetime} | {current_datetime} | {total_time}\n")



print("Computation time logged successfully.")
