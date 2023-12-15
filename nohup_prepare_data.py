import os
import sys
import datetime

# os.system("nohup sh -c '" +
#          sys.executable + " ../asr_vibravox.py > nohup_out.txt" +
#          "' &")

print("Defining logfile")
logfile = "./outputs_nohup/nohup_out_prepare_data_" + datetime.datetime.now().strftime(
    "%Y-%m-%d-%H-%M-%S"
)

# print("Function to display output in terminal in real-time")
# os.system("show_output()  { \n       tail -f " + logfile + ".txt \n}")

print("Launch prepare_data.py with nohup, redirecting output to log file")
os.system(
    "nohup " + sys.executable + " ./prepare_data.py > " + logfile + ".txt 2>&1" + " &"
)
print("Display output in terminal")
os.system("tail -f " + logfile + ".txt")
