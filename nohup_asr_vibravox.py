import os
import sys
import datetime

# os.system("nohup sh -c '" +
#          sys.executable + " ../train_model.py > nohup_out.txt" +
#          "' &")

print("Defining logfile")
logfile = "./outputs_nohup/nohup_out_vibravox_asr_" + datetime.datetime.now().strftime(
    "%Y-%m-%d-%H-%M-%S"
)

# print("Function to display output in terminal in real-time")
# os.system("show_output()  { \n       tail -f " + logfile + ".txt \n}")

print("Launch train_model.py with nohup, redirecting output to log file")
os.system(
    "nohup " + sys.executable + " ./train_model.py > " + logfile + ".txt 2>&1" + " &"
)
print("Display output in terminal")
os.system("tail -f " + logfile + ".txt")
