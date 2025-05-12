from ctypes import *
import sys
import time
from dwfconstants import *
import pandas as pd
from pathlib import Path
import signal

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

import experiment.exp_LED as led
import experiment.exp_audio as audio
from experiment.experiment import Experiment
from experiment.exp_parser import parse_script
import experiment.exp_plot as exp_plot
 
#check library loading errors
szerr = create_string_buffer(512)
dwf.FDwfGetLastErrorMsg(szerr)
if szerr[0] != b'\0':
    print(str(szerr.value))

#declare ctype variables
hdwf = c_int()
cDevice = c_int()
cConfig = c_int()
cInfo = c_int()
iDevId = c_int()
iDevRev = c_int()
hzAcq = c_double(5000)
cAvailable = c_int()
cLost = c_int()
cCorrupted = c_int()
sts = c_byte()

#declare string variables
devicename = create_string_buffer(64)
serialnum = create_string_buffer(16)

red_pin = 9
yellow_pin = 8
green_pin = 10
channel = 0
fLost = 0
fCorrupted = 0

path = Path(sys.argv[1])
script = []
comment = ''
if path.is_file():
    code = path.read_text()
    (script, comment) = parse_script(code)
else:
    print("No input file")
    exit(-1)

#print DWF version
version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("DWF Version: "+str(version.value))

# enumerate connected devices
dwf.FDwfEnum(devidDiscovery2, byref(cDevice))

#open Analog Studio
for idevice in range(0, cDevice.value):
    dwf.FDwfEnumDeviceName (c_int(idevice), devicename)
    dwf.FDwfEnumSN (c_int(idevice), serialnum)
    dwf.FDwfEnumDeviceType (c_int(idevice), byref(iDevId), byref(iDevRev))
    if iDevId.value != devidDiscovery2.value:
        print("Device id "+str(iDevId.value)+" is not equal to "+str(devidDiscovery2.value))
        continue

    others = create_string_buffer(128)
    print("------------------------------")
    print("Device "+str(idevice)+" : ")
    print("\tName: " + str(devicename.value.decode()) + " " + str(serialnum.value.decode()))
    print("\tID: "+str(iDevId.value)+" rev: "+chr(0x40+(iDevRev.value&0xF))+" "+hex(iDevRev.value))
    dwf.FDwfDeviceOpen(c_int(idevice), byref(hdwf))
    if hdwf.value == 0:
        dwf.FDwfGetLastErrorMsg(szerr)
        print(str(szerr.value))
        dwf.FDwfDeviceCloseAll()
        sys.exit(0)

if hdwf.value == 0:
    print("No device found")
    sys.exit(0)

subject = -1
cycle = -1
while True:
    print("Enter subject no")
    line = input()
    if line.isnumeric():
        subject = int(line)
        break
    else:
        print("Try again")

while True:
    print("Enter cycle no")
    line = input()
    if line.isnumeric():
        cycle = int(line)
        break
    else:
        print("Try again")

def sigint_handler(signum, frame):
    dwf.FDwfDigitalOutReset(hdwf)
    dwf.FDwfDeviceCloseAll()
    sys.exit(-1)
signal.signal(signal.SIGINT, sigint_handler)

dwf.FDwfDigitalOutReset(hdwf)
# Turn off auto-configuration
dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(0))

exp = Experiment(int(hzAcq.value), red_pin, yellow_pin, green_pin, hdwf)
exp.read_instructions(script)
nSamples = int(exp.experiment_duration())
rgdSamples = (c_double*nSamples)()

#Reading clock frequency
#c_double(100 000 000.0)
hzSys = c_double()
dwf.FDwfDigitalOutInternalClockInfo(hdwf, byref(hzSys))

#set up acquisition
dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(0), c_int(1))
dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(0), c_double(5))
dwf.FDwfAnalogInAcquisitionModeSet(hdwf, acqmodeRecord)
dwf.FDwfAnalogInFrequencySet(hdwf, hzAcq)
dwf.FDwfAnalogInRecordLengthSet(hdwf, c_double(nSamples/hzAcq.value)) # -1 infinite record length
dwf.FDwfAnalogInConfigure(hdwf, c_int(1), c_int(0))

#wait at least 2 seconds for the offset to stabilize
print(comment)
time.sleep(3)

#set up LEDs
led.configure_LED(hdwf, red_pin, int(hzAcq.value))
led.configure_LED(hdwf, yellow_pin, int(hzAcq.value))
led.configure_LED(hdwf, green_pin, int(hzAcq.value))

#set up audio
audio.configure_audio(hdwf, channel)

print("Starting oscilloscope")
dwf.FDwfAnalogInConfigure(hdwf, c_int(0), c_int(1))

cSamples = 0
state = exp.init_state()
exp.actual_experiment_state(state, cSamples)

while cSamples < nSamples:
    dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))
    if cSamples == 0 and (sts == DwfStateConfig or sts == DwfStatePrefill or sts == DwfStateArmed) :
        # Acquisition not yet started.
        continue

    dwf.FDwfAnalogInStatusRecord(hdwf, byref(cAvailable), byref(cLost), byref(cCorrupted))
    
    cSamples += cLost.value

    if cLost.value :
        fLost = 1
    if cCorrupted.value :
        fCorrupted = 1

    if cAvailable.value==0 :
        continue

    if cSamples+cAvailable.value > nSamples :
        cAvailable = c_int(nSamples-cSamples)
    
    dwf.FDwfAnalogInStatusData(hdwf, c_int(1), byref(rgdSamples, sizeof(c_double)*cSamples), cAvailable) # get channel 2 data
    cSamples += cAvailable.value

    old_state = state
    state = exp.state(state, cSamples)
    exp.event(cSamples)
    if (old_state != state):
        exp.actual_experiment_state(state, cSamples)

if hdwf.value != 0:
    dwf.FDwfDigitalOutReset(hdwf)

# ensure all devices are closed
dwf.FDwfDeviceCloseAll()

print("Recording done")
if fLost:
    print("Samples were lost! Reduce frequency")
if fCorrupted:
    print("Samples could be corrupted! Reduce frequency")

start = 0
state = exp.init_state()
samples = []
f = open(f"subject_{subject}_medium_c{cycle}.csv", "w")
for i in range(len(rgdSamples)):
    state = exp.act_state(state, i)
    entry = (start, rgdSamples[i], state.value)
    f.write("%s,%s,%s\n" % entry)
    samples.append(entry)
    start += 1/hzAcq.value
f.close()

df = pd.DataFrame(samples, columns= ['Time (s)', 'Voltage (V)', 'State'])
exp_plot.plot_measurements(df)