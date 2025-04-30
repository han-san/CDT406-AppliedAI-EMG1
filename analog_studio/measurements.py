from ctypes import *
import sys
import time
from dwfconstants import *
import matplotlib.pyplot as plt
import numpy

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

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
nSamples = 70000
rgdSamples = (c_double*nSamples)()
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
fLost = 0
fCorrupted = 0

red_one = 3.5
yellow_one = 4.0
green_one = 4.5
led_over_one = 5.0
red_two = 7.5
yellow_two = 8.0
green_two = 8.5
led_over_two = 9.0

def configure_LED(pin):
    # prescaler to 2kHz, SystemFrequency/1kHz/2
    dwf.FDwfDigitalOutDividerSet(hdwf, c_int(pin), c_int(int(hzSys.value/50/2)))
    # 1 tick low, 1 tick high
    dwf.FDwfDigitalOutCounterSet(hdwf, c_int(pin), c_int(1), c_int(1))

def turnon_LED(pin):
    # 100 Hz pulse on IO 
    dwf.FDwfDigitalOutEnableSet(hdwf, c_int(pin), c_int(1))
    # Generate pattern
    dwf.FDwfDigitalOutConfigure(hdwf, c_int(1))

def turnoff_LED(pin):
    dwf.FDwfDigitalOutEnableSet(hdwf, c_int(pin), c_int(0))
    dwf.FDwfDigitalOutConfigure(hdwf, c_int(0))

#print DWF version
version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("DWF Version: "+str(version.value))

# enumerate connected devices
dwf.FDwfEnum(devidDiscovery2, byref(cDevice))
# dwf.FDwfEnum(c_int(0), byref(cDevice))

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

dwf.FDwfDigitalOutReset(hdwf)
# Turn off auto-configuration
dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(0))

cBufMax = c_int()
dwf.FDwfAnalogInBufferSizeInfo(hdwf, 0, byref(cBufMax))
print("Device buffer size: "+str(cBufMax.value)) 

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
time.sleep(3)

#set up LEDs
configure_LED(red_pin)
configure_LED(yellow_pin)
configure_LED(green_pin)

print("Starting oscilloscope")
dwf.FDwfAnalogInConfigure(hdwf, c_int(0), c_int(1))

cSamples = 0
red_f = False # red LED is ON
yellow_f = False # yellow LED is ON
green_f = False # green LED is ON
over_f = True # all LEDs are OFF
hold_start = 0 # end of the first LED cycle
hold_finish = 0 # end of the second LED cycle

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

    freq = hzAcq.value

    if (((cSamples >= red_one * freq and cSamples < yellow_one * freq) or (cSamples >= red_two * freq and cSamples < yellow_two * freq)) and not red_f):
        turnon_LED(red_pin)
        # print('cSamples Red: '+str(cSamples))
        red_f = True
        over_f = False
    elif (((cSamples >= yellow_one * freq and cSamples < green_one * freq) or (cSamples >= yellow_two * freq and cSamples < green_two * freq)) and not yellow_f):
        turnoff_LED(red_pin)
        turnon_LED(yellow_pin)
        # print('cSamples Yellow: '+str(cSamples))
        yellow_f = True
    elif (((cSamples >= green_one * freq and cSamples < led_over_one * freq) or (cSamples >= green_two * freq and cSamples < led_over_two * freq)) and not green_f):
        turnoff_LED(yellow_pin)
        turnon_LED(green_pin)
        green_f = True
        # print('cSamples Green: '+str(cSamples))
    elif (((cSamples >= led_over_one * freq and cSamples < red_two * freq) or (cSamples >= led_over_two * freq)) and not over_f):
        turnoff_LED(green_pin)
        red_f = False
        yellow_f = False
        green_f = False
        over_f = True
        # print('cSamples Over: '+str(cSamples))
        if (cSamples < led_over_two * freq):
            hold_start = cSamples
        else:
            hold_finish = cSamples


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
f = open(time.strftime("%d%m-%H%M%S", time.localtime())+"record.csv", "w")
# for v in rgdSamples:
#     f.write("%s,%s\n" % (start,v))
#     start += 1 / freq
# f.close()
state = 0 # 0 - rest, 1 - grip, 2 - hold, 3 - release
grip_interval = 0.5 * freq
release_interval = 0.5 * freq
for i in range(len(rgdSamples)):
    if (i >= hold_start and i < hold_start + grip_interval):
        state = 1
    elif (i >= hold_start + grip_interval and i < hold_finish):
        state = 2
    elif (i >= hold_finish and i < hold_finish + release_interval):
        state = 3
    else:
        state = 0
    f.write("%s,%s,%s\n" % (start, rgdSamples[i], state))
    start += 1/freq
f.close()
  
plt.plot(numpy.fromiter(rgdSamples, dtype = float)[1::10])
plt.show()