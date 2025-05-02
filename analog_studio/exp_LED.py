from ctypes import c_int
from dwfconstants import *
import sys

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

def configure_LED(hdwf, pin: int, freq):
    # prescaler to 100 Hz
    dwf.FDwfDigitalOutDividerSet(hdwf, c_int(pin), c_int(int(freq/50/2)))
    # 1 tick low, 1 tick high
    dwf.FDwfDigitalOutCounterSet(hdwf, c_int(pin), c_int(1), c_int(1))

def turnon_LED(hdwf, pin: int):
    # 100 Hz pulse on IO 
    dwf.FDwfDigitalOutEnableSet(hdwf, c_int(pin), c_int(1))
    # Generate pattern
    dwf.FDwfDigitalOutConfigure(hdwf, c_int(1))

def turnoff_LED(hdwf, pin: int):
    dwf.FDwfDigitalOutEnableSet(hdwf, c_int(pin), c_int(0))
    dwf.FDwfDigitalOutConfigure(hdwf, c_int(0))