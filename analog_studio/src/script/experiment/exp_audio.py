from ctypes import c_int
from script.dwfconstants import *
import sys

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

def configure_audio(hdwf, channel: int):
    dwf.FDwfAnalogOutNodeFunctionSet(hdwf, c_int(channel), AnalogOutNodeCarrier, funcSine)
    dwf.FDwfAnalogOutNodeFrequencySet(hdwf, c_int(channel), AnalogOutNodeCarrier, c_double(500))
    dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, c_int(channel), AnalogOutNodeCarrier, c_double(3.0))
    dwf.FDwfAnalogOutNodeOffsetSet(hdwf, c_int(channel), AnalogOutNodeCarrier, c_double(0))

def turnon_audio(hdwf, channel: int):
    dwf.FDwfAnalogOutNodeEnableSet(hdwf, c_int(channel), AnalogOutNodeCarrier, c_int(1))
    dwf.FDwfAnalogOutConfigure(hdwf, c_int(channel), c_int(1))

def turnoff_audio(hdwf, channel: int):
    dwf.FDwfAnalogOutNodeEnableSet(hdwf, c_int(channel), AnalogOutNodeCarrier, c_int(0))
    dwf.FDwfAnalogOutConfigure(hdwf, c_int(channel), c_int(0))