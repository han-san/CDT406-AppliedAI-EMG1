# Analog Studio Discovery based data collection
The implementation of tiny DSL for EMG hand movement data collection with Digilent's Analog Discovery Studio

## Board Configuration
- LEDs: red(DIO 9), yellow(DIO 8), green (DIO 10); 50 Hz each
- ADC: 5 kHZ, Analog Output Channel 2
- Audio: sine wave, 500 Hz, 3.0 V, 0 V offset, Analog Input Channel 1

## DSL
Tiny DSL to control stimuli and collect ADC readings in a synchronized manner. See `examples` directory.

### Supported states
- rest
- grip
- hold
- release

### Supported commands
- `wait <sec>`, wait for specified amount of seconds, <sec> can be a floating point number
- `led_on [R,Y,G]` - turn on an LED
- `led_off [R, Y, G]` - turn off an LED
- `audio_on` - turn on audio signal
- `audio_off` - turn off audio signal

### for-loop
```
for <num> begin
  state:
    inst
    inst
    ...
  state:
    ...
end
```

## Run
```bash
$ export PYTHONPATH=/path/to/src
$ python -m src/script/measurements /path/to/script
```
Collected readings are plotted and saved to `.csv` file.
