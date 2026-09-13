# Wearable step counter

A wearable step counter using accelerometer and gyroscope data. The main thing I'm trying to get right is telling the difference between actual walking and just swinging your arm.

I started with a simple step counter, then added motion logging so I could record different movements and compare them in Python.

The electronics are simple:

- **LSM6DS3 motion sensor** - has an accelerometer and a gyroscope. The accelerometer measures acceleration, including gravity. The gyroscope measures how fast the sensor is rotating. Using both helps compare walking with other arm movements.
- **Microcontroller board** - runs the Arduino code and reads the sensor over I2C, which is the connection they use to send data. V2 saves recordings in the board's flash memory so they are still there after unplugging it.
- **Built-in LED** - shows the recording progress. Five slow flashes followed by a solid light mean the recording has saved.

What's in `src`:

- `Wearable_StepCounter_V1` - the original step counter. Looks for movement peaks and confirms walking after three valid candidates.
- `Wearable_MotionLogger_V1` - the first logger and Python scripts for comparing walking in place, arm swings, typing, staying still, and random movement.
- `Wearable_MotionLogger_V2` - records motion to the board's storage so it can run away from the laptop. Includes the newer walking classifier and a script for finding individual steps.

The recordings, CSV results, and plots are included. `Types_of_walking.txt` lists the movements used for the tests.

To run the newer Python scripts from the project folder:

```sh
python -m pip install numpy pandas matplotlib scikit-learn
cd src/Wearable_MotionLogger_V2
python Wearable_StepClassifier_V2_Enhanced.py
python Wearable_StepEventDetector_V1.py
```

The classifier compares walking and arm swings. The step detector works through the walking recordings and saves step times, counts, and plots. Running these again replaces the existing results.

For the Arduino sketches, open the `.ino` file in Arduino IDE. They use the LSM6DS3 library and Serial Monitor at 115200 baud. V2 also needs the board's NVM3 storage support. Its serial menu has options to arm, dump, and clear a recording.

Still working on checking the detected steps against recordings with a known step count.
