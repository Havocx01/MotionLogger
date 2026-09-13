#include "LSM6DS3.h"
#include "Wire.h"
#include "math.h"

LSM6DS3 myIMU(I2C_MODE, 0x6A);

// Change this before each activity test.
const char* activity = "TYPING";

bool logging = false;

unsigned long loggingStartTime = 0;

void setup() {
  Serial.begin(115200);

  if (myIMU.begin() != 0) {
    Serial.println("Device error");
  } else {
    Serial.println("Device OK!");
  }

  Serial.println("Type 's' to start recording");
  Serial.println("Type 'x' to stop recording");
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == 's') {
      logging = true;
      loggingStartTime = millis();

      Serial.println("Recording started");
      Serial.println("time,accelX,accelY,accelZ,accelMagnitude,"
                     "gyroX,gyroY,gyroZ,gyroMagnitude,activity");
    }

    if (command == 'x') {
      logging = false;

      Serial.println("Recording stopped");
    }
  }

  if (!logging) {
    return;
  }

  unsigned long currentTime = millis() - loggingStartTime;

  float accelX = myIMU.readFloatAccelX();
  float accelY = myIMU.readFloatAccelY();
  float accelZ = myIMU.readFloatAccelZ();
  float accelMagnitude = sqrt(sq(accelX) + sq(accelY) + sq(accelZ));

  float gyroX = myIMU.readFloatGyroX();
  float gyroY = myIMU.readFloatGyroY();
  float gyroZ = myIMU.readFloatGyroZ();
  float gyroMagnitude = sqrt(sq(gyroX) + sq(gyroY) + sq(gyroZ));

  Serial.print(currentTime);
  Serial.print(",");

  Serial.print(accelX);
  Serial.print(",");

  Serial.print(accelY);
  Serial.print(",");

  Serial.print(accelZ);
  Serial.print(",");

  Serial.print(accelMagnitude);
  Serial.print(",");

  Serial.print(gyroX);
  Serial.print(",");

  Serial.print(gyroY);
  Serial.print(",");

  Serial.print(gyroZ);
  Serial.print(",");

  Serial.print(gyroMagnitude);
  Serial.print(",");

  Serial.println(activity);

  delay(50);
}
