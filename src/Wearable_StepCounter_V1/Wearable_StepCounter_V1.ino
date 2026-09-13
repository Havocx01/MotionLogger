#include "LSM6DS3.h"
#include "Wire.h"
#include "math.h"

// Create an instance of the LSM6DS3 class
LSM6DS3 myIMU(I2C_MODE, 0x6A);  // I2C device address

int stepCount = 0;
int candidateCount = 0;

bool wasAboveThreshold = false;
bool walkingConfirmed = false;

unsigned long lastCandidateTime = 0;

float gyroMagnitudes[3];

void setup() {
  // Start serial communication
  Serial.begin(115200);

  // Configure the IMU
  if (myIMU.begin() != 0) {
    Serial.println("Device error");
  } else {
    Serial.println("Device OK!");
  }
}

void loop() {
  const float thresholdValue = 0.20f;

  const unsigned long minimumCandidateGap = 250;
  const unsigned long maximumCandidateGap = 1000;

  // Read acceleration
  float accelX = myIMU.readFloatAccelX();
  float accelY = myIMU.readFloatAccelY();
  float accelZ = myIMU.readFloatAccelZ();

  // Calculate total acceleration
  float magnitude = sqrt(
    sq(accelX) +
    sq(accelY) +
    sq(accelZ)
  );

  // Remove the constant 1g caused by gravity
  float movement = fabs(magnitude - 1.0f);

  unsigned long currentTime = millis();

  // Detect a new movement peak
  if (movement > thresholdValue && !wasAboveThreshold) {
    wasAboveThreshold = true;

    // Read gyroscope
    float gyroX = myIMU.readFloatGyroX();
    float gyroY = myIMU.readFloatGyroY();
    float gyroZ = myIMU.readFloatGyroZ();

    // Calculate total rotational movement
    float gyroMagnitude = sqrt(
      sq(gyroX) +
      sq(gyroY) +
      sq(gyroZ)
    );

    if (lastCandidateTime == 0) {
      // Start a new candidate sequence
      candidateCount = 1;

      gyroMagnitudes[candidateCount - 1] = gyroMagnitude;

      Serial.println("Candidate 1 active");
      Serial.print("Gyro magnitude: ");
      Serial.println(gyroMagnitude);

    } else {
      unsigned long candidateGap =
        currentTime - lastCandidateTime;

      if (
        candidateGap >= minimumCandidateGap &&
        candidateGap <= maximumCandidateGap
      ) {
        // Continue the current candidate sequence
        candidateCount += 1;

        // Only store the first three gyro values
        if (candidateCount <= 3) {
          gyroMagnitudes[candidateCount - 1] = gyroMagnitude;
        }

        Serial.print("Candidate ");
        Serial.print(candidateCount);
        Serial.println(" active");

        Serial.print("Gyro magnitude: ");
        Serial.println(gyroMagnitude);

      } else {
        // Restart the candidate sequence
        candidateCount = 1;
        walkingConfirmed = false;

        gyroMagnitudes[0] = gyroMagnitude;

        Serial.println("Candidate 1 active");
        Serial.print("Gyro magnitude: ");
        Serial.println(gyroMagnitude);
      }
    }

    lastCandidateTime = currentTime;

    // Confirm walking after three valid candidates
    if (candidateCount >= 3 && !walkingConfirmed) {
      walkingConfirmed = true;
      stepCount += candidateCount;

      // Calculate the average gyro magnitude
      float gyroMagnitudeAverage =
        (
          gyroMagnitudes[0] +
          gyroMagnitudes[1] +
          gyroMagnitudes[2]
        ) / 3.0f;

      // Find the maximum and minimum gyro magnitudes
      float maximumGyroMagnitude = gyroMagnitudes[0];
      float minimumGyroMagnitude = gyroMagnitudes[0];

      if (gyroMagnitudes[1] > maximumGyroMagnitude) {
        maximumGyroMagnitude = gyroMagnitudes[1];
      }

      if (gyroMagnitudes[1] < minimumGyroMagnitude) {
        minimumGyroMagnitude = gyroMagnitudes[1];
      }

      if (gyroMagnitudes[2] > maximumGyroMagnitude) {
        maximumGyroMagnitude = gyroMagnitudes[2];
      }

      if (gyroMagnitudes[2] < minimumGyroMagnitude) {
        minimumGyroMagnitude = gyroMagnitudes[2];
      }

      // Calculate the spread
      float gyroMagnitudeDifference =
        maximumGyroMagnitude - minimumGyroMagnitude;

      Serial.print("Gyro average: ");
      Serial.println(gyroMagnitudeAverage);

      Serial.print("Maximum gyro magnitude: ");
      Serial.println(maximumGyroMagnitude);

      Serial.print("Minimum gyro magnitude: ");
      Serial.println(minimumGyroMagnitude);

      Serial.print("Gyro spread: ");
      Serial.println(gyroMagnitudeDifference);

      Serial.println("Walking confirmed");

      Serial.print("Step count: ");
      Serial.println(stepCount);

    } else if (walkingConfirmed) {
      stepCount += 1;

      Serial.print("Step count: ");
      Serial.println(stepCount);
    }
  }

  // Allow the next movement peak to be detected
  if (movement < thresholdValue && wasAboveThreshold) {
    wasAboveThreshold = false;
  }

  // Reset the candidate sequence after a pause
  if (
    lastCandidateTime != 0 &&
    currentTime - lastCandidateTime > maximumCandidateGap
  ) {
    candidateCount = 0;
    walkingConfirmed = false;
    lastCandidateTime = 0;
  }

  delay(50);
}