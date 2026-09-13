#include "LSM6DS3.h"
#include "Wire.h"
#include "math.h"
#include "string.h"

#include "nvm3_default.h"
#include "sl_status.h"

LSM6DS3 myIMU(I2C_MODE, 0x6A);

const unsigned long countdownTime = 10000;
const unsigned long recordingTime = 30000;
const unsigned long sampleInterval = 50;

const int maximumSamples = 600;

struct __attribute__((packed)) MotionSample {
  uint16_t time;

  int16_t accelX;
  int16_t accelY;
  int16_t accelZ;

  int16_t gyroX;
  int16_t gyroY;
  int16_t gyroZ;
};

const int samplesPerChunk = 280;

const int maximumChunks = (maximumSamples + samplesPerChunk - 1) / samplesPerChunk;

const uint8_t stateIdle = 0;
const uint8_t stateArmed = 1;
const uint8_t stateRecording = 2;
const uint8_t stateSaved = 3;
const uint8_t stateError = 4;

const uint8_t errorNone = 0;
const uint8_t errorHeaderWrite = 1;
const uint8_t errorChunkWrite = 2;
const uint8_t errorFinalHeader = 3;

struct __attribute__((packed)) RecordingHeader {
  uint32_t magic;

  uint8_t version;
  uint8_t state;

  uint16_t sampleCount;
  uint8_t chunkCount;

  uint8_t errorStage;
  int16_t failedChunk;
  uint32_t lastErrorCode;

  char activity[24];
};

const uint32_t recordingMagic = 0x4D4C5633;

const uint8_t storageVersion = 3;

// Keep recording keys away from the EEPROM keys.
const nvm3_ObjectKey_t headerKey = 0x0F000;

const nvm3_ObjectKey_t dataKeyBase = 0x0F100;

MotionSample samples[maximumSamples];

RecordingHeader header;

// The built-in LED is active low.
void ledOn() {
  digitalWrite(LED_BUILTIN, LOW);
}

void ledOff() {
  digitalWrite(LED_BUILTIN, HIGH);
}

void successSignal() {
  ledOff();
  delay(1000);

  for (int i = 0; i < 5; i++) {
    ledOn();
    delay(600);

    ledOff();
    delay(400);
  }

  ledOn();
}

void errorSignal() {
  for (int i = 0; i < 12; i++) {
    ledOn();
    delay(100);

    ledOff();
    delay(100);
  }
}

void resetHeader() {
  memset(&header, 0, sizeof(header));

  header.magic = recordingMagic;
  header.version = storageVersion;
  header.state = stateIdle;
  header.sampleCount = 0;
  header.chunkCount = 0;
  header.errorStage = errorNone;
  header.failedChunk = -1;
  header.lastErrorCode = 0;
  header.activity[0] = '\0';
}

sl_status_t writeNvmObject(nvm3_ObjectKey_t key, const void* data, size_t length) {
  sl_status_t status = nvm3_writeData(nvm3_defaultHandle, key, data, length);

  if (status == SL_STATUS_OK) {
    return status;
  }

  if (nvm3_repackNeeded(nvm3_defaultHandle)) {
    sl_status_t repackStatus = nvm3_repack(nvm3_defaultHandle);

    if (repackStatus != SL_STATUS_OK) {
      return repackStatus;
    }

    status = nvm3_writeData(nvm3_defaultHandle, key, data, length);
  }

  return status;
}

bool saveHeader() {
  sl_status_t status = writeNvmObject(headerKey, &header, sizeof(header));

  return (status == SL_STATUS_OK);
}

bool loadHeader() {
  uint32_t objectType = 0;

  size_t objectLength = 0;

  sl_status_t status = nvm3_getObjectInfo(nvm3_defaultHandle, headerKey, &objectType, &objectLength);

  if (status != SL_STATUS_OK) {
    resetHeader();

    return saveHeader();
  }

  if (objectType != NVM3_OBJECTTYPE_DATA || objectLength != sizeof(header)) {
    resetHeader();

    return saveHeader();
  }

  status = nvm3_readData(nvm3_defaultHandle, headerKey, &header, sizeof(header));

  if (status != SL_STATUS_OK) {
    resetHeader();

    return saveHeader();
  }

  if (header.magic != recordingMagic || header.version != storageVersion || header.sampleCount > maximumSamples || header.chunkCount > maximumChunks ||
      header.state > stateError) {
    resetHeader();

    return saveHeader();
  }

  header.activity[sizeof(header.activity) - 1] = '\0';

  return true;
}

void deleteOldChunks() {
  Serial.println("Cleaning current storage...");

  for (int i = 0; i < 10; i++) {
    nvm3_deleteObject(nvm3_defaultHandle, dataKeyBase + i);
  }

  // Remove chunks left by V2.2 recordings.
  for (int i = 0; i < 50; i++) {
    nvm3_deleteObject(nvm3_defaultHandle, 0x0F200 + i);
  }

  // Remove the old storage test object.
  nvm3_deleteObject(nvm3_defaultHandle, 0x0F080);

  if (nvm3_repackNeeded(nvm3_defaultHandle)) {
    Serial.println("Repacking storage...");

    sl_status_t status = nvm3_repack(nvm3_defaultHandle);

    Serial.print("Repack result: 0x");
    Serial.println((uint32_t)status, HEX);
  }

  Serial.println("Storage cleanup complete.");
}

const char* stateName() {
  if (header.state == stateIdle) {
    return "IDLE";
  }

  if (header.state == stateArmed) {
    return "ARMED";
  }

  if (header.state == stateRecording) {
    return "INTERRUPTED / INCOMPLETE";
  }

  if (header.state == stateSaved) {
    return "SAVED";
  }

  if (header.state == stateError) {
    return "ERROR";
  }

  return "UNKNOWN";
}

const char* errorStageName() {
  if (header.errorStage == errorNone) {
    return "NONE";
  }

  if (header.errorStage == errorHeaderWrite) {
    return "START HEADER";
  }

  if (header.errorStage == errorChunkWrite) {
    return "DATA CHUNK";
  }

  if (header.errorStage == errorFinalHeader) {
    return "FINAL HEADER";
  }

  return "UNKNOWN";
}

void printMenu() {
  Serial.println();
  Serial.println("================================");
  Serial.println("       MOTION LOGGER V2.3");
  Serial.println("================================");
  Serial.println();
  Serial.println("1 - Arm WALKING recording");
  Serial.println("2 - Arm ARM_SWING recording");
  Serial.println("3 - Dump saved recording");
  Serial.println("4 - Clear recording");
  Serial.println("5 - Show status");
  Serial.println("6 - Storage test");
  Serial.println("M - Show menu");
  Serial.println();
  Serial.println("Choose an option:");
}

void printStatus() {
  Serial.println();
  Serial.println("---------- STATUS ----------");
  Serial.print("State: ");
  Serial.println(stateName());
  Serial.print("Sample size: ");
  Serial.print(sizeof(MotionSample));
  Serial.println(" bytes");
  Serial.print("Samples per chunk: ");
  Serial.println(samplesPerChunk);
  Serial.print("Maximum chunks: ");
  Serial.println(maximumChunks);
  Serial.print("NVM3 max object size: ");
  Serial.print(nvm3_defaultHandle->maxObjectSize);
  Serial.println(" bytes");

  if (header.state == stateSaved) {
    Serial.println("Saved recording: YES");
    Serial.print("Activity: ");
    Serial.println(header.activity);
    Serial.print("Samples: ");
    Serial.println(header.sampleCount);
    Serial.print("Chunks: ");
    Serial.println(header.chunkCount);
  } else {
    Serial.println("Saved recording: NO");
  }

  if (header.state == stateArmed) {
    Serial.print("Armed activity: ");
    Serial.println(header.activity);
  }

  if (header.state == stateError) {
    Serial.println();
    Serial.println("SAVE FAILURE DETAILS:");
    Serial.print("Stage: ");
    Serial.println(errorStageName());
    Serial.print("Failed chunk: ");
    Serial.println(header.failedChunk);
    Serial.print("NVM3 error code: 0x");
    Serial.println(header.lastErrorCode, HEX);
  }

  Serial.println("----------------------------");
}

void clearRecording() {
  Serial.println();
  Serial.println("Clearing old recording...");

  deleteOldChunks();

  resetHeader();

  if (saveHeader()) {
    Serial.println("Recording cleared.");
  } else {
    Serial.println("ERROR: Could not save clean header.");
  }
}

void storageTest() {
  Serial.println();
  Serial.println("Running NVM3 storage test...");

  uint8_t testData[224];

  for (int i = 0; i < 224; i++) {
    testData[i] = (uint8_t)i;
  }

  const nvm3_ObjectKey_t testKey = 0x0F080;

  sl_status_t status = writeNvmObject(testKey, testData, sizeof(testData));

  Serial.print("Write result: 0x");
  Serial.println((uint32_t)status, HEX);

  if (status != SL_STATUS_OK) {
    Serial.println("STORAGE TEST FAILED.");

    return;
  }

  uint8_t readBack[224];

  status = nvm3_readData(nvm3_defaultHandle, testKey, readBack, sizeof(readBack));

  Serial.print("Read result: 0x");
  Serial.println((uint32_t)status, HEX);

  if (status != SL_STATUS_OK) {
    Serial.println("STORAGE TEST FAILED.");

    return;
  }

  bool matches = memcmp(testData, readBack, sizeof(testData)) == 0;

  if (matches) {
    Serial.println("STORAGE TEST PASSED.");
  } else {
    Serial.println("STORAGE TEST FAILED: data mismatch.");
  }

  nvm3_deleteObject(nvm3_defaultHandle, testKey);
}

void armRecording(const char* activityName) {
  if (header.state == stateSaved) {
    Serial.println();
    Serial.println("A recording is already saved.");
    Serial.println("Dump it, then clear it first.");

    return;
  }

  if (header.state == stateRecording || header.state == stateError) {
    Serial.println();
    Serial.println("Previous recording is incomplete.");
    Serial.println("Use option 4 first.");

    return;
  }

  deleteOldChunks();

  if (nvm3_repackNeeded(nvm3_defaultHandle)) {
    Serial.println("Preparing storage...");

    sl_status_t status = nvm3_repack(nvm3_defaultHandle);

    if (status != SL_STATUS_OK) {
      Serial.print("Storage preparation failed: 0x");
      Serial.println((uint32_t)status, HEX);

      return;
    }
  }

  header.state = stateArmed;
  header.sampleCount = 0;
  header.chunkCount = 0;
  header.errorStage = errorNone;
  header.failedChunk = -1;
  header.lastErrorCode = 0;

  strncpy(header.activity, activityName, sizeof(header.activity) - 1);

  header.activity[sizeof(header.activity) - 1] = '\0';

  sl_status_t status = writeNvmObject(headerKey, &header, sizeof(header));

  if (status != SL_STATUS_OK) {
    Serial.print("Could not arm. NVM3 error: 0x");
    Serial.println((uint32_t)status, HEX);

    return;
  }

  Serial.println();
  Serial.println("================================");
  Serial.println("        RECORDING ARMED");
  Serial.println("================================");
  Serial.print("Activity: ");
  Serial.println(header.activity);
  Serial.println();
  Serial.println("Unplug from laptop.");
  Serial.println("Connect to phone/power source.");
  Serial.println();
  Serial.println("10 sec countdown");
  Serial.println("30 sec recording");
  Serial.println("then saving");
  Serial.println("5 slow flashes + long solid light = FULLY SAVED");
}

void countdown() {
  for (int second = 10; second > 0; second--) {
    ledOn();
    delay(200);

    ledOff();
    delay(800);
  }
}

bool saveMotionData(int sampleCount) {
  int chunkCount = (sampleCount + samplesPerChunk - 1) / samplesPerChunk;

  for (int chunkIndex = 0; chunkIndex < chunkCount; chunkIndex++) {
    int startSample = chunkIndex * samplesPerChunk;

    int remaining = sampleCount - startSample;

    int count = remaining > samplesPerChunk ? samplesPerChunk : remaining;

    size_t bytes = count * sizeof(MotionSample);

    nvm3_ObjectKey_t key = dataKeyBase + chunkIndex;

    ledOn();
    delay(40);

    sl_status_t status = writeNvmObject(key, &samples[startSample], bytes);

    ledOff();
    delay(40);

    if (status != SL_STATUS_OK) {
      header.state = stateError;
      header.errorStage = errorChunkWrite;
      header.failedChunk = chunkIndex;
      header.lastErrorCode = (uint32_t)status;

      writeNvmObject(headerKey, &header, sizeof(header));

      return false;
    }
  }

  header.sampleCount = sampleCount;
  header.chunkCount = chunkCount;

  return true;
}

bool recordMotion() {
  header.state = stateRecording;
  header.sampleCount = 0;
  header.chunkCount = 0;
  header.errorStage = errorNone;
  header.failedChunk = -1;
  header.lastErrorCode = 0;

  sl_status_t status = writeNvmObject(headerKey, &header, sizeof(header));

  if (status != SL_STATUS_OK) {
    header.state = stateError;
    header.errorStage = errorHeaderWrite;
    header.lastErrorCode = (uint32_t)status;

    errorSignal();

    return false;
  }

  int sampleCount = 0;

  unsigned long recordingStartTime = millis();

  unsigned long nextSampleTime = recordingStartTime;

  ledOn();

  while (millis() - recordingStartTime < recordingTime) {
    unsigned long currentTime = millis();

    if (currentTime < nextSampleTime) {
      continue;
    }

    if (sampleCount >= maximumSamples) {
      break;
    }

    float accelX = myIMU.readFloatAccelX();
    float accelY = myIMU.readFloatAccelY();
    float accelZ = myIMU.readFloatAccelZ();

    float gyroX = myIMU.readFloatGyroX();
    float gyroY = myIMU.readFloatGyroY();
    float gyroZ = myIMU.readFloatGyroZ();

    samples[sampleCount].time = (uint16_t)(currentTime - recordingStartTime);
    samples[sampleCount].accelX = (int16_t)(accelX * 1000.0f);
    samples[sampleCount].accelY = (int16_t)(accelY * 1000.0f);
    samples[sampleCount].accelZ = (int16_t)(accelZ * 1000.0f);
    samples[sampleCount].gyroX = (int16_t)(gyroX * 10.0f);
    samples[sampleCount].gyroY = (int16_t)(gyroY * 10.0f);
    samples[sampleCount].gyroZ = (int16_t)(gyroZ * 10.0f);

    sampleCount++;

    nextSampleTime += sampleInterval;
  }

  ledOff();

  if (!saveMotionData(sampleCount)) {
    errorSignal();

    return false;
  }

  header.state = stateSaved;
  header.errorStage = errorNone;
  header.failedChunk = -1;
  header.lastErrorCode = 0;

  status = writeNvmObject(headerKey, &header, sizeof(header));

  if (status != SL_STATUS_OK) {
    header.state = stateError;
    header.errorStage = errorFinalHeader;
    header.lastErrorCode = (uint32_t)status;

    writeNvmObject(headerKey, &header, sizeof(header));

    errorSignal();

    return false;
  }

  successSignal();

  return true;
}

void printSample(const MotionSample& sample) {
  float accelX = sample.accelX / 1000.0f;
  float accelY = sample.accelY / 1000.0f;
  float accelZ = sample.accelZ / 1000.0f;

  float gyroX = sample.gyroX / 10.0f;
  float gyroY = sample.gyroY / 10.0f;
  float gyroZ = sample.gyroZ / 10.0f;

  float accelMagnitude = sqrt(sq(accelX) + sq(accelY) + sq(accelZ));

  float gyroMagnitude = sqrt(sq(gyroX) + sq(gyroY) + sq(gyroZ));

  Serial.print(sample.time);
  Serial.print(",");

  Serial.print(accelX, 3);
  Serial.print(",");

  Serial.print(accelY, 3);
  Serial.print(",");

  Serial.print(accelZ, 3);
  Serial.print(",");

  Serial.print(accelMagnitude, 3);
  Serial.print(",");

  Serial.print(gyroX, 1);
  Serial.print(",");

  Serial.print(gyroY, 1);
  Serial.print(",");

  Serial.print(gyroZ, 1);
  Serial.print(",");

  Serial.print(gyroMagnitude, 2);
  Serial.print(",");

  Serial.println(header.activity);
}

void dumpRecording() {
  if (header.state != stateSaved) {
    Serial.println();
    Serial.println("No complete saved recording.");

    return;
  }

  Serial.println();
  Serial.println("Recording started");
  Serial.println("time,accelX,accelY,accelZ,accelMagnitude,"
                 "gyroX,gyroY,gyroZ,gyroMagnitude,activity");

  MotionSample chunk[samplesPerChunk];

  int remaining = header.sampleCount;

  int printed = 0;

  for (int chunkIndex = 0; chunkIndex < header.chunkCount; chunkIndex++) {
    int count = remaining > samplesPerChunk ? samplesPerChunk : remaining;

    size_t bytes = count * sizeof(MotionSample);

    nvm3_ObjectKey_t key = dataKeyBase + chunkIndex;

    uint32_t objectType = 0;

    size_t objectLength = 0;

    sl_status_t status = nvm3_getObjectInfo(nvm3_defaultHandle, key, &objectType, &objectLength);

    if (status != SL_STATUS_OK || objectType != NVM3_OBJECTTYPE_DATA || objectLength != bytes) {
      Serial.println();
      Serial.print("ERROR: Missing/corrupt chunk ");
      Serial.println(chunkIndex);

      return;
    }

    status = nvm3_readData(nvm3_defaultHandle, key, chunk, bytes);

    if (status != SL_STATUS_OK) {
      Serial.println();
      Serial.print("ERROR reading chunk ");
      Serial.print(chunkIndex);
      Serial.print(". Code: 0x");
      Serial.println((uint32_t)status, HEX);

      return;
    }

    for (int i = 0; i < count; i++) {
      printSample(chunk[i]);

      printed++;
    }

    remaining -= count;
  }

  Serial.println("Recording stopped");
  Serial.println();
  Serial.print("Samples dumped: ");
  Serial.println(printed);
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);

  ledOff();

  Serial.begin(115200);

  if (myIMU.begin() != 0) {
    while (true) {
      errorSignal();

      delay(500);
    }
  }

  if (!loadHeader()) {
    while (true) {
      errorSignal();

      delay(500);
    }
  }

  if (header.state == stateArmed) {
    countdown();

    recordMotion();

    return;
  }

  unsigned long start = millis();

  while (!Serial && millis() - start < 5000) {
    delay(10);
  }

  printStatus();

  printMenu();
}

void loop() {
  if (Serial.available() <= 0) {
    return;
  }

  char command = Serial.read();

  if (command == '\n' || command == '\r') {
    return;
  }

  if (command == '1') {
    armRecording("WALKING");
  } else if (command == '2') {
    armRecording("ARM_SWING");
  } else if (command == '3') {
    dumpRecording();
  } else if (command == '4') {
    clearRecording();

    printMenu();
  } else if (command == '5') {
    printStatus();
  } else if (command == '6') {
    storageTest();
  } else if (command == 'm' || command == 'M') {
    printMenu();
  } else {
    Serial.println();
    Serial.println("Unknown option.");

    printMenu();
  }
}
