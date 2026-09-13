import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

activityFiles = {
    "WALKING_IN_PLACE": ["WALKING_IN_PLACE_01.txt", "WALKING_IN_PLACE_02.txt", "WALKING_IN_PLACE_03.txt"],
    "ARM_SWING": ["ARM_SWING_01.txt", "ARM_SWING_02.txt", "ARM_SWING_03.txt"],
    "RANDOM_MOVEMENT": ["RANDOM_MOVEMENT_01.txt", "RANDOM_MOVEMENT_02.txt", "RANDOM_MOVEMENT_03.txt"],
    "STILL": ["STILL_01.txt", "STILL_02.txt", "STILL_03.txt"],
    "TYPING": ["TYPING_01.txt", "TYPING_02.txt", "TYPING_03.txt"],
}

numericColumns = ["time", "accelX", "accelY", "accelZ", "accelMagnitude", "gyroX", "gyroY", "gyroZ", "gyroMagnitude"]


def loadRecording(fileName):
    print("Loading:", fileName)

    data = pd.read_csv(fileName, skiprows=1, engine="python")

    data.columns = data.columns.str.strip()

    for column in numericColumns:
        if column not in data.columns:
            raise ValueError(f"{fileName} is missing column: {column}")

    rowsBeforeCleaning = len(data)

    for column in numericColumns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=numericColumns).copy()

    rowsAfterCleaning = len(data)

    removedRows = rowsBeforeCleaning - rowsAfterCleaning

    print("  Valid samples:", rowsAfterCleaning, "| Removed bad rows:", removedRows)

    data = data.sort_values("time")
    data = data.reset_index(drop=True)

    data["timeSeconds"] = data["time"] / 1000.0

    return data


def calculateFrequencyFeatures(window):
    signal = window["accelMagnitude"].to_numpy(dtype=float)

    time = window["timeSeconds"].to_numpy(dtype=float)

    if len(signal) < 4:
        return 0.0, 0.0, 0.0

    timeDifferences = np.diff(time)
    timeDifferences = timeDifferences[timeDifferences > 0]

    if len(timeDifferences) == 0:
        return 0.0, 0.0, 0.0

    averageTimeDifference = np.mean(timeDifferences)

    if averageTimeDifference <= 0:
        return 0.0, 0.0, 0.0

    signal = signal - np.mean(signal)

    fftValues = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(len(signal), d=averageTimeDifference)

    fftPower = np.abs(fftValues) ** 2

    valid = (frequencies >= 0.5) & (frequencies <= 3.0)

    if not np.any(valid):
        return 0.0, 0.0, 0.0

    validFrequencies = frequencies[valid]

    validPower = fftPower[valid]

    motionBandPower = np.sum(validPower)

    if motionBandPower <= 0:
        return 0.0, 0.0, 0.0

    strongestIndex = np.argmax(validPower)

    dominantFrequency = validFrequencies[strongestIndex]

    dominantFrequencyPower = validPower[strongestIndex]

    dominantFrequencyPowerRatio = dominantFrequencyPower / motionBandPower

    return (float(dominantFrequency), float(dominantFrequencyPowerRatio), float(motionBandPower))


def calculateRegularity(window):
    signal = window["accelMagnitude"].to_numpy(dtype=float)

    time = window["timeSeconds"].to_numpy(dtype=float)

    if len(signal) < 4:
        return 0.0

    signal = signal - np.mean(signal)

    timeDifferences = np.diff(time)
    timeDifferences = timeDifferences[timeDifferences > 0]

    if len(timeDifferences) == 0:
        return 0.0

    averageTimeDifference = np.mean(timeDifferences)

    if averageTimeDifference <= 0:
        return 0.0

    correlation = np.correlate(signal, signal, mode="full")
    correlation = correlation[len(correlation) // 2 :]

    if correlation[0] == 0:
        return 0.0

    correlation = correlation / correlation[0]

    # Look for repeated movement between 0.3 and 1.2 seconds.
    minimumLag = int(0.3 / averageTimeDifference)

    maximumLag = int(1.2 / averageTimeDifference)

    minimumLag = max(minimumLag, 1)

    maximumLag = min(maximumLag, len(correlation) - 1)

    if minimumLag > maximumLag:
        return 0.0

    regularity = np.max(correlation[minimumLag : maximumLag + 1])

    return float(regularity)


def extractWindows(data, activity, recording):
    windowSize = 2.0

    # Skip the first second while the sensor settles.
    windowStart = 1.0

    features = []

    maximumTime = data["timeSeconds"].max()

    while windowStart + windowSize <= maximumTime:
        windowEnd = windowStart + windowSize

        window = data[(data["timeSeconds"] >= windowStart) & (data["timeSeconds"] < windowEnd)].copy()

        if len(window) > 1:
            accelMagnitudeStd = window["accelMagnitude"].std(ddof=0)

            gyroMagnitudeStd = window["gyroMagnitude"].std(ddof=0)

            accelMagnitudeRange = window["accelMagnitude"].max() - window["accelMagnitude"].min()

            gyroMagnitudeRange = window["gyroMagnitude"].max() - window["gyroMagnitude"].min()

            timeDifference = window["timeSeconds"].diff()

            accelerationDifference = window["accelMagnitude"].diff()
            accelJerk = (accelerationDifference / timeDifference).abs()
            accelJerk = accelJerk.replace([np.inf, -np.inf], np.nan).dropna()

            if len(accelJerk) > 0:
                meanAccelJerk = accelJerk.mean()

                maxAccelJerk = accelJerk.max()

            else:
                meanAccelJerk = 0.0

                maxAccelJerk = 0.0

            dominantAccelFrequency, dominantFrequencyPowerRatio, motionBandPower = calculateFrequencyFeatures(window)

            accelRegularity = calculateRegularity(window)

            features.append(
                {
                    "activity": activity,
                    "recording": recording,
                    "windowStart": windowStart,
                    "windowEnd": windowEnd,
                    "samples": len(window),
                    "accelMagnitudeStd": accelMagnitudeStd,
                    "gyroMagnitudeStd": gyroMagnitudeStd,
                    "accelMagnitudeRange": accelMagnitudeRange,
                    "gyroMagnitudeRange": gyroMagnitudeRange,
                    "meanAccelJerk": meanAccelJerk,
                    "maxAccelJerk": maxAccelJerk,
                    "dominantAccelFrequency": dominantAccelFrequency,
                    "dominantFrequencyPowerRatio": dominantFrequencyPowerRatio,
                    "motionBandPower": motionBandPower,
                    "accelRegularity": accelRegularity,
                }
            )

        windowStart += windowSize

    return features


allFeatures = []

for activity, files in activityFiles.items():
    for i, fileName in enumerate(files):
        data = loadRecording(fileName)

        features = extractWindows(data, activity, i + 1)

        allFeatures.extend(features)

featureData = pd.DataFrame(allFeatures)

if featureData.empty:
    raise ValueError("No valid feature windows were created.")

print("\nFeature windows:")

print(featureData)

print("\nAverage features by activity:")

averageFeatures = featureData.groupby("activity")[
    [
        "accelMagnitudeStd",
        "accelMagnitudeRange",
        "meanAccelJerk",
        "maxAccelJerk",
        "dominantAccelFrequency",
        "dominantFrequencyPowerRatio",
        "motionBandPower",
        "accelRegularity",
    ]
].mean()

print(averageFeatures)

plt.figure()

for activity in activityFiles.keys():
    activityData = featureData[featureData["activity"] == activity]

    plt.scatter(activityData["meanAccelJerk"], activityData["accelRegularity"], label=activity)

plt.xlabel("Mean Acceleration Jerk (g/sec)")

plt.ylabel("Acceleration Regularity")

plt.title("Motion Classification - " "Jerk vs Regularity")

plt.legend()

plt.figure()

for activity in activityFiles.keys():
    activityData = featureData[featureData["activity"] == activity]

    plt.scatter(activityData["dominantAccelFrequency"], activityData["dominantFrequencyPowerRatio"], label=activity)

plt.xlabel("Dominant Acceleration Frequency (Hz)")

plt.ylabel("Dominant Frequency Power Ratio")

plt.title("Motion Classification - " "Frequency vs Rhythm Strength")

plt.legend()

plt.figure()

for activity in activityFiles.keys():
    activityData = featureData[featureData["activity"] == activity]

    plt.scatter(activityData["dominantFrequencyPowerRatio"], activityData["accelRegularity"], label=activity)

plt.xlabel("Dominant Frequency Power Ratio")

plt.ylabel("Acceleration Regularity")

plt.title("Motion Classification - " "Rhythm Strength vs Regularity")

plt.legend()

plt.figure()

for activity in activityFiles.keys():
    activityData = featureData[featureData["activity"] == activity]

    plt.scatter(activityData["accelMagnitudeRange"], activityData["meanAccelJerk"], label=activity)

plt.xlabel("Acceleration Range (g)")

plt.ylabel("Mean Acceleration Jerk (g/sec)")

plt.title("Motion Classification - " "Range vs Jerk")

plt.legend()

plt.figure()

for activity in activityFiles.keys():
    activityData = featureData[featureData["activity"] == activity]

    plt.scatter(activityData["motionBandPower"], activityData["dominantFrequencyPowerRatio"], label=activity)

plt.xlabel("Motion Band Power")

plt.ylabel("Dominant Frequency Power Ratio")

plt.title("Motion Classification - " "Motion Power vs Rhythm Strength")

plt.legend()

plt.show()
