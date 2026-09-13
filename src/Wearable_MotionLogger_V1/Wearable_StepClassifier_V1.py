import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

activityFiles = {
    "WALKING_IN_PLACE": ["WALKING_IN_PLACE_01.txt", "WALKING_IN_PLACE_02.txt", "WALKING_IN_PLACE_03.txt"],
    "ARM_SWING": ["ARM_SWING_01.txt", "ARM_SWING_02.txt", "ARM_SWING_03.txt"],
    "RANDOM_MOVEMENT": ["RANDOM_MOVEMENT_01.txt", "RANDOM_MOVEMENT_02.txt", "RANDOM_MOVEMENT_03.txt"],
    "STILL": ["STILL_01.txt", "STILL_02.txt", "STILL_03.txt"],
    "TYPING": ["TYPING_01.txt", "TYPING_02.txt", "TYPING_03.txt"],
}

numericColumns = ["time", "accelX", "accelY", "accelZ", "accelMagnitude", "gyroX", "gyroY", "gyroZ", "gyroMagnitude"]

featureColumns = [
    "accelMagnitudeStd",
    "gyroMagnitudeStd",
    "accelMagnitudeRange",
    "gyroMagnitudeRange",
    "meanAccelJerk",
    "maxAccelJerk",
    "dominantAccelFrequency",
    "dominantFrequencyPowerRatio",
    "motionBandPower",
    "accelRegularity",
]


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

    print("  Valid samples:", rowsAfterCleaning, "| Removed bad rows:", rowsBeforeCleaning - rowsAfterCleaning)

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

    maximumTime = data["timeSeconds"].max()

    features = []

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

            if activity == "WALKING_IN_PLACE":
                target = 1

            else:
                target = 0

            features.append(
                {
                    "activity": activity,
                    "recording": recording,
                    "windowStart": windowStart,
                    "windowEnd": windowEnd,
                    "samples": len(window),
                    "target": target,
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
    raise ValueError("No feature windows were created.")

print("\nTotal feature windows:", len(featureData))

print("Walking windows:", len(featureData[featureData["target"] == 1]))

print("Not walking windows:", len(featureData[featureData["target"] == 0]))

featureData.to_csv("Wearable_Features_V1.csv", index=False)

print("\nSaved feature data to " "Wearable_Features_V1.csv")

allTrueLabels = []
allPredictions = []
allProbabilities = []

featureImportances = []

# Hold out one recording from each activity for testing.
for testRecording in [1, 2, 3]:
    print("\n")
    print("=" * 60)

    print("TESTING RECORDING", testRecording)

    print("=" * 60)

    trainData = featureData[featureData["recording"] != testRecording]

    testData = featureData[featureData["recording"] == testRecording]

    XTrain = trainData[featureColumns]

    yTrain = trainData["target"]

    XTest = testData[featureColumns]

    yTest = testData["target"]

    model = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2, class_weight="balanced", random_state=42)

    model.fit(XTrain, yTrain)

    predictions = model.predict(XTest)

    probabilities = model.predict_proba(XTest)[:, 1]

    accuracy = accuracy_score(yTest, predictions)

    print("\nAccuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification report:")

    print(classification_report(yTest, predictions, target_names=["NOT_WALKING", "WALKING"], zero_division=0))

    predictionResults = testData[["activity", "recording", "windowStart", "windowEnd"]].copy()

    predictionResults["actual"] = np.where(yTest.to_numpy() == 1, "WALKING", "NOT_WALKING")

    predictionResults["prediction"] = np.where(predictions == 1, "WALKING", "NOT_WALKING")

    predictionResults["walkingProbability"] = probabilities

    print("\nPredictions:")

    print(predictionResults.to_string(index=False))

    allTrueLabels.extend(yTest.to_list())

    allPredictions.extend(predictions.tolist())

    allProbabilities.extend(probabilities.tolist())

    featureImportances.append(model.feature_importances_)

print("\n")
print("=" * 60)

print("OVERALL RESULTS")

print("=" * 60)

overallAccuracy = accuracy_score(allTrueLabels, allPredictions)

print("\nOverall accuracy:", round(overallAccuracy * 100, 2), "%")

print("\nOverall classification report:")

print(classification_report(allTrueLabels, allPredictions, target_names=["NOT_WALKING", "WALKING"], zero_division=0))

matrix = confusion_matrix(allTrueLabels, allPredictions)

display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Not Walking", "Walking"])

display.plot()

plt.title("StepClassifier V1 - " "Confusion Matrix")

averageImportance = np.mean(featureImportances, axis=0)

importanceData = pd.DataFrame({"feature": featureColumns, "importance": averageImportance})
importanceData = importanceData.sort_values("importance", ascending=False)

print("\nFeature importance:")

print(importanceData.to_string(index=False))

plt.figure()

plt.barh(importanceData["feature"], importanceData["importance"])

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title("StepClassifier V1 - " "Feature Importance")

plt.gca().invert_yaxis()

plt.show()
