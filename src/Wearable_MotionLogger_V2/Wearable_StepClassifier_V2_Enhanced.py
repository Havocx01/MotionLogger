from pathlib import Path
from io import StringIO
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

scriptDir = Path(__file__).resolve().parent
dataDir = scriptDir / "Data"

outputFeatures = scriptDir / "Wearable_Features_V2_Enhanced.csv"
outputPredictions = scriptDir / "Wearable_Predictions_V2_Enhanced.csv"
outputRecordingResults = scriptDir / "Wearable_RecordingResults_V2_Enhanced.csv"
outputImportance = scriptDir / "Wearable_FeatureImportance_V2_Enhanced.csv"
outputImportancePlot = scriptDir / "Wearable_FeatureImportance_V2_Enhanced.png"

windowSeconds = 2.0
ignoreFirstSeconds = 1.0

minSamplesPerWindow = 30

minWalkingFrequency = 0.5
maxWalkingFrequency = 3.0

maxSpectralFrequency = 8.0

numericColumns = ["time", "accelX", "accelY", "accelZ", "accelMagnitude", "gyroX", "gyroY", "gyroZ", "gyroMagnitude"]


def findFolder(parent, possibleNames):
    folders = {item.name.lower(): item for item in parent.iterdir() if item.is_dir()}

    for name in possibleNames:
        match = folders.get(name.lower())

        if match is not None:
            return match

    raise FileNotFoundError(f"Could not find folder inside {parent}. " f"Tried: {possibleNames}")


def stripSerialPrefix(line):
    return re.sub(r"^\s*\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\s*->\s*", "", line).strip()


def loadRecording(filePath, activityLabel, target):
    print(f"Loading: {activityLabel} / {filePath.name}")

    rawLines = filePath.read_text(encoding="utf-8", errors="ignore").splitlines()

    lines = [stripSerialPrefix(line) for line in rawLines]

    headerIndex = None

    for index, line in enumerate(lines):
        if line.startswith("time,") and "accelX" in line and "gyroMagnitude" in line:
            headerIndex = index
            break

    if headerIndex is None:
        raise ValueError(f"{filePath.name}: CSV header not found.")

    csvLines = [lines[headerIndex]]

    for line in lines[headerIndex + 1 :]:
        if line.startswith("Recording stopped"):
            break

        if line.count(",") >= 9:
            csvLines.append(line)

    if len(csvLines) <= 1:
        raise ValueError(f"{filePath.name}: no sample rows found.")

    data = pd.read_csv(StringIO("\n".join(csvLines)))

    data.columns = data.columns.str.strip()

    for column in numericColumns:
        if column not in data.columns:
            raise ValueError(f"{filePath.name} missing: {column}")

    rowsBefore = len(data)

    for column in numericColumns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=numericColumns).copy()
    data = data.sort_values("time").reset_index(drop=True)

    rowsAfter = len(data)

    data["timeSeconds"] = data["time"] / 1000.0

    data["recording"] = filePath.stem
    data["activityLabel"] = activityLabel
    data["target"] = target

    # Include the activity because both folders can use the same filename.
    data["recordingGroup"] = activityLabel + "__" + filePath.stem

    print(f"  Samples: {rowsAfter} | " f"Removed: {rowsBefore - rowsAfter}")

    return data


def safeStd(values):
    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        return 0.0

    return float(np.std(values, ddof=1))


def safeRange(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return 0.0

    return float(np.max(values) - np.min(values))


def safeMean(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return 0.0

    return float(np.mean(values))


def safePercentile(values, percentile):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return 0.0

    return float(np.percentile(values, percentile))


def safeCorrelation(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    length = min(len(a), len(b))

    if length < 3:
        return 0.0

    a = a[:length]
    b = b[:length]

    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])


def averageSampleTime(window):
    timeValues = window["timeSeconds"].to_numpy(dtype=float)

    differences = np.diff(timeValues)
    differences = differences[differences > 0]

    if len(differences) == 0:
        return np.nan

    return float(np.mean(differences))


def calculateRegularity(signal, averageDt, minimumPeriod=0.3, maximumPeriod=1.2):
    signal = np.asarray(signal, dtype=float)

    if len(signal) < 4 or not np.isfinite(averageDt) or averageDt <= 0:
        return 0.0

    centered = signal - np.mean(signal)

    energy = np.sum(centered**2)

    if energy <= 1e-12:
        return 0.0

    minimumLag = max(1, int(round(minimumPeriod / averageDt)))

    maximumLag = min(len(centered) - 1, int(round(maximumPeriod / averageDt)))

    if maximumLag < minimumLag:
        return 0.0

    correlations = []

    for lag in range(minimumLag, maximumLag + 1):
        numerator = np.sum(centered[:-lag] * centered[lag:])

        correlations.append(numerator / energy)

    if not correlations:
        return 0.0

    return float(np.max(correlations))


def calculateSpectralFeatures(signal, averageDt):
    signal = np.asarray(signal, dtype=float)

    defaults = {
        "dominantFrequency": 0.0,
        "dominantPowerRatio": 0.0,
        "walkingBandPower": 0.0,
        "spectralEntropy": 0.0,
        "lowBandPowerRatio": 0.0,
        "stepBandPowerRatio": 0.0,
        "upperBandPowerRatio": 0.0,
        "highFrequencyPowerRatio": 0.0,
        "secondHarmonicRatio": 0.0,
    }

    if len(signal) < 4 or not np.isfinite(averageDt) or averageDt <= 0:
        return defaults

    centered = signal - np.mean(signal)

    frequencies = np.fft.rfftfreq(len(centered), d=averageDt)

    power = np.abs(np.fft.rfft(centered)) ** 2

    validMask = (frequencies >= 0.5) & (frequencies <= maxSpectralFrequency)

    validPower = power[validMask]

    validFrequencies = frequencies[validMask]

    totalPower = float(np.sum(validPower))

    if len(validPower) == 0 or totalPower <= 1e-12:
        return defaults

    walkingMask = (validFrequencies >= minWalkingFrequency) & (validFrequencies <= maxWalkingFrequency)

    walkingPower = validPower[walkingMask]

    walkingFrequencies = validFrequencies[walkingMask]

    walkingBandPower = float(np.sum(walkingPower))

    if len(walkingPower) > 0:
        dominantIndex = int(np.argmax(walkingPower))

        dominantFrequency = float(walkingFrequencies[dominantIndex])

        dominantPower = float(walkingPower[dominantIndex])

    else:
        dominantFrequency = 0.0
        dominantPower = 0.0

    dominantPowerRatio = dominantPower / walkingBandPower if walkingBandPower > 1e-12 else 0.0

    normalizedPower = validPower / totalPower

    positivePower = normalizedPower[normalizedPower > 0]

    if len(positivePower) > 1:
        spectralEntropy = -np.sum(positivePower * np.log(positivePower)) / np.log(len(positivePower))
    else:
        spectralEntropy = 0.0

    def bandRatio(minimumFrequency, maximumFrequency):
        mask = (validFrequencies >= minimumFrequency) & (validFrequencies < maximumFrequency)

        return float(np.sum(validPower[mask]) / totalPower)

    lowBandPowerRatio = bandRatio(0.5, 1.25)

    stepBandPowerRatio = bandRatio(1.25, 2.25)

    upperBandPowerRatio = bandRatio(2.25, 3.5)

    highFrequencyPowerRatio = bandRatio(3.5, maxSpectralFrequency + 0.001)

    secondHarmonicRatio = 0.0

    if dominantFrequency > 0:
        secondFrequency = 2.0 * dominantFrequency

        if secondFrequency <= maxSpectralFrequency:
            tolerance = 0.30

            secondMask = np.abs(validFrequencies - secondFrequency) <= tolerance

            secondPower = float(np.sum(validPower[secondMask]))

            if dominantPower > 1e-12:
                secondHarmonicRatio = secondPower / dominantPower

    return {
        "dominantFrequency": dominantFrequency,
        "dominantPowerRatio": dominantPowerRatio,
        "walkingBandPower": walkingBandPower,
        "spectralEntropy": float(spectralEntropy),
        "lowBandPowerRatio": lowBandPowerRatio,
        "stepBandPowerRatio": stepBandPowerRatio,
        "upperBandPowerRatio": upperBandPowerRatio,
        "highFrequencyPowerRatio": highFrequencyPowerRatio,
        "secondHarmonicRatio": secondHarmonicRatio,
    }


def calculatePeakFeatures(signal, times):
    signal = np.asarray(signal, dtype=float)

    times = np.asarray(times, dtype=float)

    defaults = {
        "impactPeakCount": 0.0,
        "impactPeakRate": 0.0,
        "impactIntervalMean": 0.0,
        "impactIntervalStd": 0.0,
        "impactIntervalCV": 0.0,
        "impactPeakMean": 0.0,
    }

    if len(signal) < 5 or len(times) != len(signal):
        return defaults

    # Let the peak threshold follow the amount of movement in this window.
    threshold = np.median(signal) + 0.75 * np.std(signal)

    minimumGapSeconds = 0.20

    peakIndices = []

    lastPeakTime = -1e9

    for index in range(1, len(signal) - 1):
        if signal[index] > signal[index - 1] and signal[index] >= signal[index + 1] and signal[index] >= threshold:
            currentTime = times[index]

            if currentTime - lastPeakTime >= minimumGapSeconds:
                peakIndices.append(index)

                lastPeakTime = currentTime

    if not peakIndices:
        return defaults

    peakTimes = times[peakIndices]

    peakValues = signal[peakIndices]

    duration = times[-1] - times[0]

    peakRate = len(peakIndices) / duration if duration > 0 else 0.0

    intervals = np.diff(peakTimes)

    intervalMean = float(np.mean(intervals)) if len(intervals) > 0 else 0.0

    intervalStd = float(np.std(intervals, ddof=1)) if len(intervals) > 1 else 0.0

    intervalCV = intervalStd / intervalMean if intervalMean > 1e-12 else 0.0

    return {
        "impactPeakCount": float(len(peakIndices)),
        "impactPeakRate": float(peakRate),
        "impactIntervalMean": intervalMean,
        "impactIntervalStd": intervalStd,
        "impactIntervalCV": float(intervalCV),
        "impactPeakMean": float(np.mean(peakValues)),
    }


def calculateAccelDirectionFeatures(window):
    vectors = window[["accelX", "accelY", "accelZ"]].to_numpy(dtype=float)

    if len(vectors) < 2:
        return {"accelDirectionChangeMean": 0.0, "accelDirectionChangeStd": 0.0, "accelDirectionChangeP90": 0.0}

    first = vectors[:-1]
    second = vectors[1:]

    firstNorm = np.linalg.norm(first, axis=1)

    secondNorm = np.linalg.norm(second, axis=1)

    denominator = firstNorm * secondNorm

    valid = denominator > 1e-9

    if not np.any(valid):
        return {"accelDirectionChangeMean": 0.0, "accelDirectionChangeStd": 0.0, "accelDirectionChangeP90": 0.0}

    cosine = np.sum(first[valid] * second[valid], axis=1) / denominator[valid]
    cosine = np.clip(cosine, -1.0, 1.0)

    angles = np.degrees(np.arccos(cosine))

    return {"accelDirectionChangeMean": safeMean(angles), "accelDirectionChangeStd": safeStd(angles), "accelDirectionChangeP90": safePercentile(angles, 90)}


def extractWindowFeatures(window, recordingName, recordingGroup, activityLabel, target, windowNumber, windowStart, windowEnd):
    timeValues = window["timeSeconds"].to_numpy(dtype=float)

    averageDt = averageSampleTime(window)

    accelMagnitude = window["accelMagnitude"].to_numpy(dtype=float)

    gyroMagnitude = window["gyroMagnitude"].to_numpy(dtype=float)

    accelVectors = window[["accelX", "accelY", "accelZ"]].to_numpy(dtype=float)

    gyroVectors = window[["gyroX", "gyroY", "gyroZ"]].to_numpy(dtype=float)

    dt = np.diff(timeValues)

    validDt = dt > 0

    accelMagnitudeDifference = np.diff(accelMagnitude)

    scalarJerk = np.zeros(len(dt), dtype=float)

    scalarJerk[validDt] = np.abs(accelMagnitudeDifference[validDt]) / dt[validDt]

    scalarJerk = scalarJerk[validDt]

    accelVectorDifference = np.diff(accelVectors, axis=0)

    vectorJerk = np.zeros(len(dt), dtype=float)

    if np.any(validDt):
        vectorJerk[validDt] = np.linalg.norm(accelVectorDifference[validDt], axis=1) / dt[validDt]

    vectorJerk = vectorJerk[validDt]

    jerkTimes = timeValues[1:][validDt]

    gyroVectorDifference = np.diff(gyroVectors, axis=0)

    angularAcceleration = np.zeros(len(dt), dtype=float)

    if np.any(validDt):
        angularAcceleration[validDt] = np.linalg.norm(gyroVectorDifference[validDt], axis=1) / dt[validDt]

    angularAcceleration = angularAcceleration[validDt]

    dynamicAccel = np.abs(accelMagnitude - 1.0)

    accelSpectrum = calculateSpectralFeatures(accelMagnitude, averageDt)

    jerkSpectrum = calculateSpectralFeatures(vectorJerk, averageDt)

    accelRegularity = calculateRegularity(accelMagnitude, averageDt)

    jerkRegularity = calculateRegularity(vectorJerk, averageDt)

    gyroRegularity = calculateRegularity(gyroMagnitude, averageDt)

    peakFeatures = calculatePeakFeatures(vectorJerk, jerkTimes)

    directionFeatures = calculateAccelDirectionFeatures(window)

    accelGyroCorrelation = safeCorrelation(accelMagnitude, gyroMagnitude)
    accelStd = safeStd(accelMagnitude)

    gyroStd = safeStd(gyroMagnitude)
    gyroAccelStdRatio = gyroStd / accelStd if accelStd > 1e-9 else 0.0

    features = {
        "recording": recordingName,
        "recordingGroup": recordingGroup,
        "activityLabel": activityLabel,
        "target": target,
        "windowNumber": windowNumber,
        "windowStartSeconds": windowStart,
        "windowEndSeconds": windowEnd,
        "sampleCount": len(window),
        "accelMagnitudeStd": accelStd,
        "gyroMagnitudeStd": gyroStd,
        "accelMagnitudeRange": safeRange(accelMagnitude),
        "gyroMagnitudeRange": safeRange(gyroMagnitude),
        "meanAccelJerk": safeMean(scalarJerk),
        "maxAccelJerk": (float(np.max(scalarJerk)) if len(scalarJerk) > 0 else 0.0),
        "dominantAccelFrequency": accelSpectrum["dominantFrequency"],
        "dominantFrequencyPowerRatio": accelSpectrum["dominantPowerRatio"],
        "motionBandPower": accelSpectrum["walkingBandPower"],
        "accelRegularity": accelRegularity,
    }

    features.update(
        {
            "accelMagnitudeMean": safeMean(accelMagnitude),
            "accelMagnitudeMedian": safePercentile(accelMagnitude, 50),
            "accelMagnitudeIQR": (safePercentile(accelMagnitude, 75) - safePercentile(accelMagnitude, 25)),
            "accelMagnitudeP90": safePercentile(accelMagnitude, 90),
            "accelMagnitudeP95": safePercentile(accelMagnitude, 95),
            "dynamicAccelMean": safeMean(dynamicAccel),
            "dynamicAccelP90": safePercentile(dynamicAccel, 90),
            "dynamicAccelP95": safePercentile(dynamicAccel, 95),
            "dynamicAccelRMS": float(np.sqrt(np.mean(dynamicAccel**2))),
            "scalarJerkStd": safeStd(scalarJerk),
            "scalarJerkP90": safePercentile(scalarJerk, 90),
            "scalarJerkP95": safePercentile(scalarJerk, 95),
            "scalarJerkAbove3Rate": (float(np.mean(scalarJerk > 3.0)) if len(scalarJerk) > 0 else 0.0),
            "scalarJerkAbove5Rate": (float(np.mean(scalarJerk > 5.0)) if len(scalarJerk) > 0 else 0.0),
            "vectorJerkMean": safeMean(vectorJerk),
            "vectorJerkStd": safeStd(vectorJerk),
            "vectorJerkP90": safePercentile(vectorJerk, 90),
            "vectorJerkP95": safePercentile(vectorJerk, 95),
            "vectorJerkMax": (float(np.max(vectorJerk)) if len(vectorJerk) > 0 else 0.0),
            "vectorJerkAbove5Rate": (float(np.mean(vectorJerk > 5.0)) if len(vectorJerk) > 0 else 0.0),
            "angularAccelerationMean": safeMean(angularAcceleration),
            "angularAccelerationP90": safePercentile(angularAcceleration, 90),
            "gyroMagnitudeMean": safeMean(gyroMagnitude),
            "gyroMagnitudeP90": safePercentile(gyroMagnitude, 90),
            "gyroMagnitudeP95": safePercentile(gyroMagnitude, 95),
            "accelGyroCorrelation": accelGyroCorrelation,
            "gyroAccelStdRatio": gyroAccelStdRatio,
            "jerkRegularity": jerkRegularity,
            "gyroRegularity": gyroRegularity,
            "jerkDominantFrequency": jerkSpectrum["dominantFrequency"],
            "jerkDominantPowerRatio": jerkSpectrum["dominantPowerRatio"],
            "jerkMotionBandPower": jerkSpectrum["walkingBandPower"],
            "accelSpectralEntropy": accelSpectrum["spectralEntropy"],
            "accelLowBandPowerRatio": accelSpectrum["lowBandPowerRatio"],
            "accelStepBandPowerRatio": accelSpectrum["stepBandPowerRatio"],
            "accelUpperBandPowerRatio": accelSpectrum["upperBandPowerRatio"],
            "accelHighFrequencyPowerRatio": accelSpectrum["highFrequencyPowerRatio"],
            "accelSecondHarmonicRatio": accelSpectrum["secondHarmonicRatio"],
        }
    )

    features.update(peakFeatures)

    features.update(directionFeatures)

    return features


def extractRecordingFeatures(data):
    recordingName = data["recording"].iloc[0]

    recordingGroup = data["recordingGroup"].iloc[0]

    activityLabel = data["activityLabel"].iloc[0]

    target = int(data["target"].iloc[0])

    recordingEnd = float(data["timeSeconds"].max())

    windowStart = ignoreFirstSeconds

    windowNumber = 0

    rows = []

    while windowStart + windowSeconds <= recordingEnd:
        windowEnd = windowStart + windowSeconds

        window = data[(data["timeSeconds"] >= windowStart) & (data["timeSeconds"] < windowEnd)].copy()

        if len(window) >= minSamplesPerWindow:
            rows.append(
                extractWindowFeatures(
                    window=window,
                    recordingName=recordingName,
                    recordingGroup=recordingGroup,
                    activityLabel=activityLabel,
                    target=target,
                    windowNumber=windowNumber,
                    windowStart=windowStart,
                    windowEnd=windowEnd,
                )
            )

            windowNumber += 1

        windowStart += windowSeconds

    print(f"  Windows: {len(rows)}")

    return rows


baselineFeatures = [
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

metadataColumns = ["recording", "recordingGroup", "activityLabel", "target", "windowNumber", "windowStartSeconds", "windowEndSeconds", "sampleCount"]


def evaluateModel(features, featureColumns, modelName, modelFactory):
    X = features[featureColumns].to_numpy(dtype=float)

    y = features["target"].to_numpy(dtype=int)

    groups = features["recordingGroup"].to_numpy()

    # Keep each recording together when splitting training and test data.
    logo = LeaveOneGroupOut()

    predictions = np.zeros(len(features), dtype=int)

    probabilities = np.zeros(len(features), dtype=float)

    for foldNumber, (trainIndex, testIndex) in enumerate(logo.split(X, y, groups), start=1):
        model = modelFactory()

        model.fit(X[trainIndex], y[trainIndex])

        predictions[testIndex] = model.predict(X[testIndex])

        probabilities[testIndex] = model.predict_proba(X[testIndex])[:, 1]

    accuracy = accuracy_score(y, predictions)

    balancedAccuracy = balanced_accuracy_score(y, predictions)

    precision = precision_score(y, predictions, zero_division=0)

    recall = recall_score(y, predictions, zero_division=0)

    f1 = f1_score(y, predictions, zero_division=0)

    matrix = confusion_matrix(y, predictions, labels=[0, 1])

    print()
    print("========================================")
    print(modelName)
    print("========================================")
    print()

    print(f"Window accuracy:          {accuracy * 100:.2f}%")

    print(f"Balanced accuracy:        {balancedAccuracy * 100:.2f}%")

    print(f"Walking precision:        {precision * 100:.2f}%")

    print(f"Walking recall:           {recall * 100:.2f}%")

    print(f"Walking F1:               {f1 * 100:.2f}%")

    print()
    print("Confusion matrix [[TN, FP], [FN, TP]]:")

    print(matrix)

    result = features[metadataColumns].copy()

    result["model"] = modelName

    result["prediction"] = predictions

    result["walkingProbability"] = probabilities

    result["correct"] = predictions == y

    recordingResults = (
        result.groupby(["model", "activityLabel", "recording", "recordingGroup", "target"])
        .agg(
            windowAccuracy=("correct", "mean"),
            meanWalkingProbability=("walkingProbability", "mean"),
            predictedWalkingWindows=("prediction", "sum"),
            totalWindows=("prediction", "size"),
        )
        .reset_index()
    )

    recordingResults["recordingPrediction"] = (recordingResults["meanWalkingProbability"] >= 0.5).astype(int)

    recordingResults["recordingCorrect"] = recordingResults["recordingPrediction"] == recordingResults["target"]

    recordingAccuracy = recordingResults["recordingCorrect"].mean()

    print()
    print(f"Whole-recording accuracy: {recordingAccuracy * 100:.2f}%")

    print()
    print("Held-out recording results:")

    display = recordingResults[
        ["activityLabel", "recording", "target", "meanWalkingProbability", "windowAccuracy", "recordingPrediction", "recordingCorrect"]
    ].copy()

    display["meanWalkingProbability"] = display["meanWalkingProbability"].round(3)

    display["windowAccuracy"] = (display["windowAccuracy"] * 100).round(1)

    print(display.to_string(index=False))

    return (
        result,
        recordingResults,
        {
            "model": modelName,
            "windowAccuracy": accuracy,
            "balancedAccuracy": balancedAccuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "recordingAccuracy": recordingAccuracy,
        },
    )


def main():
    if not dataDir.exists():
        raise FileNotFoundError(f"Data folder not found: {dataDir}")

    walkingFolder = findFolder(dataDir, ["Walking", "Waiking"])

    armSwingFolder = findFolder(dataDir, ["Arm_Swing", "ArmSwing", "Arm Swing"])

    walkingFiles = sorted(walkingFolder.glob("*.txt"))

    armSwingFiles = sorted(armSwingFolder.glob("*.txt"))

    if not walkingFiles:
        raise FileNotFoundError("No Walking .txt files found.")

    if not armSwingFiles:
        raise FileNotFoundError("No Arm_Swing .txt files found.")

    print()
    print("========================================")
    print("  WEARABLE CLASSIFIER V2 - ENHANCED")
    print("========================================")
    print()

    print(f"Walking recordings: {len(walkingFiles)}")

    print(f"Arm-swing recordings: {len(armSwingFiles)}")

    print()

    recordings = []

    for filePath in walkingFiles:
        recordings.append(loadRecording(filePath, "WALKING", 1))

    for filePath in armSwingFiles:
        recordings.append(loadRecording(filePath, "ARM_SWING", 0))

    print()
    print("Extracting enhanced features...")
    print()

    featureRows = []

    for recording in recordings:
        featureRows.extend(extractRecordingFeatures(recording))

    features = pd.DataFrame(featureRows)

    if features.empty:
        raise ValueError("No feature windows were created.")

    enhancedFeatures = [column for column in features.columns if column not in metadataColumns]
    enhancedFeatures = [column for column in enhancedFeatures if pd.api.types.is_numeric_dtype(features[column])]

    features[enhancedFeatures] = features[enhancedFeatures].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    features.to_csv(outputFeatures, index=False)

    print()
    print(f"Windows: {len(features)}")

    print(f"Baseline features: {len(baselineFeatures)}")

    print(f"Enhanced features: {len(enhancedFeatures)}")

    def baselineRandomForest():
        return RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)

    def enhancedRandomForest():
        return RandomForestClassifier(
            n_estimators=800, max_depth=8, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1
        )

    def enhancedExtraTrees():
        return ExtraTreesClassifier(n_estimators=800, max_depth=8, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)

    allPredictions = []
    allRecordingResults = []
    summaryRows = []

    predictions, recordingResults, summary = evaluateModel(features, baselineFeatures, "Baseline RF - 10 features", baselineRandomForest)

    allPredictions.append(predictions)

    allRecordingResults.append(recordingResults)

    summaryRows.append(summary)

    predictions, recordingResults, summary = evaluateModel(features, enhancedFeatures, "Enhanced RF", enhancedRandomForest)

    allPredictions.append(predictions)

    allRecordingResults.append(recordingResults)

    summaryRows.append(summary)

    predictions, recordingResults, summary = evaluateModel(features, enhancedFeatures, "Enhanced ExtraTrees", enhancedExtraTrees)

    allPredictions.append(predictions)

    allRecordingResults.append(recordingResults)

    summaryRows.append(summary)

    combinedPredictions = pd.concat(allPredictions, ignore_index=True)

    combinedPredictions.to_csv(outputPredictions, index=False)

    combinedRecordingResults = pd.concat(allRecordingResults, ignore_index=True)

    combinedRecordingResults.to_csv(outputRecordingResults, index=False)

    summaryFrame = pd.DataFrame(summaryRows)

    print()
    print("========================================")
    print("MODEL COMPARISON")
    print("========================================")
    print()

    displaySummary = summaryFrame.copy()

    for column in ["windowAccuracy", "balancedAccuracy", "precision", "recall", "f1", "recordingAccuracy"]:
        displaySummary[column] = (displaySummary[column] * 100).round(2)

    print(displaySummary.to_string(index=False))

    # Use all recordings for feature importance only, after testing is complete.
    finalModel = enhancedRandomForest()

    finalModel.fit(features[enhancedFeatures].to_numpy(dtype=float), features["target"].to_numpy(dtype=int))

    importance = (
        pd.DataFrame({"feature": enhancedFeatures, "importance": finalModel.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    importance.to_csv(outputImportance, index=False)

    print()
    print("========================================")
    print("TOP 20 ENHANCED FEATURES")
    print("========================================")
    print()

    print(importance.head(20).to_string(index=False))

    topImportance = importance.head(20).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 8))

    plt.barh(topImportance["feature"], topImportance["importance"])

    plt.xlabel("Random Forest importance")

    plt.title("Top V2 Enhanced Features")

    plt.tight_layout()

    plt.savefig(outputImportancePlot, dpi=160)

    plt.close()

    print()
    print("========================================")
    print("OUTPUT FILES")
    print("========================================")
    print()

    print(outputFeatures)

    print(outputPredictions)

    print(outputRecordingResults)

    print(outputImportance)

    print(outputImportancePlot)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
