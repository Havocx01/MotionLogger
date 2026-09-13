# Detected steps are estimates until checked against known step counts.

from pathlib import Path
from io import StringIO
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

scriptDir = Path(__file__).resolve().parent
dataDir = scriptDir / "Data"

eventOutput = scriptDir / "Wearable_StepEvents_V1.csv"

countOutput = scriptDir / "Wearable_StepCounts_V1.csv"

plotDir = scriptDir / "StepEventPlots_V1"

ignoreFirstSeconds = 1.0

minStepFrequencyHz = 1.0
maxStepFrequencyHz = 3.2

minimumPeakGapSeconds = 0.25

peakThresholdZ = 0.70

scoreSmoothingSamples = 3

# Add known counts by filename, for example: {"22_known_40_steps": 40}.
knownStepCounts = {}

numericColumns = ["time", "accelX", "accelY", "accelZ", "accelMagnitude", "gyroX", "gyroY", "gyroZ", "gyroMagnitude"]


def findFolder(parent, possibleNames):
    folders = {item.name.lower(): item for item in parent.iterdir() if item.is_dir()}

    for name in possibleNames:
        match = folders.get(name.lower())

        if match is not None:
            return match

    raise FileNotFoundError(f"Could not find one of these folders inside {parent}: " f"{possibleNames}")


def stripSerialPrefix(line):
    return re.sub(r"^\s*\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\s*->\s*", "", line).strip()


def robustZ(values):
    values = np.asarray(values, dtype=float)

    median = np.median(values)

    mad = np.median(np.abs(values - median))

    # Scale MAD to match standard deviation for normally distributed values.
    scale = 1.4826 * mad

    if scale < 1e-9:
        scale = np.std(values)

    if scale < 1e-9:
        return np.zeros_like(values)

    return (values - median) / scale


def movingAverage(values, windowSize):
    values = np.asarray(values, dtype=float)

    if windowSize <= 1:
        return values.copy()

    series = pd.Series(values)

    return series.rolling(window=windowSize, center=True, min_periods=1).mean().to_numpy(dtype=float)


def loadRecording(filePath):
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

    data = pd.read_csv(StringIO("\n".join(csvLines)))

    data.columns = data.columns.str.strip()

    for column in numericColumns:
        if column not in data.columns:
            raise ValueError(f"{filePath.name} missing column: {column}")

        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=numericColumns).sort_values("time").reset_index(drop=True)

    data["timeSeconds"] = data["time"] / 1000.0

    data = data[data["timeSeconds"] >= ignoreFirstSeconds].copy()
    data = data.reset_index(drop=True)

    return data


def buildStepSignals(data):
    times = data["timeSeconds"].to_numpy(dtype=float)

    accel = data[["accelX", "accelY", "accelZ"]].to_numpy(dtype=float)
    accelMagnitude = data["accelMagnitude"].to_numpy(dtype=float)

    dtValues = np.diff(times)

    validDt = dtValues[dtValues > 0]

    if len(validDt) == 0:
        raise ValueError("Recording contains no valid time differences.")

    averageDt = float(np.mean(validDt))

    samplesPerSecond = 1.0 / averageDt

    gravityWindow = max(3, int(round(samplesPerSecond * 0.8)))

    # Track the local gravity baseline as the wrist changes orientation.
    localBaseline = pd.Series(accelMagnitude).rolling(window=gravityWindow, center=True, min_periods=1).mean().to_numpy(dtype=float)

    dynamicAccel = np.abs(accelMagnitude - localBaseline)

    vectorJerk = np.zeros(len(data), dtype=float)

    accelDifference = np.diff(accel, axis=0)

    dt = np.diff(times)

    valid = dt > 0

    jerkValues = np.zeros(len(dt), dtype=float)

    jerkValues[valid] = np.linalg.norm(accelDifference[valid], axis=1) / dt[valid]

    vectorJerk[1:] = jerkValues

    directionChange = np.zeros(len(data), dtype=float)

    first = accel[:-1]
    second = accel[1:]

    firstNorm = np.linalg.norm(first, axis=1)

    secondNorm = np.linalg.norm(second, axis=1)

    denominator = firstNorm * secondNorm

    directionValues = np.zeros(len(first), dtype=float)

    validDirection = denominator > 1e-9

    cosine = np.zeros(len(first), dtype=float)

    cosine[validDirection] = np.sum(first[validDirection] * second[validDirection], axis=1) / denominator[validDirection]

    cosine = np.clip(cosine, -1.0, 1.0)

    directionValues[validDirection] = np.degrees(np.arccos(cosine[validDirection]))

    directionChange[1:] = directionValues

    jerkZ = robustZ(vectorJerk)

    dynamicZ = robustZ(dynamicAccel)

    directionZ = robustZ(directionChange)

    jerkEvidence = np.maximum(jerkZ, 0.0)

    dynamicEvidence = np.maximum(dynamicZ, 0.0)

    directionEvidence = np.maximum(directionZ, 0.0)

    score = 0.50 * jerkEvidence + 0.30 * directionEvidence + 0.20 * dynamicEvidence
    score = movingAverage(score, scoreSmoothingSamples)

    signals = pd.DataFrame(
        {
            "timeSeconds": times,
            "accelMagnitude": accelMagnitude,
            "dynamicAccel": dynamicAccel,
            "vectorJerk": vectorJerk,
            "directionChange": directionChange,
            "stepScore": score,
        }
    )

    return (signals, averageDt)


def estimateCadence(score, averageDt):
    score = np.asarray(score, dtype=float)

    if len(score) < 10:
        return np.nan

    centered = score - np.mean(score)

    frequencies = np.fft.rfftfreq(len(centered), d=averageDt)

    power = np.abs(np.fft.rfft(centered)) ** 2

    walkingMask = (frequencies >= minStepFrequencyHz) & (frequencies <= maxStepFrequencyHz)

    walkingFrequencies = frequencies[walkingMask]

    walkingPower = power[walkingMask]

    if len(walkingPower) == 0:
        return np.nan

    dominantIndex = int(np.argmax(walkingPower))

    return float(walkingFrequencies[dominantIndex])


def detectStepPeaks(signals, cadenceHz):
    times = signals["timeSeconds"].to_numpy(dtype=float)

    score = signals["stepScore"].to_numpy(dtype=float)

    scoreMedian = float(np.median(score))

    scoreMad = float(np.median(np.abs(score - scoreMedian)))

    scoreScale = 1.4826 * scoreMad

    if scoreScale < 1e-9:
        scoreScale = float(np.std(score))

    threshold = scoreMedian + peakThresholdZ * scoreScale

    if np.isfinite(cadenceHz) and cadenceHz > 0:
        expectedInterval = 1.0 / cadenceHz

        minimumGap = max(minimumPeakGapSeconds, 0.52 * expectedInterval)

    else:
        expectedInterval = np.nan
        minimumGap = minimumPeakGapSeconds

    candidates = []

    for index in range(1, len(score) - 1):
        if score[index] > score[index - 1] and score[index] >= score[index + 1] and score[index] >= threshold:
            candidates.append(index)

    # Keep the stronger peak when two candidates are too close together.
    selected = []

    for index in candidates:
        if not selected:
            selected.append(index)
            continue

        previousIndex = selected[-1]

        gap = times[index] - times[previousIndex]

        if gap >= minimumGap:
            selected.append(index)

        elif score[index] > score[previousIndex]:
            selected[-1] = index

    return (selected, threshold, expectedInterval, minimumGap)


def savePlot(recordingName, signals, peakIndices, threshold):
    plotDir.mkdir(parents=True, exist_ok=True)

    times = signals["timeSeconds"].to_numpy()

    score = signals["stepScore"].to_numpy()

    plt.figure(figsize=(12, 5))

    plt.plot(times, score, label="Step impact score")

    plt.axhline(threshold, linestyle="--", label="Adaptive threshold")

    if peakIndices:
        plt.scatter(times[peakIndices], score[peakIndices], label="Detected step")

    plt.xlabel("Time (seconds)")

    plt.ylabel("Step impact score")

    plt.title(f"{recordingName} - detected steps: {len(peakIndices)}")

    plt.legend()

    plt.tight_layout()

    plt.savefig(plotDir / f"{recordingName}.png", dpi=150)

    plt.close()


def main():
    if not dataDir.exists():
        raise FileNotFoundError(f"Data folder not found: {dataDir}")

    walkingFolder = findFolder(dataDir, ["Walking", "Waiking"])

    walkingFiles = sorted(walkingFolder.glob("*.txt"))

    if not walkingFiles:
        raise FileNotFoundError(f"No walking .txt files found in {walkingFolder}")

    print()
    print("========================================")
    print("   WEARABLE STEP EVENT DETECTOR V1")
    print("========================================")
    print()

    print(f"Walking recordings: {len(walkingFiles)}")

    print()

    allEvents = []
    countRows = []

    for filePath in walkingFiles:
        recordingName = filePath.stem

        print(f"Analyzing: {recordingName}")

        data = loadRecording(filePath)

        signals, averageDt = buildStepSignals(data)

        cadenceHz = estimateCadence(signals["stepScore"].to_numpy(), averageDt)

        peakIndices, threshold, expectedInterval, minimumGap = detectStepPeaks(signals, cadenceHz)

        peakTimes = signals["timeSeconds"].to_numpy()[peakIndices]

        peakScores = signals["stepScore"].to_numpy()[peakIndices]

        intervals = np.diff(peakTimes)

        meanInterval = float(np.mean(intervals)) if len(intervals) > 0 else np.nan

        intervalStd = float(np.std(intervals, ddof=1)) if len(intervals) > 1 else np.nan

        intervalCV = intervalStd / meanInterval if (np.isfinite(meanInterval) and meanInterval > 0 and np.isfinite(intervalStd)) else np.nan

        detectedCount = len(peakIndices)

        knownCount = knownStepCounts.get(recordingName)

        if knownCount is not None:
            countError = detectedCount - knownCount

            absoluteError = abs(countError)

            percentError = absoluteError / knownCount * 100.0 if knownCount > 0 else np.nan

        else:
            countError = np.nan
            absoluteError = np.nan
            percentError = np.nan

        countRows.append(
            {
                "recording": recordingName,
                "detectedSteps": detectedCount,
                "estimatedCadenceHz": cadenceHz,
                "estimatedStepsPerMinute": (cadenceHz * 60.0 if np.isfinite(cadenceHz) else np.nan),
                "expectedStepIntervalSeconds": expectedInterval,
                "minimumPeakGapSeconds": minimumGap,
                "meanDetectedIntervalSeconds": meanInterval,
                "detectedIntervalStd": intervalStd,
                "detectedIntervalCV": intervalCV,
                "knownSteps": knownCount,
                "countError": countError,
                "absoluteError": absoluteError,
                "percentError": percentError,
            }
        )

        for stepNumber, peakIndex in enumerate(peakIndices, start=1):
            allEvents.append(
                {
                    "recording": recordingName,
                    "stepNumber": stepNumber,
                    "timeSeconds": signals["timeSeconds"].iloc[peakIndex],
                    "stepScore": signals["stepScore"].iloc[peakIndex],
                    "vectorJerk": signals["vectorJerk"].iloc[peakIndex],
                    "dynamicAccel": signals["dynamicAccel"].iloc[peakIndex],
                    "directionChange": signals["directionChange"].iloc[peakIndex],
                }
            )

        savePlot(recordingName, signals, peakIndices, threshold)

        print(f"  Detected steps: {detectedCount}")

        if np.isfinite(cadenceHz):
            print(f"  Estimated cadence: " f"{cadenceHz:.2f} Hz " f"({cadenceHz * 60.0:.1f} steps/min)")

        if np.isfinite(meanInterval):
            print(f"  Mean detected gap: " f"{meanInterval:.3f} s")

        if knownCount is not None:
            print(f"  Known steps: {knownCount}")

            print(f"  Error: {countError:+d}")

        print()

    eventFrame = pd.DataFrame(allEvents)

    countFrame = pd.DataFrame(countRows)

    eventFrame.to_csv(eventOutput, index=False)

    countFrame.to_csv(countOutput, index=False)

    print("========================================")
    print("STEP COUNT SUMMARY")
    print("========================================")
    print()

    displayColumns = ["recording", "detectedSteps", "estimatedCadenceHz", "estimatedStepsPerMinute", "meanDetectedIntervalSeconds", "detectedIntervalCV"]

    print(countFrame[displayColumns].round(3).to_string(index=False))

    print()
    print("========================================")
    print("OUTPUT FILES")
    print("========================================")
    print()

    print(eventOutput)

    print(countOutput)

    print(plotDir)

    print()
    print("NEXT VALIDATION:")

    print("Record one controlled walking file while counting " "the exact number of physical steps you take.")

    print("40-60 steps is enough for the first calibration test.")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
