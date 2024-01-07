"""
Rutvi Bheda,rb1859@rit.edu
AI Lab 3
"""


import string
from typing import List
import math
import pickle
import sys


class Node:
    def __init__(self, **kwargs):
        self.value = kwargs.get("value")
        self.left: Node = kwargs.get("left")
        self.right: Node = kwargs.get("right")
        self.attribute = kwargs.get("attribute")
        self.size = kwargs.get('size')


class Stump:
    __slots__ = 'left', 'right', 'attribute'

    def __init__(self) -> None:
        self.left = None
        self.right = None
        self.attribute = None


"""
below stated are the five distinct attributes that are used for the decision tree and the adaBoost
"""


def frequentlyUsedDutchWords(sentence):
    usedWords = ['van', 'ik', 'je', 'zo', 'niet', 'de', 'dat', 'en', 'het', 'dit', 'jullie', 'heeft', 'wie',
                 'deze', 'ze', 'uit', 'ook', 'dan', 'hem', 'op', 'te', 'weten', 'voor', 'hun', 'allemaal', 'wanneer',
                 'wij', 'hij', 'zij', 'maar', 'hen', 'ons', 'dus', 'naar', 'hebben', 'wat', 'meest', 'geen']
    getClearSentence = sentence.translate(str.maketrans("", "", string.punctuation)).lower()
    getWords = getClearSentence.split()
    for word in getWords:
        if word in usedWords:
            return True
    return False


def frequentlyUsedEnglishWords(sentence):
    usedWords = ['and', 'this', 'not', 'of', 'he', 'she', 'it', 'they', 'them', 'these',
                 'but', 'and', 'then', 'than', 'to', 'be', 'your', 'so', 'also', 'us', 'I',
                 'for', 'not', 'on', 'for', 'up', 'was', 'to']
    getClearSentence = sentence.translate(str.maketrans("", "", string.punctuation)).lower()
    getWords = getClearSentence.split()
    for word in getWords:
        if word in usedWords:
            return True
    return False


def endWordsinDutch(sentence):
    endWords = ['en', 'el', 'heid']
    getClearSentence = sentence.translate(str.maketrans("", "", string.punctuation)).lower()
    getWords = getClearSentence.split()
    for word in getWords:
        for suffix in endWords:
            if word.endswith(suffix):
                return True
    return False


def articlesInEnglish(sentence):
    getClearSentence = sentence.translate(str.maketrans("", "", string.punctuation)).lower()
    getWords = getClearSentence.split()
    for word in getWords:
        if word == 'an' or word == 'the' or word == 'a':
            return True
    return False


def articlesInDutch(sentence):
    getClearSentence = sentence.translate(str.maketrans("", "", string.punctuation)).lower()
    getWords = getClearSentence.split()
    for word in getWords:
        if word == 'de' or word == 'een' or word == 'het':
            return True
    return False

def processFile(input_file):
    count = 0
    with open(input_file, 'r', encoding='utf-8') as file:
        fileContent = file.readlines()
        sentencesList = []
        label = {}
        for line in fileContent:
            words = line.split('|')
            inputLabels = words[0].strip()
            label[count] = inputLabels  # --if tuple is needed
            count = count + 1
            fifteenLengthSentence = ' '.join(words[1].split()[:15])
            sentencesList.append(fifteenLengthSentence)
        return sentencesList, label


def generateFeaturesTest(sentencesList):
    """
    function to convert sentences to features
    :param sentencesList: list of sentences
    :return: features
    """
    attribute1 = []
    attribute2 = []
    attribute3 = []
    attribute4 = []
    attribute5 = []
    for sentence in sentencesList:
        attribute1.append(frequentlyUsedEnglishWords(sentence))
        attribute2.append(frequentlyUsedDutchWords(sentence))
        attribute3.append(endWordsinDutch(sentence))
        attribute4.append(articlesInDutch(sentence))
        attribute5.append(articlesInEnglish(sentence))
    attributes = [attribute1, attribute2, attribute3, attribute4, attribute5]
    return attributes


def predict_stump(stump, datum):
    if datum[stump.attribute] == False:
        return stump.right
    return stump.left


def create_stump(feature_data, labels, weights, attributes):
    bestAttribute, bestIG, stump = findBestAttribute(iterate(feature_data, labels, {}, weights), attributes)
    return bestAttribute, bestIG, stump


def adaboost(feature_data, labels, stump_count) -> List[Stump]:
    stumps = []
    n = len(feature_data)
    weights = [1 / n] * n
    attributes = [i for i in range(5)]
    for i in range(stump_count):
        attribute, ig, stump = create_stump(feature_data, labels, weights, attributes)
        attributes.remove(attribute)
        error = 0.1
        correct = 0
        incorrect = 0
        for i in range(len(feature_data)):
            datum = feature_data[i]
            if predict_stump(stump, datum) == labels[i]:
                correct += 1
            else:
                error += weights[i]
                incorrect += 1
        for i in range(len(feature_data)):
            datum = feature_data[i]
            if predict_stump(stump, datum) == labels[i]:
                weights[i] = weights[i] * (error / (1 - error))
        weightSum = sum(weights)
        for i in range(len(weights)):
            weights[i] /= weightSum
        amountOfSay = (1 - error) / error
        stumps.append([stump, amountOfSay])
    return stumps


def evaluateEntropy(value):
    """
    function to calculate the entropy value
    :param value: value
    :return: entropy value
    """
    if value == 1:
        return 0
    p = value
    q = 1 - value
    if 0 < p < 1 and 0 < q < 1:
        entropy = - (p * math.log(p, 2) + q * math.log(q, 2))
        return entropy
    else:
        return 0


def checkTestFile(input_file):
    """
    function to get the test file in proper format
    :param input_file: input file
    :return: sentence list
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        fileContent = file.readlines()
        sentencesList = []
        for line in fileContent:
            words = line.split()
            fifteenLengthSentence = ' '.join(words[:15])
            sentencesList.append(fifteenLengthSentence)
        return sentencesList


def predictWithAllStumps(stumps, datum):
    nl = en = 0
    for stump, amountOfSay in stumps:
        pred = predict_stump(stump, datum)
        if pred == 'en':
            en += amountOfSay
        else:
            nl += amountOfSay
    return "en" if en > nl else "nl"


def adaPredict(hypothesis, test_file):
    """
    predicting the ada model
    :param hypothesis: hypothesis file
    :param test_file: file to test
    :return:
    """
    model = pickle.load(open(hypothesis, 'rb'))
    sentencesList = checkTestFile(test_file)
    feature_data = generateFeaturesTest(sentencesList)
    feature_data = transpose(feature_data)
    predictions = []
    for row in feature_data:
        prediction = predictWithAllStumps(model, row)
        predictions.append(prediction)
    print(*predictions, sep='\n')


def decisionTreePrediction(hypothesis, test_file):
    """
    function to predict the decision tree
    :param hypothesis: hypothesis file
    :param test_file: test file
    """
    model = pickle.load(open(hypothesis, 'rb'))

    sentencesList = checkTestFile(test_file)
    feature_data = generateFeaturesTest(sentencesList)
    if (len(feature_data) == 0):
        raise Exception("No data for training")
    feature_data = transpose(feature_data)
    for datum in feature_data:
        prediction = predict(model, datum)
        print(prediction)


def predict(parent: Node, datum):
    if parent.value:
        return parent.value
    feature_val = datum[parent.attribute]
    if feature_val == False:
        return predict(parent.right, datum)
    return predict(parent.left, datum)


def decisionTreeTraining(input_file, hypothesis):
    """
    function for training and forming a decision tree
    :param input_file: input file
    :param hypothesis: model
    """
    sentList, label = processFile(input_file)

    feature_data = generateFeaturesTest(sentList)
    if (len(feature_data) == 0):
        raise Exception("No data for training")

    attributeValues = {}
    feature_data = transpose(feature_data)
    parentNode = createTree(feature_data, attributeValues, label)
    createModel = open(hypothesis, 'wb')
    pickle.dump(parentNode, createModel)


def transpose(arr):
    result = [0] * len(arr[0])
    for i in range(len(result)):
        result[i] = [0] * len(arr)
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            result[j][i] = arr[i][j]
    return result


def verifyAttributeValues(datum, attributeValues):
    for attribute in attributeValues:
        if datum[attribute] != attributeValues[attribute]:
            return False
    return True


def getInformationGainForAttribute(dataSubset, attribute):
    count = len(dataSubset)
    stump = Stump()
    stump.attribute = attribute
    stump.left = "en"
    stump.right = "nl"
    if (count == 0):
        return 0 , stump
    dutchFalseCount = dutchTrueCount = englishTrueCount = englishFalseCount = 0
    for label, datum, weight in dataSubset:
        english = (label == 'en')
        attributeValueForDatum = datum[attribute]
        if attributeValueForDatum:
            if english:
                englishTrueCount += weight
            else:
                dutchTrueCount += weight
        else:
            if english:
                englishFalseCount += weight
            else:
                dutchFalseCount += weight

    totalTrue = englishTrueCount + dutchTrueCount
    totalFalse = englishFalseCount + dutchFalseCount
    count = totalTrue + totalFalse
    englishTrueFraction = englishTrueCount / (totalTrue or 1)
    englishFalseFraction = englishFalseCount / (totalFalse or 1)
    remainingTrue = (totalTrue / count) * evaluateEntropy(englishTrueFraction)
    remainingFalse = (totalFalse / count) * evaluateEntropy(englishFalseFraction)
    addedValue = remainingTrue + remainingFalse
    englishCount = englishTrueCount + englishFalseCount
    englishFraction = englishCount / count
    overallEntropy = evaluateEntropy(englishFraction)
    informationGain = overallEntropy - addedValue
    stump = Stump()
    stump.attribute = attribute
    if englishTrueCount < dutchTrueCount:
        stump.left = 'nl'
    else:
        stump.left = 'en'
    if englishFalseCount < dutchFalseCount:
        stump.right = 'nl'
    else:
        stump.right = 'en'
    return informationGain, stump


def iterate(feature_data, labels, attributeValues, weights=None):
    if weights == None:
        weights = [1] * len(feature_data)
    for i in range(len(feature_data)):
        if not verifyAttributeValues(feature_data[i], attributeValues): continue
        yield labels[i], feature_data[i], weights[i]


def findBestAttribute(dataSubsetGenerator, remainingAttributes):
    bestIG = -1
    bestAttribute = None
    bestStump = None
    dataSubset = [val for val in dataSubsetGenerator]

    for attribute in remainingAttributes:
        informationGain, stump = getInformationGainForAttribute(dataSubset, attribute)
        if informationGain > bestIG:
            bestIG = informationGain
            bestAttribute = attribute
            bestStump = stump
    return bestAttribute, bestIG, bestStump


def createTree(feature_data, attributeValues, labels: dict) -> Node:
    if len(attributeValues) == len(feature_data[0]):
        diffCount = 0
        count = 0
        for label, _, _ in iterate(feature_data, labels, attributeValues):
            if label == 'en':
                diffCount += 1
            else:
                diffCount -= 1
            count += 1
        if diffCount >= 0:
            return Node(value='en', size=1)
        else:
            return Node(value='nl', size=1)
    remainingAttributes = [attr for attr in range(len(feature_data[0])) if attr not in attributeValues]
    bestAttribute, bestIG, _ = findBestAttribute(iterate(feature_data, labels, attributeValues), remainingAttributes)
    attributeValues[bestAttribute] = True
    left = createTree(feature_data, attributeValues, labels)
    attributeValues[bestAttribute] = False
    right = createTree(feature_data, attributeValues, labels)
    del attributeValues[bestAttribute]
    return Node(left=left, right=right, attribute=bestAttribute, size=left.size + right.size)


def main():
    if sys.argv[1] == 'train':
        input_file = sys.argv[2]
        hypothesis = sys.argv[3]
        if sys.argv[4] == 'dt':
            decisionTreeTraining(input_file, hypothesis)
        elif sys.argv[4] == 'ada':
            sentList, label = processFile(input_file)
            feature_data = generateFeaturesTest(sentList)
            feature_data = transpose(feature_data)
            ada = adaboost(feature_data, label, 5)
            pickle.dump(ada, open(hypothesis, 'wb'))
    elif sys.argv[1] == 'predict':
        model = pickle.load(open(sys.argv[2], 'rb'))
        if type(model) == Node:
            decisionTreePrediction(sys.argv[2], sys.argv[3])
        else:
            adaPredict(sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()