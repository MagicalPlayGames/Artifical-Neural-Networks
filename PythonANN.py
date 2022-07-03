import random
from numpy import exp
from numpy import power
import ProgramParameters

#Neuron
class node:
    curSum = None;
    previousSum = None;
    output = None;
    currentWeights = [0.0];
    errorAdjustments = [0.0];

#Layer of Neurons
class layer:
    nodes = [node()];

    #Brain with layers
class brain:
    input = layer();
    layers = [layer()];
    output = None;
    realOutput = None;
    learningRate = None;

class Program:
    algorithmChoice = 0;
    iterations = 0;
    #Accuracy
    success = 0;
    total = 0;
    accuracy = 0;

    divisbleBy = 0;
    control = None;
    
    #This ANN calculates the probability a random number is divisble by divisbleBy

    #Constructor 1
    def __init__(self,layers, nodes, divisbleBy, iterations, learningRate, algoNum):
        self.algorithmChoice = algoNum;
        self.iterations = iterations;
        self.divisbleBy = divisbleBy;
        self.control.learningRate = learningRate;
        self.control = brain();
        self.control.layers = [layer()];
        for i in range(layers):
            self.control.layers[i].nodes = [node()];
            for j in range(nodes[i]):
                newestNode = node();
                newestNode.curSum = newestNode.output = 0;
                if (i == layers-1):
                    newestNode.currentWeights = [0.0];
                else:
                    newestNode.currentWeights = [0.0]*nodes[i+1];
                newestNode.errorAdjustments = [0.0]*len(newestNode.currentWeights);
                self.control.layers[i].nodes[j] = newestNode;

    #Constructor 2
    def __init__(self,input):
        self.algorithmChoice = input.alogirthmNumber;
        self.iterations = input.iterations;
        self.divisbleBy = input.divisbleBy;
        self.control = brain();
        self.control.learningRate = input.learningRate;
        self.control.layers = [layer()]*input.layers;
        for  i in range(input.layers):
            self.control.layers[i].nodes = [node()]*input.nodes[i];
            for j in range(input.nodes[i]):
                newestNode = node();
                newestNode.curSum = newestNode.output = 0;
                if (i == input.layers - 1):
                    newestNode.currentWeights = [0.0]
                else:
                    newestNode.currentWeights = [0.0]*input.nodes[i+1];
                newestNode.errorAdjustments = [0.0]*len(newestNode.currentWeights);
                self.control.layers[i].nodes[j] = newestNode;
    
    #Set all weights to random variables
    def resetWeights(self):
        for i in range(len(self.control.layers)):
            for j in range(len(self.control.layers[i].nodes)):
                for w in range(len(self.control.layers[i].nodes[j].currentWeights)):
                    self.control.layers[i].nodes[j].currentWeights[w] = random.random();
    
    #reads 10101 string input and puts it into the input layer, setting all input weights to 1
    def readInput(self,input):
        nums = list(input);
        self.control.input.nodes = [node()]*len(nums);
        i = 0;
        for num in nums:
            self.control.input.nodes[i] = node();
            self.control.input.nodes[i].output = (ord(num) - 48);
            self.control.input.nodes[i].currentWeights = [0.0]*len(self.control.layers[0].nodes);
            for j in range(len(self.control.layers[0].nodes)):
                self.control.input.nodes[i].currentWeights[j] = 1.0;
            i+=1;
    
    def readTrueOutput(self,output):
        self.control.realOutput = output;
    
    #Forward Prorogation
    def writeOutput(self):
            finalOutput = 0.0;
            inputFlag = True;
            layersLength = len(self.control.layers);
            for i in range(layersLength):
                curLayer = self.control.layers[i];
                previousLayer = layer();
                if (inputFlag):
                    previousLayer = self.control.input;
                    inputFlag = False;
                else:
                    previousLayer = self.control.layers[i - 1];
                for j in range(len(curLayer.nodes)):
                    curNode = curLayer.nodes[j];
                    for w in range(len(previousLayer.nodes)):
                        previousNode = previousLayer.nodes[w];
                        curNode.curSum += previousNode.output * previousNode.currentWeights[0];
                    self.control.layers[i].nodes[j].previousSum = curNode.curSum;
                    self.control.layers[i].nodes[j].output = sigmoid(curNode.curSum);
                    self.control.layers[i].nodes[j].curSum = 0;
            for i in range(len(self.control.layers[layersLength - 1].nodes)):
                finalOutput += self.control.layers[layersLength - 1].nodes[i].output * self.control.layers[layersLength - 1].nodes[i].currentWeights[0];
            self.control.output = tanh(finalOutput);

    #Updates the weights
    #Algorithm 0: findSingleError(),findMatrixErrors
    #Algorithm 1: findSingleError(), errorTotalRespectCurWeight()
    #Algorithm 2: topError(),hiddenErrors()
    def updateAllWeights(self):
            for i in range(len(self.control.layers) - 1, -1, -1):
                curLayer = self.control.layers[i];
                for j in range(len(curLayer.nodes)):
                    curNode = curLayer.nodes[j];
                    if (i == len(self.control.layers) - 1):
                        #Output layer to last hidden Layer
                        if(self.algorithmChoice==2):
                            errorAdjust = self.topError();
                        #layer to layer
                        else:
                            errorAdjust = findSingleError(curNode.currentWeights, self.control.realOutput - self.control.output);
                    else:
                        if (self.algorithmChoice==0):
                            errorAdjust = findMatrixErrors(curLayer.nodes, self.control.layers[i + 1].nodes);
                        elif(self.algorithmChoice==1):
                            errorAdjust = errorTotalRespectCurWeight(curNode, self.control.layers[i + 1].nodes);
                        else:
                            errorAdjust = hiddenErrors(curLayer, self.control.layers[i+1]);
        
                    self.control.layers[i].nodes[j].errorAdjustments = errorAdjust;
        
                    #Edit weights
                    for w in range(len(curNode.currentWeights)):
                        self.control.layers[i].nodes[j].currentWeights[w] -= errorAdjust[w] * self.control.learningRate;

    #Errors = -(outcome-trueOutcome) * (sigmoidPrime(non-activated output))DOT(previous non-activated output^T)
    def topError(self): 
            outputPrime = [[0.0]];
            outputPrime[0][0] = sigmoidPrime(self.control.output);
            outputsTransposed = [[0.0]];
            outputsTransposed[0] = [0.0]*len(self.control.layers[len(self.control.layers) - 1].nodes);
            for i in range(len(self.control.layers[len(self.control.layers)-1].nodes)):
                outputsTransposed[0][i] = self.control.layers[len(self.control.layers) - 1].nodes[i].previousSum;
            outputsTransposed = transpose(outputsTransposed);
        
            product = dotProduct(outputPrime, outputsTransposed);
            for i in range(len(product)):
                product[i] *= -(self.control.output - self.control.realOutput);
            return product;

    #Overall Processing
    def processData(self,inputs, expectedOutcome):
            self.readInput(inputs);
            self.writeOutput();
            self.readTrueOutput(expectedOutcome);
            return (abs(self.control.output - self.control.realOutput) < 0.5);
     
    #Operation for input and output
    def testOp(self):
            for j in range(self.iterations):
                #FIX ME
                choice = random.randint(0, 255);
                inputs = "{0:b}".format(choice);
                
                while (len(inputs)< 8):
                    inputs = "0" + inputs;
                if (choice % self.divisbleBy == 0):
                    output = 1;
                else:
                    output = 0;
                outcome = self.processData(inputs, output);
                if (outcome):
                    self.success+=1;
                else:
                    self.updateAllWeights();
                self.total+=1;
            self.accuracy = self.success/self.total;
            self.success = 0;
            self.total = 0;
            return self.accuracy;

#Errors = previousError*sigmoidPrime(non-activated output)DOT(currentWeights^T)DOT(upperInputs^T)
def hiddenErrors(curLayer, upperLayer):

        previousErrors = [[0.0]]*len(upperLayer.nodes);
        for i in range(len(previousErrors)):
            previousErrors[i] = [0.0]*len(upperLayer.nodes[i].errorAdjustments);
            for j in range(len(previousErrors[i])):
                previousErrors[i][j] = upperLayer.nodes[i].errorAdjustments[j];
        
        sigmoidPrimes = [[0.0]];
        sigmoidPrimes[0] = [0.0]*len(upperLayer.nodes);
        for i in range(len(sigmoidPrimes[0])):
            sigmoidPrimes[0][i] = sigmoidPrime(upperLayer.nodes[i].previousSum);
        weightsTransposed = [0.0]*len(upperLayer.nodes);
        for i in range(len(weightsTransposed)):
            weightsTransposed[i] = upperLayer.nodes[i].currentWeights;
        weightsTransposed = transpose(weightsTransposed);
    
        inputsTransposed = [[0.0]];
        inputsTransposed[0] = [0.0]*len(curLayer.nodes);
        for i in range(len(inputsTransposed[0])):
            inputsTransposed[0][i] = curLayer.nodes[i].output;
    
        inputsTransposed = transpose(inputsTransposed);
    
        products = [[0.0]];
        products[0] = dotProduct(sigmoidPrimes, weightsTransposed);
        products[0] = dotProduct(products, inputsTransposed);
        return dotProduct(products, previousErrors);
    
    
#The Total Error in respect to the Current Output    
def errorTotalRespectCurOutput(outcome, trueOutcome):
    return -(trueOutcome - outcome);

#The Current Output in respect to the Net Output
def curOutputRespectNetOutput(outcome):
    return outcome * (1 - outcome);

#The Net Output in respect to the current weights
def netOutputRespectCurWeights(weight, outcome):
        return weight * outcome;

#The Total Error in respect to the current weights
#The Net Output in respect to the current weights * The Total Error in respect to the Current Output * The Current Output in respect to the Net Output      
def errorTotalRespectCurWeight(curNode, upperLayer):
        errors = [0.0]*len(curNode.currentWeights);
        for i in range(len(errors)):
            errorTotal = 0;
            for j in range(len(upperLayer[i].errorAdjustments)):
                errorTotal += upperLayer[i].errorAdjustments[j];
            curOutput = upperLayer[i].previousSum;
    
            netOutput = upperLayer[i].output;
            curWeight = curNode.currentWeights[i];
            errors[i] = errorTotalRespectCurOutput(curOutput, errorTotal) * curOutputRespectNetOutput(netOutput) * netOutputRespectCurWeights(curWeight, netOutput);
        return errors;


#(e^(x) - e^(-x)) / (e^(x) + e^(-x))
def tanh(x):
    EX = exp(x);
    negativeEX = exp(-x);
    return (EX - negativeEX) / (EX + negativeEX);

#Mean Square Error
def errorMSE(outcome, trueOutcome):
    return power(trueOutcome - outcome, 2);

#Error1 = w1/totalWeights * error
def findSingleError(weights, error):
    totalWeights = 0;
    for i in range(len(weights)):
        totalWeights += weights[i];
    for i in range(len(weights)):
        weights[i] = (weights[i] / totalWeights) * error;
    return weights;

#Errors = weights * previousErrors
def findMatrixErrors(curLayer, upperLayer):
    weights = [[0.0]]*len(curLayer);
    for i in range(len(weights)):
        weights[i] = curLayer[i].currentWeights;

    errors = [[0.0]]*len(upperLayer[0].currentWeights);
    for i in range(len(errors)):
        errors[i] = [0.0]*len(upperLayer);
        for j in range(len(errors[i])):
            errors[i][j] = upperLayer[j].errorAdjustments[i];
    errors = transpose(errors);
    return dotProduct(errors, weights);

#Dot product of two matricies
def _dotProduct(errors, weights):
    if (len(errors) > len(weights)):
        size = len(errors);
    else:
        size = len(weights);

    outcomes = [0.0]*size;

    for i in range(size):
        outcomes[i] = 0;
        if (size == len(errors)):
            for j in range(len(errors[i])):
                if (i >= len(weights) or j>=len(errors[i])):
                    break;
                for w in range(len(weights[i])):
                    outcomes[i] += weights[i][w] * errors[i][j];
        else:
            for j in range(len(weights[i])):
                if (i >= len(errors) or j >= len(weights)):
                    break;
                for w in range(len(errors[i])):
                    outcomes[i] += weights[j][i] * errors[i][w];
    return outcomes;

#flip matrix on its side
def transpose(weights):
    newWeights = [[0.0]]*len(weights[0]);
    for i in range(len(weights[0])):
        newWeights[i] = [0.0]*len(weights);
        for j in range(len(weights)):
            newWeights[i][j] = weights[j][i];
    return newWeights;

#e^x/(1+e^x)
def sigmoid(x): 
    EX = exp(x);
    return EX / (1 + EX);
    
#(e^-x)/((1+e^-x)^2)
def sigmoidPrime(x):
    negativeEX = exp(-x);
    return negativeEX / (power(1 + negativeEX, 2));
    