import java.util.Random;
import java.lang.Math.*;
public class Program
{
    //Runs per operate action
    private byte algorithmChoice;
    private int iterations;
    //Accuracy
    private int success;
    private int total;
    private float accuracy;

    private byte divisbleBy;

    //Neuron
    public class node
    {
        //Calc for output
        public float curSum;
        //Store for later
        public float previousSum;
        //Output
        public float output;
        //current Weights from previous layer to this node
        public float[] currentWeights;
        //adjustment storage for weight backward propergation
        public float[] errorAdjustments;
    };
    //Layer of Neurons
    public class layer
    {
        public node[] nodes;
    }

    //Brain with layers
    private class brain
    {
        //Input
        public layer input;
        //Hidden Layers
        public layer[] layers;
        //Output (change to multiple outputs)
        public float output;
        //Expected output
        public float realOutput;
        public float learningRate;
    }

    private brain control;


    //This ANN calculates the probability a random number is divisble by divisbleBy

    //Constructor 1
    public Program(byte layers, int[] nodes, byte divisbleBy,int iterations,float learningRate, byte algoNum)
    {
        this.algorithmChoice = algoNum;
        this.iterations = iterations;
        this.divisbleBy = divisbleBy;
        control.learningRate = learningRate;
        control = new brain();
        control.layers = new layer[layers];
        for (int i=0;i<layers;i++)
        {
            control.layers[i].nodes = new node[nodes[i]];
            for (int j = 0;j<nodes[i];j++)
            {
                node newestNode = new node();
                newestNode.curSum = newestNode.output = 0;
                if (i == layers-1)
                    newestNode.currentWeights = new float[1];
                else
                    newestNode.currentWeights = new float[nodes[i + 1]];
                newestNode.errorAdjustments = new float[newestNode.currentWeights.length];
                control.layers[i].nodes[j] = newestNode;
            }
        }
    }

    //Constructor 2
    public Program(ProgramParameters input)
    {
        this.algorithmChoice = input.alogirthmNumber;
        this.iterations = input.iterations;
        this.divisbleBy = input.divisbleBy;
        control = new brain();
        control.learningRate = input.learningRate;
        control.layers = new layer[input.layers];
        for (int i = 0; i < input.layers; i++)
        {
            control.layers[i] = new layer();
            control.layers[i].nodes = new node[input.nodes[i]];
            for (int j = 0; j < input.nodes[i]; j++)
            {
                node newestNode = new node();
                newestNode.curSum = newestNode.output = 0;
                if (i == input.layers - 1)
                    newestNode.currentWeights = new float[1];
                else
                    newestNode.currentWeights = new float[input.nodes[i+1]];
                newestNode.errorAdjustments = new float[newestNode.currentWeights.length];
                control.layers[i].nodes[j] = newestNode;
            }
        }
    }

    //Operation for input and output
    public float testOp()
    {
        for (int j = 0; j < iterations; j++)
        {
            Random rnd = new Random();
            int choice = rnd.nextInt(0, 256);
            String inputs = Integer.toBinaryString(choice);
            while (inputs.length() < 8)
                inputs = "0" + inputs;
            int output;
            if (choice % divisbleBy == 0)
                output = 1;
            else
                output = 0;
            boolean outcome = processData(inputs, output);
            if (outcome)
                success++;
            else
                updateAllWeights();
            total++;
        }
        accuracy = ((float)success) / ((float)total);
        success = 0;
        total = 0;
        return accuracy;
    }

    //Overall Processing
    private boolean processData(String inputs, int expectedOutcome)
    {
        readInput(inputs);
        writeOutput();
        readTrueOutput(expectedOutcome);
        return (Math.abs(control.output - control.realOutput) < 0.5);
    }

    //Set all weights to random variables
    public void resetWeights()
    {
        Random rnd = new Random();
        for (int i = 0; i < control.layers.length; i++)
        {
            for (int j = 0; j < control.layers[i].nodes.length; j++)
            {
                for (int w = 0; w < control.layers[i].nodes[j].currentWeights.length; w++)
                {
                    control.layers[i].nodes[j].currentWeights[w] = rnd.nextFloat();
                }
            }
        }
    }


    //reads 10101 string input and puts it into the input layer, setting all input weights to 1
    private void readInput(String input)
    {
        char[] nums = input.toCharArray();
        control.input = new layer();
        control.input.nodes = new node[nums.length];
        int i = 0;
        for (char num : nums)
        {
            control.input.nodes[i] = new node();
            control.input.nodes[i].output = ((int)num) - 48;
            control.input.nodes[i].currentWeights = new float[control.layers[0].nodes.length];
            for (int j = 0; j < control.layers[0].nodes.length; j++)
                control.input.nodes[i].currentWeights[j] = 1.0f;
            i++;
        }
    }

    private void readTrueOutput(int output)
    {
        control.realOutput = output;
    }

    //Forward Proprogation
    private void writeOutput()
    {
        float finalOutput = 0;
        boolean inputFlag = true;
        int layerslength = control.layers.length;
        for (int i = 0; i < layerslength; i++)
        {
            layer curLayer = control.layers[i];
            layer previousLayer;
            if (inputFlag)
            {
                previousLayer = control.input;
                inputFlag = false;
            }
            else
            {
                previousLayer = control.layers[i - 1];
            }
            for (int j = 0; j < curLayer.nodes.length; j++)
            {
                node curNode = curLayer.nodes[j];
                for (int w = 0; w < previousLayer.nodes.length; w++)
                {
                    node previousNode = previousLayer.nodes[w];
                    curNode.curSum += previousNode.output * previousNode.currentWeights[0];
                }
                control.layers[i].nodes[j].previousSum = curNode.curSum;
                control.layers[i].nodes[j].output = sigmoid(curNode.curSum);
                control.layers[i].nodes[j].curSum = 0;
            }
        }
        for (int i = 0; i < control.layers[layerslength - 1].nodes.length; i++)
        {
            finalOutput += control.layers[layerslength - 1].nodes[i].output * control.layers[layerslength - 1].nodes[i].currentWeights[0];
        }
        control.output = tanh(finalOutput);
    }

    //Updates the weights
    //Algorithm 0: findSingleError(),findMatrixErrors
    //Algorithm 1: findSingleError(), errorTotalRespectCurWeight()
    //Algorithm 2: topError(),hiddenErrors()
    private void updateAllWeights()
    {
        for (int i = control.layers.length - 1; i > -1; i--)
        {
            layer curLayer = control.layers[i];
            for (int j = 0; j < curLayer.nodes.length; j++)
            {
                node curNode = curLayer.nodes[j];
                float[] errorAdjust;
                if (i == control.layers.length - 1)
                //output layer to last hidden layer
                {
                    if(algorithmChoice==2)
                        errorAdjust = topError();
                    else
                        errorAdjust = findSingleError(curNode.currentWeights, control.realOutput - control.output);
                }
                else
                {
                    //layer to layer
                    if (algorithmChoice==0)
                        errorAdjust = findMatrixErrors(curLayer.nodes, control.layers[i + 1].nodes);
                    else if(algorithmChoice==1)
                        errorAdjust = errorTotalRespectCurWeight(curNode, control.layers[i + 1].nodes);
                    else
                        errorAdjust = hiddenErrors(curLayer, control.layers[i+1]);

                }

                control.layers[i].nodes[j].errorAdjustments = errorAdjust;

                for (int w = 0; w < curNode.currentWeights.length; w++)
                {
                    //edit weights
                    control.layers[i].nodes[j].currentWeights[w] -= errorAdjust[w] * control.learningRate;
                }
            }
        }
    }

    //e^x/(1+e^x)
    private float sigmoid(float x)
    {
        float EX = (float)Math.exp(x);
        return EX / (1 + EX);
    }

    //(e^-x)/((1+e^-x)^2)
    private float sigmoidPrime(float x)
    {
        float negativeEX = (float)Math.exp(-x);
        return negativeEX / ((float)Math.pow(1 + negativeEX, 2));
    }

    //Errors = -(outcome-trueOutcome) * (sigmoidPrime(non-activated output))DOT(previous non-activated output^T)
    private float[] topError()
    {
        //https://www.bogotobogo.com/python/scikit-learn/Artificial-Neural-Network-ANN-4-Backpropagation.php
        //top error
        float[][] outputPrime = new float[1][];
        outputPrime[0] = new float[1];
        outputPrime[0][0] = sigmoidPrime(control.output);
        float[][] outputsTransposed = new float[1][];
        outputsTransposed[0] = new float[control.layers[control.layers.length - 1].nodes.length];
        for (int i =0;i<control.layers[control.layers.length-1].nodes.length;i++)
        {
            outputsTransposed[0][i] = control.layers[control.layers.length - 1].nodes[i].previousSum;
        }
        outputsTransposed = transpose(outputsTransposed);

        float[] product = dotProduct(outputPrime, outputsTransposed);
        for(int i =0;i<product.length;i++)
        {
            product[i] *= -(control.output - control.realOutput);
        }
        return product;
    }


    //Errors = previousError*sigmoidPrime(non-activated output)DOT(currentWeights^T)DOT(upperInputs^T)
    private float[] hiddenErrors(layer curLayer, layer upperLayer)
    {
        //hidden errors
        float[][] previousErrors = new float[upperLayer.nodes.length][];
        for (int i = 0; i < previousErrors.length; i++)
        {
            previousErrors[i] = new float[upperLayer.nodes[i].errorAdjustments.length];
            for (int j = 0; j < previousErrors[i].length; j++)
            {
                previousErrors[i][j] = upperLayer.nodes[i].errorAdjustments[j];
            }
        }

        float[][] sigmoidPrimes = new float[1][];
        sigmoidPrimes[0] = new float[upperLayer.nodes.length];
        for (int i =0;i<sigmoidPrimes[0].length;i++)
        {
            sigmoidPrimes[0][i] = sigmoidPrime(upperLayer.nodes[i].previousSum);
        }
        float[][] weightsTransposed = new float[upperLayer.nodes.length][];
        for(int i =0;i<weightsTransposed.length;i++)
        {
            weightsTransposed[i] = upperLayer.nodes[i].currentWeights;
        }
        weightsTransposed = transpose(weightsTransposed);

        float[][] inputsTransposed = new float[1][];
        inputsTransposed[0] = new float[curLayer.nodes.length];
        for (int i = 0; i < inputsTransposed[0].length; i++)
        {
            inputsTransposed[0][i] = curLayer.nodes[i].output;
        }

        inputsTransposed = transpose(inputsTransposed);

        float[][] products = new float[1][];
        products[0] = dotProduct(sigmoidPrimes, weightsTransposed);
        products[0] = dotProduct(products, inputsTransposed);
        return dotProduct(products, previousErrors);
    }


    //The Total Error in respect to the Current Output
    private float errorTotalRespectCurOutput(float outcome, float trueOutcome)
    {
        return -(trueOutcome - outcome);
    }

    //The Current Output in respect to the Net Output
    private float curOutputRespectNetOutput(float outcome)
    {
        return outcome * (1 - outcome);
    }


    //The Net Output in respect to the current weights

    private float netOutputRespectCurWeights(float weight, float outcome)
    {
        return weight * outcome;
    }

    //The Total Error in respect to the current weights
    //The Net Output in respect to the current weights * The Total Error in respect to the Current Output * The Current Output in respect to the Net Output
    private float[] errorTotalRespectCurWeight(node curNode, node[] upperLayer)
    {
        //https://www.edureka.co/blog/backpropagation/
        //Just read it
        float[] errors = new float[curNode.currentWeights.length];
        for (int i = 0; i < errors.length; i++)
        {
            float errorTotal = 0;
            for (int j = 0; j < upperLayer[i].errorAdjustments.length; j++)
                errorTotal += upperLayer[i].errorAdjustments[j];
            float curOutput = upperLayer[i].previousSum;

            float netOutput = upperLayer[i].output;
            float curWeight = curNode.currentWeights[i];
            errors[i] = errorTotalRespectCurOutput(curOutput, errorTotal) * curOutputRespectNetOutput(netOutput) * netOutputRespectCurWeights(curWeight, netOutput);
        }
        return errors;
    }



    //(e^(x) - e^(-x)) / (e^(x) + e^(-x))
    private float tanh(float x)
    {
        float EX = (float)Math.exp(x);
        float negativeEX = (float)Math.exp(-x);
        return (EX - negativeEX) / (EX + negativeEX);
    }

    //Mean Square Error
    private float errorMSE(float outcome,float trueOutcome)
    {
        return (float)Math.pow(trueOutcome - outcome, 2);
    }

    //Error1 = w1/totalWeights * error
    private float[] findSingleError(float[] weights, float error)
    {
        float totalWeights = 0;
        for (int i = 0; i < weights.length; i++)
            totalWeights += weights[i];
        for (int i = 0; i < weights.length; i++)
            weights[i] = (weights[i] / totalWeights) * error;

        return weights;
    }

    //Errors = weights * previousErrors
    private float[] findMatrixErrors(node[] curLayer, node[] upperLayer)
    {
        //curLayer for weights
        //upperLayer for errors

        //weights: [node][weight]
        //errors: [weights of node][node]


        float[][] weights = new float[curLayer.length][];
        for (int i = 0; i < weights.length; i++)
            weights[i] = curLayer[i].currentWeights;

        float[][] errors = new float[upperLayer[0].currentWeights.length][];
        for (int i = 0; i < errors.length; i++)
        {
            errors[i] = new float[upperLayer.length];
            for (int j = 0; j < errors[i].length; j++)
            {
                errors[i][j] = upperLayer[j].errorAdjustments[i];
            }
        }
        errors = transpose(errors);
        return dotProduct(errors, weights);
    }

    //Dot product of two matricies
    private float[] dotProduct(float[][] errors, float[][] weights)
    {
        int size;
        if (errors.length > weights.length)
            size = errors.length;
        else
            size = weights.length;

        float[] outcomes = new float[size];

        for (int i = 0; i < size; i++)
        {
            outcomes[i] = 0;
            if (size == errors.length)
            {
                for (int j = 0; j < errors[i].length; j++)
                {
                    if (i >= weights.length || j>=errors[i].length)
                        break;
                    for (int w = 0; w < weights[i].length; w++)
                    {
                        outcomes[i] += weights[i][w] * errors[i][j];
                    }
                }
            }
            else
            {
                for (int j = 0; j < weights[i].length; j++)
                {
                    if (i >= errors.length || j >= weights.length)
                        break;
                    for (int w = 0; w < errors[i].length; w++)
                    {
                        outcomes[i] += weights[j][i] * errors[i][w];
                    }
                }
            }
        }
        return outcomes;
    }

    //flip matrix on its side
    private float[][] transpose(float[][] weights)
    {
        float[][] newWeights = new float[weights[0].length][];
        for (int i = 0; i < weights[0].length; i++)
        {
            newWeights[i] = new float[weights.length];
            for (int j = 0; j < weights.length; j++)
            {
                newWeights[i][j] = weights[j][i];
            }
        }
        return newWeights;
    }
}

