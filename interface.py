from PythonANN import Program
from ProgramParameters import ProgramParameters


def setNewNetwork(layers, nodes, divisbleBy, iterations, learningRate,algoNum):
    input = ProgramParameters();
    input._init_(layers, nodes, divisbleBy, iterations, learningRate,algoNum);
    return Program(input);

userInput = True;
choice = 0;
network = None;
if (userInput):
    print("This ANN judges if a number is divisble by a number");
    while (choice < 3):
        print("Input Arguments as follows");
        print("layerSize,nodeSizes per layer as an array,divisbleBy,numOfIterations,learningRate");
        print("Example:\n3,[8 4 1],3,1000,0.3");
        inputs = input().split(',');
        nodeSize = inputs[1][1: len(inputs[1]) - 1].split(' ');
        nodeSizeInputs = [0]*len(nodeSize);
        for i in range(len(nodeSize)):
            nodeSizeInputs[i] = int(nodeSize[i]);
        inputsAsNums = [0.0]*5;
        for i in range(4):
            if (i != 1):
                inputsAsNums[i] = int(inputs[i])
        inputsAsNums[4] = float(inputs[4]);

        network = setNewNetwork(inputsAsNums[0], nodeSizeInputs, inputsAsNums[2], inputsAsNums[3], inputsAsNums[4],1);
        choice = 1;
        while (choice == 1):
            choice = 0;
            network.resetWeights();
            while (choice == 0):
                print("\nAccuracy: " + str(network.testOp()) + "\n\n");
                print("Select an option:\n0: Run with current weights\n1: Reset weights and run\n2: Restart\n3: Exit ");
                choice = int(input());
else:
    print("Algorithm comparison test initiated");
    node0 = [ 8, 4, 4, 2, 2, 1 ];
    node1 = [ 6, 4, 4, 2, 2, 1 ];
    node2 = [ 8, 2, 4, 3, 2, 1 ];
    node3 = [ 4, 4, 3, 2, 2, 1 ];
    node4 = [ 4, 4, 4, 2, 1, 1 ];
    node5 = [ 2, 4, 4, 2, 1, 1 ];
    node6 = [ 6, 6, 4, 2, 2, 1 ];
    nodes = [ node0, node1, node2, node3,node4,node5,node6];
    thisSet = [0.0]*100;
    for aN in range(3):
        topInfo = "";
        toptop = 0;
        for l in range(1,6):
            for n in range(len(nodes)):
                lR = 0.001;
                while(lR<1.0):
                    network = setNewNetwork(l, nodes[n], 2, 10, lR, aN);
                    network.resetWeights();
                    topOutput = 0;
                    for i in range(50):
                        thisSet[i] = network.testOp();
                        if (topOutput < thisSet[i]):
                            topOutput = thisSet[i];
                    if (toptop < topOutput):
                        topInfo = "Algorithm: " + str(aN) + "\nLayers:" + str(l) + " Nodes Index: " + str(n) + " Learning Rate: " + str(lR) + "\nBest Output: " + str(topOutput);
                        toptop = topOutput;
                    lR *= 5.0;
        print("Best learner:\n" + topInfo);


