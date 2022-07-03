import java.util.Scanner;

public class Main {
    private static Scanner inputReader = new Scanner(System.in);
    private static boolean userInput = false;

    public static void main(String[] args) {
        int choice = 0;
        Program network;
        if (userInput) {
            System.out.println("This ANN judges if a number is divisble by a number");
            while (choice < 3) {
                System.out.println("Input Arguments as follows");
                System.out.println("layerSize,nodeSizes per layer as an array,divisbleBy,numOfIterations,learningRate");
                System.out.println("Example:\n3,[8 4 1],3,1000,0.3");
                String[] inputs = inputReader.nextLine().split(",");
                String[] nodeSize = inputs[1].substring(1, inputs[1].length() - 1).split(" ");
                int[] nodeSizeInputs = new int[nodeSize.length];
                for (int i = 0; i < nodeSize.length; i++) {
                    nodeSizeInputs[i] = Byte.parseByte(nodeSize[i]);
                }
                float[] inputsAsNums = new float[5];
                for (int i = 0; i < 5; i++) {
                    if (i != 1) {
                        inputsAsNums[i] = Float.parseFloat(inputs[i]);
                    }
                }

                network = new Program(new ProgramParameters((byte) inputsAsNums[0], nodeSizeInputs, (byte) inputsAsNums[2], (int) inputsAsNums[3], inputsAsNums[4], (byte) 0));
                choice = 1;
                while (choice == 1) {
                    choice = 0;
                    network.resetWeights();
                    while (choice == 0) {
                        System.out.println("\nAccuracy: " + network.testOp() + "\n\n");
                        System.out.println("Select an option:\n0: Run with current weights\n1: Reset weights and run\n2: Restart\n3: Exit ");
                        choice = inputReader.nextByte();
                    }
                }
                inputReader.nextLine();
            }
        } else {
            System.out.println("Algorithm comparison test initiated");
            int[] node0 = {8, 4, 4, 2, 2, 1};
            int[] node1 = {6, 4, 4, 2, 2, 1};
            int[] node2 = {8, 2, 4, 3, 2, 1};
            int[] node3 = {4, 4, 3, 2, 2, 1};
            int[] node4 = {4, 4, 4, 2, 1, 1};
            int[] node5 = {2, 4, 4, 2, 1, 1};
            int[] node6 = {6, 6, 4, 2, 2, 1};
            int[][] nodes = {node0, node1, node2, node3, node4, node5, node6};
            float[] thisSet = new float[100];
            for (byte aN = 0; aN < 3; aN++) {
                String topInfo = "";
                float toptop = 0;
                for (byte l = 1; l < 6; l++) {
                    for (int n = 0; n < nodes.length; n++) {
                        for (float lR = 0.001f; lR < 1; lR *= 5) {
                            network = setNewNetwork(l, nodes[n], (byte) 2, 10, lR, aN);
                            network.resetWeights();
                            float topOutput = 0;
                            for (int i = 0; i < 50; i++) {
                                thisSet[i] = network.testOp();
                                if (topOutput < thisSet[i])
                                    topOutput = thisSet[i];
                            }
                            if (toptop < topOutput) {
                                topInfo = "Algorithm: " + aN + "\nLayers:" + l + " Nodes Index: " + n + " Learning Rate: " + lR + "\nBest Output: " + topOutput;
                                toptop = topOutput;
                            }

                            //System.out.println("Layers:" + l + " Nodes Index: " + n + " Learning Rate: " + lR + "\nBest Output: " + topOutput);
                        }
                    }
                }
                System.out.println("Best learner:\n" + topInfo);
            }
        }
    }


    private static Program setNewNetwork(byte layers, int[] nodes, byte divisbleBy, int iterations, float learningRate, byte algoNum) {
        ProgramParameters input = new ProgramParameters(layers, nodes, divisbleBy, iterations, learningRate, algoNum);
        return new Program(input);
    }
}
