﻿using System;
using System.Collections.Generic;
using System.Text;

namespace CsharpANN
{
    class Interface
    {

        private static bool userInput = true;
        static void Main(string[] args)
        {
            int choice = 0;
            Program network;
            if (userInput)
            {
                Console.WriteLine("This ANN judges if a number is divisble by a number");
                while (choice < 3)
                {
                    Console.WriteLine("Input Arguments as follows");
                    Console.WriteLine("layerSize,nodeSizes per layer as an array,divisbleBy,numOfIterations,learningRate");
                    Console.WriteLine("Example:\n3,[8 4 1],3,1000,0.3");
                    string[] inputs = Console.ReadLine().Split(',');
                    string[] nodeSize = inputs[1].Substring(1, inputs[1].Length - 2).Split(' ');
                    int[] nodeSizeInputs = new int[nodeSize.Length];
                    for (int i = 0; i < nodeSize.Length; i++)
                    {
                        Int32.TryParse(nodeSize[i], out nodeSizeInputs[i]);
                    }
                    float[] inputsAsNums = new float[5];
                    for (int i = 0; i < 5; i++)
                    {
                        if (i != 1)
                        {
                            float.TryParse(inputs[i], out inputsAsNums[i]);
                        }
                    }

                    network = new Program(new ProgramParameters((byte)inputsAsNums[0], nodeSizeInputs, (byte)inputsAsNums[2], (byte)inputsAsNums[3], inputsAsNums[4],0));
                    choice = 1;
                    while (choice == 1)
                    {
                        choice = 0;
                        network.resetWeights();
                        while (choice == 0)
                        {
                            Console.WriteLine("\nAccuracy: " + network.testOp() + "\n\n");
                            Console.WriteLine("Select an option:\n0: Run with current weights\n1: Reset weights and run\n2: Restart\n3: Exit ");
                            Int32.TryParse(Console.ReadLine(), out choice);
                        }
                    }
                }
            }
            else
            {
                Console.WriteLine("Algorithm comparison test initiated");
                int[] node0 = { 8, 4, 4, 2, 2, 1 };
                int[] node1 = { 6, 4, 4, 2, 2, 1 };
                int[] node2 = { 8, 2, 4, 3, 2, 1 };
                int[] node3 = { 4, 4, 3, 2, 2, 1 };
                int[] node4 = { 4, 4, 4, 2, 1, 1 };
                int[] node5 = { 2, 4, 4, 2, 1, 1 };
                int[] node6 = { 6, 6, 4, 2, 2, 1 };
                int[][] nodes = { node0, node1, node2, node3,node4,node5,node6};
                float[] thisSet = new float[100];
                for (byte aN = 0; aN < 3; aN++)
                {
                    String topInfo = "";
                    float toptop = 0;
                    for (byte l = 1; l < 6; l++)
                    {
                        for (int n = 0; n < nodes.Length; n++)
                        {
                            for (float lR = 0.001f; lR < 1; lR *= 5)
                            {
                                network = setNewNetwork(l, nodes[n], 2, 10, lR, aN);
                                network.resetWeights();
                                float topOutput = 0;
                                for (int i = 0; i < 50; i++)
                                {
                                    thisSet[i] = network.testOp();
                                    if (topOutput < thisSet[i])
                                        topOutput = thisSet[i];
                                }
                                if (toptop < topOutput)
                                {
                                    topInfo = "Algorithm: " + aN + "\nLayers:" + l + " Nodes Index: " + n + " Learning Rate: " + lR + "\nBest Output: " + topOutput;
                                    toptop = topOutput;
                                }

                                //Console.WriteLine("Layers:" + l + " Nodes Index: " + n + " Learning Rate: " + lR + "\nBest Output: " + topOutput);
                            }
                        }
                    }
                    Console.WriteLine("Best learner:\n" + topInfo);
                }
            }
        }


            private static Program setNewNetwork(byte layers, int[] nodes, byte divisbleBy, int iterations, float learningRate,byte algoNum)
            {
            ProgramParameters input = new ProgramParameters(layers, nodes, divisbleBy, iterations, learningRate,algoNum);
                return new Program(input);
            }
    }
}
