using System;
using System.Collections.Generic;
using System.Text;

namespace CsharpANN
{
    class Interface
    {


        static void Main(string[] args)
        {
            int choice = 0;
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

                Program network = new Program((int)inputsAsNums[0], nodeSizeInputs, (int)inputsAsNums[2], (int)inputsAsNums[3], inputsAsNums[4]);
                choice = 1;
                while (choice == 1)
                {
                    choice = 0;
                    network.resetWeights();
                    while (choice ==0)
                    {
                        Console.WriteLine("\nAccuracy: " + network.testOp() + "\n\n");
                        Console.WriteLine("Select an option:\n0: Run with current weights\n1: Reset weights and run\n2: Restart\n3: Exit ");
                        Int32.TryParse(Console.ReadLine(),out choice);
                    }
                }
            }
        }
    }
}
