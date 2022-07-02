using System;
using System.Collections.Generic;
using System.Text;

namespace CsharpANN
{
    public class ProgramParameters
    {

        public byte layers { get; set; }
        public int[] nodes { get; set; }
        public byte divisbleBy { get; set; }
        public byte alogirthmNumber { get; set; }
        public int iterations { get; set; }
        public float learningRate { get; set; }
        public ProgramParameters(byte layers1, int[] nodes1, byte divisbleBy1, int iterations, float learningRate, byte algoNum)
        {
            this.alogirthmNumber = algoNum;
            this.layers = layers1;
            this.nodes = nodes1;
            this.divisbleBy = divisbleBy1;
            this.iterations = iterations;
            this.learningRate = learningRate;
        }
    }
}
