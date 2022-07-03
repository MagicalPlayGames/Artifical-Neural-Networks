public class ProgramParameters
{

    public byte layers;
    public int[] nodes;
    public byte divisbleBy;
    public byte alogirthmNumber;
    public int iterations;
    public float learningRate;
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
