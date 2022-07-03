
class ProgramParameters:
    layers = None;
    nodes = None;
    divisbleBy = None;
    alogirthmNumber = None;
    iterations = None;
    learningRate = None;
    def _init_(self,layers1, nodes1, divisbleBy1, iterations, learningRate, algoNum):
        self.alogirthmNumber = algoNum;
        self.layers = layers1;
        self.nodes = nodes1;
        self.divisbleBy = divisbleBy1;
        self.iterations = iterations;
        self.learningRate = learningRate;
        
        
        
        