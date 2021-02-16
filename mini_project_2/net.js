function weightInitializer(input_neurons, output_neurons){

    /*
    
        Returns the weight matrix (2D list) of shape output_neurons x input_neurons
        Each weight value is taken from a Gaussian distribution of mean 0 and 
        standard deviation sqrt(2/input_neurons)

        Already existing functions that you might need to call in this definition:
        sqrt(num) returns square root of a float
        randomGaussian(μ, σ) returns a value from a Gaussian distribution with mean μ and std σ
    */

    // Write your code here
}

function biasInitializer(neurons){

    /*
        Returns the bias vector (1D list) of length neurons
        Initialized as all zeros
    */
    // Write your code here
}

function relu(x){

    /*
        This function takes two types of x's.
        Type1 is a list (which is an object in js) : Returns relu applied on each element of the list 
        Type2 is a number: Returns relu of that number
    */

    if(typeof(x)=="object"){
        
        /*
            x is a list
            return a list whose i-th element is relu's output of the i-th element of x
        */
        // Write your code here
    }
    else{

        /*
            x is a number
            return relu's output of x
        */
        // Write your code here
    }
}

function binarize(x){

    /*
        This function takes two types of x's.
        Type1 is a list (which is an object in js) : Returns binarize applied on each element of the list 
        Type2 is a number: Returns binarize operation applied on that number

        The binarize operation on a number x is the following:
        if x is negative, return 0
        else return 1
    */

    if(typeof(x)=="object"){
       /*
            x is a list
            return a list whose i-th element is the binarize output of the i-th element of x
        */
        // Write your code here
    }
    else{
        /*
            x is a number
            return binarize's output of number x
        */
        // Write your code here
    }
}

class Layer{
    constructor(input_neurons, output_neurons){
        this.weight = weightInitializer(input_neurons, output_neurons);
        this.bias = biasInitializer(output_neurons);
    }
    forward(X){

        /*
            Implement the forward function of a feedforward layer of a neural network
            returns the matrix multiplication of this.weight and X after which this.bias has been added : (WX + b)

            Already existing functions that you might need to call in this definition:
            
            Here M1 and M2 are 2D lists in JavaScript,

            matrixmultiplication(M1, M2): returns the matrix multiplication of M1 and M2
            matrixaddition(M1, M2): returns the matrix addition of M1 and M2
        */

        // Write your code here

    }
    set(layer){
        for(let i=0; i<this.weight.length; i++){
            for(let j=0; j<this.weight[i].length; j++){
                this.weight[i][j] = layer.weight[i][j];
            }
        }
        for(let i=0; i<this.bias.length;i++){
            this.bias[i] = layer.bias[i];
        }
    }

    mutate(){
        for(let i=0; i<this.weight.length; i++){
            for(let j=0; j<this.weight[i].length; j++){

                /*
                    adds values sampled from a Gaussian Distribution of mean 0 and std mutationParameter to 
                    all the weight values 


                    Already existing functions that you might need to call in this definition:
                    randomGaussian(μ, σ) returns a value from a Gaussian distribution with mean μ and std σ
                */

                // Write your code here

            }
        }
        for(let i=0; i<this.bias.length;i++){

             /*
                    adds values sampled from a Gaussian Distribution of mean 0 and std mutationParameter to 
                    all the bias values 

                    Already existing functions that you might need to call in this definition:
                    randomGaussian(μ, σ) returns a value from a Gaussian distribution with mean μ and std σ
            */

            // Write your code here

        }
    }
}

class Network{
    constructor(){
            
        /*

            Takes 7 distances (covered by rays) as input and has 4 binary outputs

            Binary Outputs:

            0: FA : Forward Acceleration 
            1: BA : Backward Acceleration
            2: TL : Turn Left
            3: TR : Turn Right

        */

        this.layer1 = new Layer(7, 10);
        this.layer2 = new Layer(10, 10);
        this.layer3 = new Layer(10, 10);
        this.layer4 = new Layer(10, 4);
    }
    forward(X){

        /*
            Implement the forward function of this neural network
            by calling forward functions of the intermediate layers 
            and calling relu activation functions in between
            The final output is binary and hence the output of
            layer4 must be passed through the binarize function which
            is already implemented in this file
        */
         // Write your code here
    }
    set(network){
        this.layer1.set(network.layer1);
        this.layer2.set(network.layer2);
        this.layer3.set(network.layer3);
        this.layer4.set(network.layer4);
    }
    mutate(){
        this.layer1.mutate();
        this.layer2.mutate();
        this.layer3.mutate();
        this.layer4.mutate();  
    }
}
