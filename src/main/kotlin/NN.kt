package io.kicrograd

import kotlin.random.Random

abstract class Module {
    abstract fun parameters(): List<Value>

    fun zeroGrad() {
        parameters().forEach { it.gradient = 0.0f }
    }
}

class Neuron(
    inputCount: Int,
    private val activation: Operator = Operator.NONE,
) : Module() {
    private val weights: MutableList<Value> = mutableListOf()
    private var bias: Value

    init {
        require(inputCount > 0) { "inputCount must be greater than 0" }
        require(
            activation == Operator.NONE ||
                    activation == Operator.RELU ||
                    activation == Operator.TANH
        ) {
            "activation must be NONE, RELU or TANH"
        }

        val r = Random
        val min = -1.0f
        val max = 1.0f

        fun generateNonZeroFloat(): Float {
            var randomFloat = r.nextFloat()
            while (randomFloat == 0f) {
                randomFloat = r.nextFloat()
            }
            return randomFloat
        }

        // WTF
        fun generateFloatInRange(): Float {
            return min + generateNonZeroFloat() * (max - min)
        }

        for (i in 0 until inputCount) {
            weights.add(Value(generateFloatInRange(), type = ValueType.PARAMETER))
        }
        bias = Value(generateFloatInRange(), type = ValueType.PARAMETER)
    }

    fun call(inputs: List<Value>): Value {
        require(inputs.size == weights.size) { "inputs.size must be equal to weights.size" }
        var sum = bias

        for (i in inputs.indices) {
            sum += inputs[i] * weights[i]
        }
        val output = when (activation) {
            Operator.TANH -> sum.tanh()
            Operator.RELU -> sum.relu()
            else -> sum
        }
        output.type = ValueType.OUTPUT
        return output
    }

    override fun parameters(): List<Value> {
        val params = mutableListOf<Value>()
        params.addAll(weights)
        params.add(bias)
        return params
    }

    override fun toString(): String {
        val prefix = when (activation) {
            Operator.TANH -> "TanH"
            Operator.RELU -> "RelU"
            else -> "Linear"
        }
        return prefix + "Neuron(weights=$weights, bias=$bias)"
    }
}

class Layer(
    private val inputCount: Int,
    outputCount: Int,
    activation: Operator = Operator.NONE,
) : Module() {
    private val neurons: MutableList<Neuron> = mutableListOf()

    init {
        require(inputCount > 0) { "inputCount must be greater than 0" }
        require(outputCount > 0) { "outputCount must be greater than 0" }

        for (i in 0 until outputCount) {
            neurons.add(Neuron(inputCount, activation))
        }
    }

    fun call(inputs: List<Value>): List<Value> {
        require(inputs.size == inputCount) { "inputs.size must be equal to inputCount" }
        val outputs = mutableListOf<Value>()
        for (neuron in neurons) {
            outputs.add(neuron.call(inputs))
        }
        return outputs
    }

    override fun parameters(): List<Value> {
        val params = mutableListOf<Value>()
        for (neuron in neurons) {
            params.addAll(neuron.parameters())
        }
        return params
    }

    override fun toString(): String {
        return "Layer(neurons=${neurons.size})"
    }
}

class MLP(
    inputCount: Int,
    outputLayerCount: List<Int>,
) : Module() {
    private val layers: MutableList<Layer> = mutableListOf()

    init {
        require(inputCount > 0) { "inputCount must be greater than 0" }
        require(outputLayerCount.isNotEmpty()) { "outputLayerCount must not be empty" }

        val layerCounts = listOf(inputCount) + outputLayerCount

        var prevLayerOutputCount = inputCount
        for (i in 0 until layerCounts.size - 1) {
            layers.add(Layer(prevLayerOutputCount, layerCounts[i], Operator.RELU))
            prevLayerOutputCount = layerCounts[i]
        }
        // Add output layer
        layers.add(Layer(prevLayerOutputCount, outputLayerCount.last(), Operator.NONE))
    }

    fun call(inputs: List<Value>): List<Value> {
        var outputs = inputs
        for (layer in layers) {
            outputs = layer.call(outputs)
        }
        return outputs
    }

    override fun parameters(): List<Value> {
        val params = mutableListOf<Value>()
        for (layer in layers) {
            params.addAll(layer.parameters())
        }
        return params
    }

    override fun toString(): String {
        return "MLP(layers=$layers)"
    }
}