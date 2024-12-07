import io.kicrograd.*
import org.junit.jupiter.api.Test

class NNTest {

    @Test
    fun testMLP() {
        val mlp = MLP(3, listOf(4, 4, 1))
        val inputs = listOf(
            listOf(2.0, 3.0, -1.0).map { Value(it.toFloat(), type = ValueType.INPUT) },
            listOf(3.0, -1.0, 0.5).map { Value(it.toFloat(), type = ValueType.INPUT) },
            listOf(0.5, 1.0, 1.0).map { Value(it.toFloat(), type = ValueType.INPUT) },
            listOf(1.0, 1.0, -1.0).map { Value(it.toFloat(), type = ValueType.INPUT) },
        )
        val outputs = listOf(1.0, -1.0, -1.0, 1.0).map { Value(it.toFloat(), type = ValueType.OUTPUT) }

        val predictionsIndex = 10
        val epochs = predictionsIndex * 20
        var previousPrediction = outputs.map { Value(0.0f) }

        for (i in 0..epochs) {
            // forward pass
            val outputPredictions = inputs.map { mlp.call(it) }.flatten()

            if (i % predictionsIndex == 0) {
                println("Epoch: $i Target     : ${outputs.map { it.data }}")
                if (i != 0) {
                    println("Epoch: $i Previous   : ${previousPrediction.map { it.data }}")
                }
                println("Epoch: $i Predictions: ${outputPredictions.map { it.data }}")
            }

            // loss calculation
            val loss = lossCalculation(outputs, outputPredictions)
            if (i % predictionsIndex == 0) {
                previousPrediction = outputPredictions
            }
            println("Epoch: $i: Loss: ${loss.data}")

            // backward pass
            mlp.zeroGrad()
            loss.backward()

            // update weights and bias
            for (param in mlp.parameters()) {
                param.data += param.gradient * -0.01f
            }
        }
    }

    // Mean Squared Error
    // L = sum E (y - y')^2
    private fun lossCalculation(outputs: List<Value>, outputPredictions: List<Value>): Value {
        var loss = Value(0.0f)
        for (i in outputs.indices) {
            // since minus or square has not been implemented, we are manually calculating it
            val outputDifferenceSquared = outputPredictions[i].plus(outputs[i].times(-1.0f))
                .times(outputPredictions[i].plus(outputs[i].times(-1.0f)))
            // summing up the loss across all the outputs
            loss += outputDifferenceSquared
        }
        return loss
    }
}