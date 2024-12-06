package io.kicrograd

enum class Operator(val op: String) {
    PLUS("+"),
    TIMES("*"),
    TANH("tanh"),
    RELU("relu"),
    NONE("none"),
}

class Value(
    val data: Float,
    private val children: List<Value> = emptyList(),
    private val operator: Operator = Operator.NONE
) {
    private var gradient = 0.0f

    operator fun plus(other: Value): Value {
        return Value(data + other.data, listOf(this, other), Operator.PLUS)
    }

    operator fun times(other: Value): Value {
        return Value(data * other.data, listOf(this, other), Operator.TIMES)
    }

    fun tanh(): Value {
        val expValue = Math.exp(2 * data.toDouble()).toFloat();
        val t = (expValue - 1.0f) / (expValue + 1.0f)
        return Value(t, listOf(this), Operator.TANH)
    }

    fun relu(): Value {
        return Value(Math.max(0.0f, data), listOf(this), Operator.RELU)
    }

    operator fun plus(other: Float): Value {
        return plus(Value(other))
    }

    operator fun times(other: Float): Value {
        return times(Value(other))
    }

    fun gradient(): Float {
        return this.gradient
    }

    fun backward() {
        this.gradient = 1.0f

        // simplified topological sort
        val topoList = mutableListOf<Value>()
        val visited = mutableSetOf<Value>()
        fun buildTopo(v: Value) {
            if (!visited.contains(v)) {
                visited.add(v)
                v.children.forEach { buildTopo(it) }
                topoList.add(v)
            }
        }
        buildTopo(this)

        // initialize final gradient to 1.0
        this.gradient = 1.0f
        topoList.reversed().forEach { it.localBackwardPropagation() }
    }

    // Set the gradient of the child nodes based on the operator and this.gradient
    // Number of children is fixed based on the operator
    private fun localBackwardPropagation() {
        when (operator) {
            Operator.PLUS -> {
                children[0].gradient += gradient
                children[1].gradient += gradient
            }

            Operator.TIMES -> {
                children[0].gradient += children[1].data * gradient
                children[1].gradient += children[0].data * gradient
            }

            Operator.TANH -> {
                children[0].gradient += (1 - data * data) * gradient
            }

            Operator.RELU -> {
                children[1].gradient += if (data > 0) gradient else 0.0f
            }

            Operator.NONE -> {
                // do nothing
            }
        }
    }

    override fun toString(): String {
        return "Value(data=$data, gradient=$gradient)"
    }

}

// Numeric overloading for Value
operator fun Int.times(value: Value): Value {
    return Value(this.toFloat()) * value
}

operator fun Int.plus(value: Value): Value {
    return Value(this.toFloat()) + value
}

operator fun Float.times(value: Value): Value {
    return Value(this) * value
}
