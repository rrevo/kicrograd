import io.kicrograd.Value
import io.kicrograd.times
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class EngineTest {

    private val delta = 0.0001f

    @Test
    fun testSingleVariable() {
        // Single variable fn -> 3x^2 + 4x  + 5
        val x = Value(4.0f)
        val xSq = x * x
        val threeXsq = 3.0f * xSq
        val fourX = 4.0f * x
        val y = threeXsq + fourX + 5.0f

        assertEquals(0.0f, y.gradient)
        y.backward()
        assertEquals(1.0f, y.gradient)

        // dy/dx = 6x + 4 @ x = 4 -> 28
        assertEquals(28.0f, x.gradient)
    }

    @Test
    fun testSingleVariableAtDifferentPoint() {
        // Single variable fn -> 3x^2 + 4x  + 5
        val x = Value(-2.0f)
        val xSq = x * x
        val threeXsq = 3.0f * xSq
        val fourX = 4.0f * x
        val y = threeXsq + fourX + 5.0f

        assertEquals(0.0f, y.gradient)
        y.backward()
        assertEquals(1.0f, y.gradient)

        // dy/dx = 6x + 4 @ x = -2 -> -8
        assertEquals(-8.0f, x.gradient)
    }

    @Test
    fun testMultiVariable() {
        // Multi variable fn L = (((a * b) + c) * f)
        val a = Value(2.0f)
        val b = Value(-3.0f)
        val c = Value(10.0f)
        val e = a * b
        val d = e + c
        val f = Value(-2.0f)
        val L = d * f

        assertEquals(0.0f, L.gradient)
        L.backward()

        // dL/dL = 1
        assertEquals(1.0f, L.gradient)

        // dL/df = value of d since its multiplication -> 4
        assertEquals(4.0f, f.gradient)

        // dL/dd = value of f -> -2
        assertEquals(-2.0f, d.gradient)

        // dL/dc = dL/dd * dd/dc aka chain rule = -2 * 1 -> -2
        assertEquals(-2.0f, c.gradient)
        assertEquals(c.gradient, e.gradient)

        // dL/da = (((a * b) + c) * f) @ a = 2, b = -3, c = 10, f = -2 -> 6
        assertEquals(6.0f, a.gradient)

        // dL/db  = dd/de * de/db -> -2 * 2 -> -4
        assertEquals(-4.0f, b.gradient)
    }

    @Test
    fun testNeuron() {
        // fn tanh(x1 * w1 + x2 * w2 + b)
        val x1 = Value(2.0f)
        val x2 = Value(0.0f)
        val w1 = Value(-3.0f)
        val w2 = Value(1.0f)
        val b = Value(6.8813735870195432f)

        val x1w1 = x1 * w1
        val x2w2 = x2 * w2
        val x1w1_x2w2 = x1w1 + x2w2
        val x1w1_x2w2_b = x1w1_x2w2 + b // n
        val y = x1w1_x2w2_b.tanh()

        assertEquals(0.7071067f, y.data, delta)

        y.backward()
        // dy/dy = 1
        assertEquals(1.0f, y.gradient)

        // dy/dn = 1 - tanh^2(n) -> 1 - 0.5 -> 0.5
        assertEquals(0.5f, x1w1_x2w2_b.gradient, delta)
        assertEquals(0.5f, x1w1_x2w2.gradient, delta)
        assertEquals(0.5f, b.gradient, delta)
        assertEquals(0.5f, x1w1.gradient, delta)
        assertEquals(0.5f, x2w2.gradient, delta)

        // dy/dw1
        assertEquals(1.0f, w1.gradient, delta)
        // dy/dw2
        assertEquals(0.0f, w2.gradient, delta)
    }
}
