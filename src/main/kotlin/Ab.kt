import org.ejml.data.*
import org.ejml.dense.row.RandomMatrices_DDRM
import org.ejml.dense.row.RandomMatrices_FDRM
import org.ejml.simple.SimpleMatrix
import org.jblas.FloatMatrix
import java.lang.System.currentTimeMillis
import java.util.*


var a = 3
fun main(args: Array<String>) {
    val b = 2

    val start = currentTimeMillis()
    val rand = Random()
    val m10b11 = DMatrixRMaj(10, 11)
    val m11b10 = DMatrixRMaj(11, 10)
    val m1000b1 = DMatrixRMaj(1000, 1)
    val m1b1000 = DMatrixRMaj(1, 1000)
    RandomMatrices_DDRM.fillGaussian(m10b11, 0.0, 1.0, rand)
    RandomMatrices_DDRM.fillGaussian(m11b10, 0.0, 1.0, rand)
    RandomMatrices_DDRM.fillGaussian(m1000b1, 0.0, 1.0, rand)
    RandomMatrices_DDRM.fillGaussian(m1b1000, 0.0, 1.0, rand)
    for (i in 0 until 10000) {
        SimpleMatrix.wrap(m10b11).mult(SimpleMatrix.wrap(m11b10))
        SimpleMatrix.wrap(m1000b1).mult(SimpleMatrix.wrap(m1b1000))
//        Nd4j.randn(10, 11).mmul(Nd4j.randn(11, 10))
//        Nd4j.randn(900, 1).mmul(Nd4j.randn(1, 900))
//        RandomMatrices_FDRM.fillGaussian(null, 0F, 1F, rand)
//        SimpleMatrix.random32(10, 11, -10F, 10F, rand)
//        FloatMatrix.randn(10, 11).mmul(FloatMatrix.randn(11, 10))
//        FloatMatrix.randn(1000, 1).mmul(FloatMatrix.randn(1, 1000))
//        val A = DoubleMatrix(arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0), doubleArrayOf(7.0, 8.0, 9.0)))
//        val x = DoubleMatrix(arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(1.0, 2.0, 3.0)))
//        val y: DoubleMatrix
//
//        y = A.mmul(x)

    }
//    val A = DoubleMatrix(arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0), doubleArrayOf(7.0, 8.0, 9.0)))
//    val x = DoubleMatrix(arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(1.0, 2.0, 3.0)))
//    val y: DoubleMatrix

//     println("result is ${A.mmul(x)[intArrayOf(0, 0)]}")

    println("it took ${currentTimeMillis() - start}")
}

fun anEx() {
    println("hey ")
}
// with double for 10000 is 61109
