import com.google.common.primitives.Floats
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms.sigmoid
import org.nd4j.linalg.ops.transforms.Transforms.sigmoidDerivative
import java.io.FileInputStream
import java.lang.System.currentTimeMillis
import java.lang.System.nanoTime
import java.nio.ByteBuffer.wrap
import java.nio.ByteOrder
import java.util.*
import java.util.zip.GZIPInputStream

/**
 * @author Joseph Gardi
 */

fun main(args: Array<String>) {
    val start = currentTimeMillis()

    val inputLayers = Mnist.images.drop(100)
                                    .map { img -> Nd4j.create(Floats.toArray(img), intArrayOf(img.size, 1)) }
    val outputLayers = Mnist.labels.drop(100)
                                    .map { correctNumber ->
        val outputLayer = Nd4j.zeros(10, 1)
        outputLayer.putScalar(correctNumber, 1)
        outputLayer
    }
    val trainingData = inputLayers zip outputLayers

    val testData = inputLayers.take(100) zip Mnist.labels.take(100)

    val network = Network(listOf(Mnist.imageWidth * Mnist.imageHeight, 30, 10))

    val startSGD = currentTimeMillis()
    network.stochasticGradientDescent(trainingData, 30, 10, 3.0F)
    println("time for SGD: ${currentTimeMillis() - startSGD}")
    println(network.evaluate(testData))
    println("took ${currentTimeMillis() - start}")


    // draw it
//    invokeLater {
//        val fr = object: JFrame() {
//            override fun paint(g: Graphics?) {
//                val first = Mnist.images[10]
//                for (y in 0 until Mnist.imageHeight) {
//                    for (x in 0 until Mnist.imageWidth) {
//                        g!!.color = Color(first[x + y*28])
//                        // each pixel is a 10 by 10 square
//                        val pixelSize = 10
//                        g.fillRect(x*pixelSize, y*pixelSize, pixelSize, pixelSize)
//                    }
//                }
//            }
//        }
//        fr.size = Dimension(500, 500)
//        fr.isVisible = true
//    }
}

class Network(val sizes: List<Int>) {

    val numLayers = sizes.size

    var biases = sizes.drop(1).map { Nd4j.randn(it, 1) }

    var weights = (sizes.dropLast(1) zip sizes.drop(1))
            .map { (leftSize, rightSize) ->
        Nd4j.randn(rightSize, leftSize)
    }

    fun feedforward(inputs: INDArray): INDArray {
        var lastLayersOutput = inputs

        for ((b, w) in biases zip weights) {
            lastLayersOutput = w.mmul(lastLayersOutput).add(b)
        }

        return lastLayersOutput
    }


    /**
     * @param trainingData list of pairs (x, y) representing the training inputs and corresponding desired outputs
     */
    fun stochasticGradientDescent(trainingData: List<Pair<INDArray, INDArray>>,
                                  epochs: Int, batchSize: Int, learningRate: Float) {
        Collections.shuffle(trainingData)

        repeat (epochs) { i ->
            for ((batchIndex, batch) in trainingData.chunked(batchSize).withIndex()) {
                updateBatch(batch, learningRate)
//                println("did batch $batchIndex")
            }
            println("finished epoch $i")
        }
    }

    fun updateBatch(batch: List<Pair<INDArray, INDArray>>, learningRate: Float) {
        val startUpdate = currentTimeMillis()
        var nabla_b = biases.map { b -> Nd4j.zeros(*b.shape()) }
        var nabla_w = weights.map { w -> Nd4j.zeros(*w.shape()) }

        for ((inputs, outputs) in batch) {
            val startOne = nanoTime()
            val (deltaNablaB, deltaNablaW) = backprop(inputs, outputs)
//            println("time for backprop ${nanoTime() - startOne}")
            nabla_b = (nabla_b zip deltaNablaB).map { (nb, dnb) -> nb.add(dnb) }
            nabla_w = (nabla_w zip deltaNablaW).map { (nw, dnw) -> nw.add(dnw) }
//            println("time for one ${nanoTime() - startOne}")
        }

        biases = (biases zip nabla_b).map { (b, nb) ->  b.sub(nb.mul(learningRate/batch.size)) }
        weights = (weights zip nabla_w).map { (w, nw) ->  w.sub(nw.mul(learningRate/batch.size)) }
//        println("time for update ${currentTimeMillis() - startUpdate}")
    }

    private fun backprop(inputs: INDArray, outputs: INDArray): Pair<MutableList<INDArray>, MutableList<INDArray>> {
        val nablaB = biases.map { b -> Nd4j.zeros(*b.shape()) }
                            .toMutableList()
        val nablaW = weights.map { w -> Nd4j.zeros(*w.shape()) }
                            .toMutableList()

        // feedforward
        val activations = mutableListOf(inputs)  // list to store all the activations, layer by layer
        val zs = mutableListOf<INDArray>()  // list to store all the z vectors, layer by layer
        val startMul = nanoTime()
        for ((b, w) in biases zip weights) {
            val s = nanoTime()
            zs.add(w.mmul(activations.last()).add(b))
            println("t1: ${activations.last().size(0)}")
            activations.add(sigmoid(zs.last()))
        }

        // backward pass

        // for last layer
        val delta = costDerivative(activations.last(), outputs).mul(sigmoidDerivative(zs.last()))

        nablaB[nablaB.lastIndex] = delta
        nablaW[nablaW.lastIndex] = delta.mmul(activations[activations.lastIndex - 1].transpose())

        // go backwards through the layers. Aways keep track of the delta for the next layer
        var deltaForNextLayer = delta
        for (layer in (numLayers - 2) until 0) {
            val z = zs.last()
            deltaForNextLayer = weights[layer + 1].transpose().mmul(deltaForNextLayer).mul(sigmoidDerivative(z))
            nablaB[layer] = deltaForNextLayer
            nablaW[layer] = deltaForNextLayer.mmul(activations[layer + 1].transpose())
        }
        println("time mult is ${nanoTime() - startMul}")

        return nablaB to nablaW
    }

    fun evaluate(testData: List<Pair<INDArray, Int>>) =
        testData.filter { (inputs, maxIndex) ->
            val outputs = feedforward(inputs)
            val outputList = (0 until outputs.length()).map { outputs.getFloat(it) }
            var maxIndex = 0
            var max = outputList[0]
            for ((i, output) in outputList.withIndex()) {
                if (output > max) {
                    max = output
                    maxIndex = i
                }
            }
            maxIndex == maxIndex
        }
                .size
}

//fun sigmoid(z: Float) = 1.0 / (1 + exp(-z))

fun costDerivative(outputActivations: INDArray, correctActivations: INDArray) = outputActivations.sub(correctActivations)


/**
 * Idx files store The sizes of the dimensions at this index
 */
val indexOfDimensions = 4

/**
 * All Idx files begin with metadata describing number of dimensions, size of dimensions, and data type.
 * The part after that is the actual data. In the Mnist dataset the data consists of unsigned bytes.
 */
class MnistParser(path: String, numberOfDimensions: Int) {

    val bytes = GZIPInputStream(FileInputStream(path)).use {
        it.readBytes()
    }

    val indexOfData = indexOfDimensions + 4 * numberOfDimensions

    /**
     * I drop the metadata. The beginning of the mnist file doesn't have the actual data. The beginning has metadata about
     * the size of dimensions, number type, and number of dimensions. The data is the part after that.
     * Also I convert from unsigned bytes to signed ints
     */
    val data = bytes.drop(indexOfData)
                    .map { it.toInt() and 0xFF }
}

object Mnist {
    val images: List<List<Int>>
    val labels: List<Int>
    val imageWidth: Int
    val imageHeight: Int

    init {
        val parsedIdx = MnistParser("train-images-idx3-ubyte.gz", 3)

        val dimensionsBuffer = wrap(parsedIdx.bytes.sliceArray(indexOfDimensions until parsedIdx.indexOfData))
        val dimensions = IntArray(3)
        dimensionsBuffer.order(ByteOrder.BIG_ENDIAN).asIntBuffer().get(dimensions)

        imageWidth = dimensions[1]
        imageHeight = dimensions[2]
        val pixelsPerImage = imageWidth * imageHeight

        images = parsedIdx.data.chunked(pixelsPerImage)

        labels = MnistParser("train-labels-idx1-ubyte.gz", 1).data
    }
}
