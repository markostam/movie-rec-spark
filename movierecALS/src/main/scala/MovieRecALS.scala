import java.io.File

import scala.io.Source

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}


/**
  * Created by markostamenovic on 1/13/17.
  */
object MovieRecALS {

  def main(args: Array[String]) = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    if (args.length != 2) {
      println("Usage: [usb root directory]/spark/bin/spark-submit --driver-memory 2g --class MovieLensALS " +
        "target/scala-*/movielens-als-ssembly-*.jar movieLensHomeDir personalRatingsFile")
      sys.exit(1)
    }

    // set up environment

    val conf = new SparkConf()
      .setAppName("MovieLensALS")
      .set("spark.executor.memory", "4g")
    val sc = new SparkContext(conf)

    // load personal ratings

    //val testRatings = loadRatings(args(1))
    //val testRatingsRDD = sc.parallelize(myRatings, 1)

    val dataDir = args(0)

    val ratings = sc.textFile(new File(dataDir,"ratings.dat").toString).map(x => x.split("::")).
      map(x => (x(3).toLong % 10, Rating(x(0).toInt, x(1).toInt, x(2).toDouble)))

    val movies = sc.textFile(new File(dataDir,"movies.dat").toString).map(x => x.split("::")).
      map(x => (x(0).toInt, x(1))).collect().toMap

    val numRatings = ratings.count
    val numUsers = ratings.map(_._2.user).distinct.count
    val numMovies = ratings.map(_._2.product).distinct.count

    println("Got " + numRatings + " ratings from "
      + numUsers + " users on " + numMovies + " movies.")

    // split into train/test/validation

    val numPartitions = math.round(numRatings/2500000).toInt
    val training = ratings.filter(x => x._1 < 6).
      values.
      //union(testRatingsRDD).
      repartition(numPartitions).
      cache()
    val validation = ratings.filter(x => x._1 >= 6 && x._1 < 8).
      values.
      repartition(numPartitions).
      cache()
    val test = ratings.filter(x => x._1 >= 8).values.cache()

    val numTraining = training.count()
    val numValidation = validation.count()
    val numTest = test.count()

    println("Training: " + numTraining + ", validation: "
      + numValidation + ", test: " + numTest)

    def computeRmse(model: MatrixFactorizationModel,
                    ratesAndPreds: RDD[((Int, Int), (Double, Double))]) : Double = {
      val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
        val err = (r1 - r2)
        err * err
      }.mean()
      math.sqrt(MSE)
    }

    // define hyperparameter search space
    val ranks = List(3, 10, 30, 100)
    val numIters = List(10, 30)
    val lambdas = List(0.0001, 0.001, 0.01, 0.1)
    var bestValidationRmse = Long.MaxValue

    // do hyperparameter search

    //generate all possible combinations
    val combos = ranks.flatMap(x => numIters.flatMap(y => lambdas.map(z => (x,y,z))))

    def hyperParamSearch(combos: List[(Int,Int,Double)], training: RDD[Rating],
                          validation: RDD[Rating], bestValidationRmse: Double = Double.MaxValue,
                          returnList : List[(MatrixFactorizationModel, Double, Int, Double, Int)] = List()) :
                          (MatrixFactorizationModel, Double, Int, Double, Int) = {

      // private func to allow tail recursion and prevent stack overflows
      @scala.annotation.tailrec
      def hyperParamSearchf(combos: List[(Int, Int, Double)], training: RDD[Rating],
                            validation: RDD[Rating], bestValidationRmse: Double,
                            returnList: List[(MatrixFactorizationModel, Double, Int, Double, Int)]):
      (MatrixFactorizationModel, Double, Int, Double, Int) = {

        val (rank, numIter, lambda) = combos.head
        val model = ALS.train(training, rank, numIter, lambda)

        val usersProducts = validation.map { case Rating(user, product, rate) =>
          (user, product)
        }
        val predictions =
          model.predict(usersProducts).map { case Rating(user, product, rate) =>
            ((user, product), rate)
          }
        val ratesAndPreds = validation.map { case Rating(user, product, rate) =>
          ((user, product), rate)
        }.join(predictions)
        val validationRmse: Double = computeRmse(model, ratesAndPreds)

        println("RMSE (validation) = " + validationRmse + " for the model trained with rank = "
          + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")

        if (combos.tail.isEmpty) {
          returnList.minBy(_._2)
        }
        else {
          val updatedReturnList = (model, validationRmse, rank, lambda, numIter) :: returnList
          hyperParamSearchf(combos.tail, training, validation, validationRmse, updatedReturnList)
        }
      }
      hyperParamSearchf(combos, training, validation, bestValidationRmse, returnList)
    }

    val bestSettings = hyperParamSearch(combos, training, validation)

  }
}
