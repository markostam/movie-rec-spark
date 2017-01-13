import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}
import org.apache.spark.rdd._

// import data and split into useful RDD's with Ratin
val ratingsLoc = "/Users/markostamenovic/code/movierec-spark/data/ml-10M100K/ratings.dat"
val moviesLoc = "/Users/markostamenovic/code/movierec-spark/data/ml-10M100K/movies.dat"

val ratings = sc.textFile(ratingsLoc).map(x => x.split("::")).
  map(x => (x(3).toLong % 10, Rating(x(0).toInt, x(1).toInt, x(2).toDouble)))

val movies = sc.textFile(moviesLoc).map(x => x.split("::")).
  map(x => (x(0).toInt, x(1))).collect().toMap

val numRatings = ratings.count
val numUsers = ratings.map(_._2.user).distinct.count
val numMovies = ratings.map(_._2.product).distinct.count

println("Got " + numRatings + " ratings from "
  + numUsers + " users on " + numMovies + " movies.")

// split into train/test/validation

val numPartitions = 4
val training = ratings.filter(x => x._1 < 6).
  values.
  //union(myRatingsRDD).
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
                ratesAndPreds: RDD[((Int, Int), (Double, Double))]) = {
  val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
    val err = (r1 - r2)
    err * err
  }.mean()
  math.sqrt(MSE)
}

val ranks = List(3, 10, 30, 100)
val lambdas = List(0.01, 0.1, 1.0, 10.0)
val numIters = List(10, 30, 100)
val bestValidationRmse = Long.MaxValue

for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {
  val model = ALS.train(training, rank, numIter, lambda)

  val usersProducts = validation.map { case Rating(user, product, rate) =>
    (user, product) }
  val predictions =
    model.predict(usersProducts).map { case Rating(user, product, rate) =>
      ((user, product), rate) }
  val ratesAndPreds = validation.map { case Rating(user, product, rate) =>
    ((user, product), rate) }.join(predictions)

  val validationRmse = computeRmse(model, ratesAndPreds)
  println("RMSE (validation) = " + validationRmse + " for the model trained with rank = "
    + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")
  if (validationRmse < bestValidationRmse) {
    println("New best RMSE: " + validationRmse)
    val bestModel = Some(model)
    val bestValidationRmse = validationRmse
    val bestRank = rank
    val bestLambda = lambda
    val bestNumIter = numIter
  }
}

val myRatedMovieIds = myRatings.map(_.product).toSet
val candidates = sc.parallelize(movies.keys.filter(!myRatedMovieIds.contains(_)).toSeq)
val recommendations = bestModel.get.
  predict(candidates.map((0, _))).
  collect().
  sortBy(- _.rating).
  take(50)

var i = 1
println("Movies recommended for you:")
recommendations.foreach { r =>
  println("%2d".format(i) + ": " + movies(r.product))
  i += 1
}
