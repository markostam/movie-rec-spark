import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}

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

// split training data

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

val rank = 10
val numIterations = 10
val lambda = 0.01
val model = ALS.train(training, rank, numIterations, lambda)

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

val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
  val err = (r1 - r2)
  err * err
}.mean()

