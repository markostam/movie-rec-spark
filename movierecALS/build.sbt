name := "movierecALS"

version := "1.0"

scalaVersion := "2.12.1"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.10" % "2.0.2" exclude ("org.apache.hadoop","hadoop-yarn-server-web-proxy"),
  "org.apache.spark" % "spark-mllib_2.10" % "2.0.2" exclude ("org.apache.hadoop","hadoop-yarn-server-web-proxy")
)