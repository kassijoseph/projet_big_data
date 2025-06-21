from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TraitementCO2").getOrCreate()
df = spark.read.csv("data/co2.csv", header=True, inferSchema=True)
df = df.select("Nom", "Cout enerie", "Bonus / Malus")
df.show(5)
