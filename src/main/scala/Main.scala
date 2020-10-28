import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{first, last, _}
import org.apache.spark.sql.functions.count

object Main {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.setMaster("local[2]")
    conf.setAppName("Enkripsi Big Data")
    val sc = new SparkContext(conf)
    val spark = SparkSession
      .builder()
      .appName("Enkripsi Big Data")
      .getOrCreate()
    // membaca csv dan membersihkan headernya. || jika menggunakan spark.read
    val inputPath = args(0)
    val outputPath = args(1)
    val thresholdParam = args(2)
    val dataframe = spark.read.option("header","true").csv(inputPath)
    //    val dataframe = spark.read.option("header","true").csv("input/dataset1.csv")
    val fs = FileSystem.get(sc.hadoopConfiguration)

    val bucketPath = new Path("bucket")
    if (fs.exists(bucketPath))
      fs.delete(bucketPath, true)

    val value_df = dataframe.selectExpr("cast(value as double) value").na.drop()
    value_df.createOrReplaceTempView("values_df")
    // mengurutkan nilai yang dienkrip menaik dan menghilangkan nilai duplikat (Growth Phase)
    val sorted_df = spark.sql("SELECT value FROM values_df ORDER BY value ASC").dropDuplicates("value")
    //    val indexed_df = sorted_df.withColumn("id",monotonically_increasing_id()) Index
    //    https://stackoverflow.com/questions/47894877/spark-monotonically-increasing-id-not-working-as-expected-in-dataframe Resikonya tidak dipararelkan
    val indexed_df = sorted_df.withColumn("id",row_number().over(Window.orderBy(lit(0)))-1).withColumn("gradient",lit(0.0)).withColumn("intercept",lit(0.0)).withColumn("prediction",lit(0.0)).withColumn("difference",lit(0.0))
    indexed_df.show()

    // threshold untuk bucket ini harus jadi parameter
    val threshold = thresholdParam.toInt

    bucketization(spark,indexed_df,threshold)
    val joinedBucket = joinAllBucket(spark,"x")
    joinedBucket.createOrReplaceTempView("joinedBucket")
    val cleanBucket = spark.sql("SELECT DISTINCT CAST(gradient as double), CAST(intercept as double), CAST(minId as int), CAST(maxId as int), CAST(minValue as double), CAST(maxValue as double), CAST(width as double) FROM joinedBucket")
    cleanBucket.createOrReplaceTempView("cleanBucket")

    spark.udf.register( "getQuadraticCoefficient", getQuadraticCoefficient _ )
    spark.udf.register( "getScaleFactorMinimum", getScaleFactorMinimum _ )
    spark.udf.register( "getMinimumWidth", getMinimumWidth _ )
    spark.udf.register( "getMinimumWidth", getMinimumWidth _ )
    spark.udf.register( "getScaleFactor", getScaleFactor _ )
    spark.udf.register( "getFlatValue", getFlatValue _ )
    spark.udf.register( "getEncryptedValue", getEncryptedValue _ )
    spark.udf.register( "getQuadraticCoefficientFromWidth", getQuadraticCoefficientFromWidth _ )
    spark.udf.register( "getScaleFactorFromWidth", getScaleFactorFromWidth _ )
    spark.udf.register( "getPlaintextRemainQuadraticCoefAndScaleFactor", getPlaintextRemainQuadraticCoefAndScaleFactor _ )
    spark.udf.register( "predictValue", predictValue _ )

    // tahap selanjutnya adalah persiapan untuk mencari scale factor
    val prepareBucket1 = spark.sql("SELECT gradient, intercept, minId, maxId, minValue, maxValue, width, getQuadraticCoefficient(gradient,intercept) as quadraticCoefficient FROM cleanBucket")
    prepareBucket1.createOrReplaceTempView("prepareTable")
    val prepareBucket2 = spark.sql("SELECT gradient, intercept, minId, maxId, minValue, maxValue,width, quadraticCoefficient, getScaleFactorMinimum(quadraticCoefficient,width) as scaleFactorMinimum FROM prepareTable")
    prepareBucket2.createOrReplaceTempView("prepareTable")
    val prepareBucket3 = spark.sql("SELECT gradient, intercept, minId, maxId, minValue, maxValue, width, quadraticCoefficient, scaleFactorMinimum, getMinimumWidth(quadraticCoefficient,scaleFactorMinimum,width) as minimumWidth FROM prepareTable")
    val prepareBucket4 = getK(spark,prepareBucket3)
    prepareBucket4.createOrReplaceTempView("prepareTable")
    // lakukan cross join antara data kotor dengan data bucket (data kotor bisa voted_df / sorted_df)
    val prepareBucket5 = prepareBucket4.crossJoin(value_df)
    val prepareBucket6 = getN(spark,prepareBucket5)
    prepareBucket6.createOrReplaceTempView("prepareTable")
    val prepareBucket7 = spark.sql("SELECT gradient, intercept, minId, maxId, minValue, maxValue, width, quadraticCoefficient, scaleFactorMinimum, minimumWidth, K, n, getScaleFactor(K,n,quadraticCoefficient,width,scaleFactorMinimum)as scaleFactor FROM prepareTable")
    prepareBucket7.createOrReplaceTempView("prepareTable")
    //    kamu mengubah prepare bucket8 dengan mendelete gradient,intercept,K, dan encrypted min val, encrypted max val, , scaleFactorMinimum, minimumWidth, K, n,
    val prepareBucket8 = spark.sql("SELECT minId, maxId, minValue, maxValue, width, quadraticCoefficient, scaleFactor, minimumWidth,K,n FROM prepareTable ORDER BY minId ASC")
    val prepareBucket9 = prepareBucket8.withColumn("bucketId",row_number().over(Window.orderBy(lit(0))))
    prepareBucket9.createOrReplaceTempView("prepareTable")
    val prepareBucketFinal = spark.sql("SELECT minId, maxId, minValue, maxValue, width, quadraticCoefficient, scaleFactor, minimumWidth,K,n,CAST(bucketId as int) FROM prepareTable ORDER BY minId ASC")
    val prepareBucketCopy = prepareBucketFinal.selectExpr("CAST(bucketId as int) as bucketIdCopy","CAST(minId as int) as minIdCopy","CAST(maxId as int) as maxIdCopy","CAST(minValue as double) as minValueCopy","CAST(maxValue as double) as maxValueCopy","CAST(width as double) as widthCopy","CAST(quadraticCoefficient as double) as quadraticCoefficientCopy","CAST(scaleFactor as double) as scaleFactorCopy","CAST(minimumWidth as double) as minimumWidthCopy","CAST(K as double) as KCopy","CAST(n as double) as nCopy")
    //    prepareBucketFinal.groupBy("bucketId").agg()

    //    variabel test1 digunakan untuk menampung rekord yang dijoin dengan ketentuan bucketId > bucketIdCopy
    //    hasil yang bakal dikeluarin adalah rekord yang nilainya gk diambil
    val forMapFunction1= prepareBucketFinal.crossJoin(prepareBucketCopy)
    //  .withColumn("listMinValue",lit(0))
    forMapFunction1.createOrReplaceTempView("prepareTable")
    val forMapFunction2= spark.sql("SELECT * FROM prepareTable WHERE bucketId>=bucketIdCopy")

    val forMapFunction3= forMapFunction2.select("bucketId","widthCopy")
      .groupBy("bucketId")
      .agg(concat_ws(",",collect_list("widthCopy")))
    //    lakukan join formapFunction3 (hanya punya bucketId dan listValue) dengan prepareBucketFinal
    val forMapFunction4= forMapFunction3.join(prepareBucketCopy,forMapFunction3("bucketId")===prepareBucketCopy("bucketIdCopy"))
    forMapFunction4.createOrReplaceTempView("prepareTable")

    val forMapFunction5= spark.sql("SELECT bucketId, `concat_ws(,, collect_list(widthCopy))` as listWidth, minIdCopy as minId,maxIdCopy as maxId, minValueCopy as minValue, maxValueCopy as maxValue, widthCopy as width, quadraticCoefficientCopy as quadraticCoefficient, scaleFactorCopy as scaleFactor, minimumWidthCopy as minimumWidth, KCopy as K,nCopy as n FROM prepareTable ORDER BY bucketId ASC")
    forMapFunction5.show()

    //    ubah prepareBucketCopy menjadi array
    val arrayString =prepareBucketFinal.collect.map(_.toSeq).flatten
    val stringToJoin = changeArrayToString(arrayString)

    val forMapFunction6 = forMapFunction5.withColumn("listBucketValue",lit(stringToJoin))
    forMapFunction6.createOrReplaceTempView("tempTable")

    val forMapFunction7 = spark.sql("SELECT bucketId,listWidth,minId,maxId,minValue,maxValue,width,quadraticCoefficient,ScaleFactor,listBucketValue,getQuadraticCoefficientFromWidth(listWidth,listBucketValue) as listQuadraticCoefficient,getScaleFactorFromWidth(listWidth,listBucketValue) as listScaleFactor,minimumWidth,K,n FROM tempTable ORDER BY bucketId ASC")
    forMapFunction7.createOrReplaceTempView("tempTable")

    val globalMinVal = spark.sql("SELECT MIN(value) as minPlaintext FROM values_df").head().get(0).toString.toDouble

    val forMapFunction8 = spark.sql(s"SELECT listWidth,minId,maxId,minValue,maxValue,width,quadraticCoefficient,ScaleFactor,listBucketValue,listQuadraticCoefficient,listScaleFactor,getPlaintextRemainQuadraticCoefAndScaleFactor(minValue,listWidth,listBucketValue,${globalMinVal})as minRemainListValue, getPlaintextRemainQuadraticCoefAndScaleFactor(maxValue,listWidth,listBucketValue,${globalMinVal})as maxRemainListValue FROM tempTable")
    forMapFunction8.createOrReplaceTempView("tempTable")

    //    listWidth:String, listQuadraticCoefficient:String, listScaleFactor:String, minPlaintext:Double, plaintext: Double, remainListValue:String
    val minValue = spark.sql("SELECT MIN(value) as minPlaintext FROM values_df")
    val minValueJoin  =   minValue.crossJoin(forMapFunction8)
    minValueJoin.createOrReplaceTempView("tempTable")
    val selectedMinValue = spark.sql("SELECT minPlaintext, quadraticCoefficient as minQuadraticCoefficient,ScaleFactor as minScaleFactor FROM tempTable WHERE minPlaintext>=minValue AND minPlaintext<= maxValue LIMIT 1")
    selectedMinValue.createOrReplaceTempView("tempTable")

    val minValueFinal = spark.sql("SELECT minPlaintext, predictValue(minPlaintext,minQuadraticCoefficient,minScaleFactor) as flatMinValue FROM tempTable")
    val forMapFunction9 = forMapFunction8.crossJoin(minValueFinal)
    forMapFunction9.createOrReplaceTempView("tempTable")

    val forMapFunctionFinalTemp1 = spark.sql("SELECT listWidth,minId,maxId,minValue,maxValue,width,quadraticCoefficient,ScaleFactor,listBucketValue,listQuadraticCoefficient,listScaleFactor, minRemainListValue, maxRemainListValue, getFlatValue(listWidth,listQuadraticCoefficient,listScaleFactor,minPlaintext,minValue,minRemainListValue)as minEncryptedValue, getFlatValue(listWidth,listQuadraticCoefficient,listScaleFactor,minPlaintext,maxValue,maxRemainListValue)as maxEncryptedValue, predictValue(minValue,quadraticCoefficient,scaleFactor)as minPredict,predictValue(maxValue,quadraticCoefficient,scaleFactor) as maxPredict,minPlaintext,flatMinValue FROM tempTable ORDER BY minId ASC")
    forMapFunctionFinalTemp1.createOrReplaceTempView("tempTable")
    val forMapFunctionFinalTemp2 = spark.sql("SELECT listWidth,minId,maxId,minValue,maxValue,width,quadraticCoefficient,ScaleFactor,listBucketValue,listQuadraticCoefficient,listScaleFactor,minRemainListValue,maxRemainListValue,minEncryptedValue,maxEncryptedValue,minPredict,maxPredict,minPlaintext,flatMinValue FROM tempTable ORDER BY minId ASC").withColumn("bucketId",row_number().over(Window.orderBy(lit(0))))
    forMapFunctionFinalTemp2.createOrReplaceTempView("tempTable")

    //    getEncryptedValue(minValue:Double,maxValue:Double,value:Double, minFlatValue:Double, maxFlatValue:Double, flatValue: Double):Double
    //    val key = spark.sql("SELECT bucketId,listWidth,minId,maxId,minValue,maxValue,width,quadraticCoefficient,ScaleFactor,listBucketValue,listQuadraticCoefficient,listScaleFactor, minRemainListValue, maxRemainListValue, minFlatValue, maxFlatValue, minPredict, maxPredict, minPlaintext, getEncryptedValue(minValue,maxValue,minValue,minFlatValue,maxFlatValue,minFlatValue) as minEncryptedValue, getEncryptedValue(minValue,maxValue,maxValue,minFlatValue,maxFlatValue,maxFlatValue) as maxEncryptedValue FROM tempTable ORDER BY bucketId ASC")
    val forMapFunctionFinal = spark.sql("SELECT bucketId,listWidth,minId,maxId,minValue,maxValue,width,quadraticCoefficient,ScaleFactor,listBucketValue,listQuadraticCoefficient,listScaleFactor,minRemainListValue,maxRemainListValue,minEncryptedValue,maxEncryptedValue,minPredict,maxPredict,minPlaintext,flatMinValue FROM tempTable ORDER BY minId")
    //    minRemain & maxRemain tidak perlu
    forMapFunctionFinal.coalesce(1)
      .write
      .option("header","true")
      .option("sep",",")
      .mode("overwrite")
      .csv(outputPath)

    sc.stop()

  }

  def setSplineFunction(spark: SparkSession, dataFrame: DataFrame): DataFrame = {
    dataFrame.createOrReplaceTempView("tempView1")
    var first = spark.sql("SELECT id,value,gradient,intercept,prediction,difference FROM tempView1 ORDER BY value ASC LIMIT 1")
    var last = spark.sql("SELECT id, value,gradient,intercept,prediction,difference FROM tempView1 ORDER BY value DESC LIMIT 1")
    var x0 = first.head().get(0).toString().toDouble
    var y0 = first.head().get(1).toString().toDouble
    var x1 = last.head().get(0).toString().toDouble
    var y1 = last.head().get(1).toString().toDouble
    print("x0: ",x0)
    print("x1: ",x1)
    print("y0: ",y0)
    print("y1: ",y1)
    var gradient = (y1-y0)/(x1-x0)*1.0
    var intercept = y0-gradient*x0*1.0
    var gradient_intercept_df = dataFrame.withColumn("gradient",lit(gradient)).withColumn("intercept",lit(intercept))
    gradient_intercept_df.createOrReplaceTempView("predTempView")
    var prediction_df = spark.sql("SELECT CAST(id as int),CAST(value as double), gradient,intercept, gradient*id+intercept as prediction, ABS(gradient*id+intercept-value) as difference FROM predTempView")
    return prediction_df
  }


  def getMaximalDifferenceId(spark: SparkSession, dataFrame: DataFrame):Int = {
    dataFrame.createOrReplaceTempView("tempView")
    dataFrame.show()
    var minDiff = spark.sql("SELECT id FROM tempView ORDER BY difference DESC LIMIT 1")
    print("maxID: ",minDiff.head().get(0).toString().toInt)
    return minDiff.head().get(0).toString().toInt
  }

  def getBelow(spark: SparkSession, dataFrame: DataFrame, thresholdId: Int):DataFrame = {
    dataFrame.withColumn("gradient",lit(0)).withColumn("intercept",lit(0)).withColumn("prediction",lit(0)).withColumn("difference",lit(0))
    dataFrame.createOrReplaceTempView("belowView")
    var below_df = spark.sql(s"SELECT id,value,gradient,intercept,prediction,difference FROM belowView WHERE id<=${thresholdId} ORDER BY id ASC")
    return below_df
  }

  def getUpper(spark: SparkSession, dataFrame: DataFrame, thresholdId: Int):DataFrame = {
    dataFrame.withColumn("gradient",lit(1)).withColumn("intercept",lit(1)).withColumn("prediction",lit(1)).withColumn("difference",lit(1))
    dataFrame.createOrReplaceTempView("upperView")
    var upper_df = spark.sql(s"SELECT id,value,gradient,intercept,prediction,difference FROM upperView WHERE id>=${thresholdId} ORDER BY id ASC")
    return upper_df
  }

  def bucketization(spark:SparkSession, dataFrame: DataFrame, threshold: Int): DataFrame= {
    var spline_df = setSplineFunction(spark,dataFrame)
    if(spline_df.count()<=threshold){
      spline_df.createOrReplaceTempView("TempTable")
      var minId = spark.sql("SELECT MIN(id) FROM TempTable").head().get(0).toString().toInt
      var maxId = spark.sql("SELECT MAX(id) FROM TempTable").head().get(0).toString().toInt
      var minValue = spark.sql("SELECT MIN(value) FROM TempTable").head().get(0).toString.toDouble
      var maxValue = spark.sql("SELECT MAX(value) FROM TempTable").head().get(0).toString.toDouble

      spline_df = spark.sql(s"SELECT id, value, gradient, intercept FROM TempTable")
        .withColumn("minId",lit(minId))
        .withColumn("maxId",lit(maxId))
        .withColumn("minValue",lit(minValue))
        .withColumn("maxValue",lit(maxValue))
        .withColumn("width",lit(maxValue-minValue))
      spline_df.write.option("header","true").mode("append").csv("bucket")
      return spline_df
    }
    else {
      val minimalThresholdId = getMaximalDifferenceId(spark, spline_df)
      var df_below = getBelow(spark,spline_df,minimalThresholdId)
      var df_upper = getUpper(spark,spline_df,minimalThresholdId)
      bucketization(spark,df_upper,threshold)
      bucketization(spark,df_below,threshold)
    }
  }

  def joinAllBucket(spark: SparkSession, path: String): DataFrame = {
    var df = spark.read.option("header", "true").csv("./bucket/*.csv")
    return df
  }

  def getQuadraticCoefficient(gradient: Double, intercept: Double):Double = {
    if(intercept == 0 ){
      return 0
    } else{
      return gradient/(2*intercept)
    }
  }

  def getScaleFactorMinimum(quadraticCoefficient:Double, width: Double): Double = {
    if(quadraticCoefficient>=0){
      return 2.0
    }
    else {
      return 2/(1+quadraticCoefficient*(2*width-1))
    }
  }

  def getMinimumWidth(quadraticCoefficient:Double, scaleFactorMinimum:Double,width:Double ):Double = {
    return scaleFactorMinimum*(quadraticCoefficient*width*width+width)
  }

  //  bisa juga ngehasilin/update dataframe
  def getK(spark: SparkSession,dataFrame: DataFrame): DataFrame = {
    dataFrame.createOrReplaceTempView("kTempTable")
    var k =  spark.sql("SELECT max(minimumWidth) as K FROM kTempTable").head().get(0).toString().toDouble
    var dataframeResult = dataFrame.withColumn("K",lit(k))
    return dataframeResult
  }

  def getN(spark: SparkSession, dataFrame: DataFrame):DataFrame = {
    dataFrame.createOrReplaceTempView("nTempTable")
    var filteredDf = spark.sql("SELECT minId as minIdTemp, COUNT(value) as n FROM nTempTable WHERE value>= minValue AND value<= maxValue GROUP BY minId ORDER BY minId ASC")
    filteredDf.createOrReplaceTempView("filteredTable")
    var joinedFilteredDf = dataFrame.join(filteredDf,dataFrame("minId") === filteredDf("minIdTemp"))
    joinedFilteredDf.selectExpr("gradient","intercept","minId","maxId","minValue","maxValue","width","quadraticCoefficient","scaleFactorMinimum","minimumWidth","K","n")
    joinedFilteredDf.createOrReplaceTempView("filteredTable")
    joinedFilteredDf = spark.sql("SELECT DISTINCT gradient, intercept, minId, maxId, minValue, maxValue, width, quadraticCoefficient, scaleFactorMinimum, minimumWidth, K, n FROM filteredTable ORDER BY minId ASC")
    return joinedFilteredDf
  }

  def getScaleFactor(K:Double,n:Double,quadraticCoefficient:Double, width:Double, scaleFactorMinimum: Double):Double = {
    var tempScaleFactor = (K*n)/(quadraticCoefficient*width*width+width)
    if(tempScaleFactor>scaleFactorMinimum){
      return tempScaleFactor
    }
    else {
      return scaleFactorMinimum
    }
  }

  def getQuadraticCoefficientFromWidth(listWidth:String, listBucketValue:String):String = {
    var i = 0
    var j = 0
    var status = 0
    var widthString = listWidth.split(",")
    var width = widthString.map(_.toDouble)
    var bucketString = listBucketValue.split(",")
    var bucket = bucketString.map(_.toDouble)
    var result = ""
    for (i <- 0 to width.size-1) {
      status = 0
      j = 0
      while (status==0) {
        var min = 0.0
        var max = 0.0
        var s = 0.0
        var curWidth = width(i)
        min=bucket(j*11+2)
        max=bucket(j*11+3)
        s=bucket(j*11+5)
        j+=1
        if(min<=curWidth && max>=curWidth){
          status=1
          if(i==0) {
            result=result+s.toString()
          }
          else {
            result= result+","+s.toString()
          }
        }
      }
    }
    return result
  }

  def getScaleFactorFromWidth(listWidth:String, listBucketValue:String):String = {
    var i = 0
    var j = 0
    var status = 0
    var widthString = listWidth.split(",")
    var width = widthString.map(_.toDouble)
    var bucketString = listBucketValue.split(",")
    var bucket = bucketString.map(_.toDouble)
    var result = ""
    for (i <- 0 to width.size-1) {
      status = 0
      j = 0
      while (status==0) {
        var min = 0.0
        var max = 0.0
        var z = 0.0
        var curWidth = width(i)
        min=bucket(j*11+2)
        max=bucket(j*11+3)
        z=bucket(j*11+6)
        j+=1
        if(min<=curWidth && max>=curWidth){
          status=1
          if(i==0) {
            result=result+z.toString()
          }
          else {
            result= result+","+z.toString()
          }
        }

      }
    }
    return result
  }

  def changeArrayToString(arrayString:Array[Any]):String = {
    var result = ""
    for (i <- 0 to arrayString.length-1) {
      if(i==0) {
        result = result+arrayString(i)
      }
      else{
        result = result+","+arrayString(i)
      }
    }
    return result
  }

  def getPlaintextRemainQuadraticCoefAndScaleFactor(plaintext:Double, listWidth:String, listBucketValue: String, minPlaintext:Double): String = {
    var splitWidthString = listWidth.split(",")
    var width = splitWidthString.map(_.toDouble)
    var splitBucketValueString = listBucketValue.split(",")
    var bucket = splitBucketValueString.map(_.toDouble)
    var remain = plaintext
    for(i <- 0 to width.size-1) {
      var temp = remain-width(i)
      if(temp>=minPlaintext) {
        remain = temp
      }
    }
    var i = 0
    var status = 0
    var result = ""
    while (status==0) {
      var min=bucket(i*11+2)
      var max=bucket(i*11+3)
      var s = bucket(i*11+5)
      var z = bucket(i*11+6)
      i+=1
      if(min<=remain && max>=remain){
        status=1
        result = remain.toString()+","+s+","+z
      }
    }
    return result
  }

  def predictValue(value:Double,quadraticCoefficient:Double,scaleFactor:Double):Double = {
    return scaleFactor*(quadraticCoefficient*value*value+value)
  }

  def getFlatValue(listWidth:String, listQuadraticCoefficient:String, listScaleFactor:String, minPlaintext:Double, plaintext: Double, remainListValue:String): Double = {
    var splitWidthString = listWidth.split(",")
    var splitWidth = splitWidthString.map(_.toDouble)
    var splitQuadraticCoefficientString = listQuadraticCoefficient.split(",")
    var splitQuadraticCoefficient = splitQuadraticCoefficientString.map(_.toDouble)
    var splitScaleFactorString = listScaleFactor.split(",")
    var splitScaleFactor = splitScaleFactorString.map(_.toDouble)
    var remainString = remainListValue.split(",")
    var remainValue = remainString(0).toDouble
    var remainQuadraticCoefficient = remainString(1).toDouble
    var remainScaleFactor = remainString(2).toDouble
    var pMin = minPlaintext
    var p = plaintext-pMin
    var fMin = pMin
    var f = fMin
    for (i <- 0 to splitWidth.size-1) {
      var temp = p - splitWidth(i)
      if(temp>=0) {
        p = temp
        f = f + (splitScaleFactor(i)*(splitQuadraticCoefficient(i)*splitWidth(i)*splitWidth(i)+splitWidth(i)))
      }
    }
    f = f + (remainScaleFactor*(remainQuadraticCoefficient*remainValue*remainValue+remainValue))
    return f
  }


  def getEncryptedValue(minValue:Double,maxValue:Double,value:Double, minFlatValue:Double, maxFlatValue:Double, flatValue: Double):Double = {
    var result = minFlatValue+ ((value-minValue)/(maxValue-minValue))*(maxFlatValue-minFlatValue)
    return result
  }

}
