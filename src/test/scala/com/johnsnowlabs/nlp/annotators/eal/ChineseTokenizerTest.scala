package com.johnsnowlabs.nlp.annotators.eal

import com.johnsnowlabs.nlp.{DocumentAssembler, SparkAccessor}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import org.apache.spark.ml.Pipeline
import org.scalatest.FlatSpec

class ChineseTokenizerTest extends FlatSpec {

  val maxWordLength = 2

  import SparkAccessor.spark.implicits._

  private val testDataSet = Seq(
    "十四不是四十"
  ).toDS.toDF("text")

  private val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  private val sentence = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

  private val chineseTokenizer = new ChineseTokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")
    .setKnowledgeBase("src/test/resources/tokenizer/sample_chinese_doc.txt")

  "A ChineseTokenizer" should "tokenize words" in {

    testDataSet.show(1, false)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        chineseTokenizer
      ))

    val tokenizerDataSet = pipeline.fit(testDataSet).transform(testDataSet)

    tokenizerDataSet.show(1, false)

  }

}
