package com.johnsnowlabs.nlp.annotators.eal

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{DoubleParam, IntParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

class ChineseTokenizer(override val uid: String) extends AnnotatorApproach[ChineseTokenizerModel]{

  def this() = this(Identifiable.randomUID("CHINESE_TOKENIZER"))

  override val description: String = "Chinese word segmentation without corpus"

  val maxWordLength = new IntParam(this, "maxWordLength", "Maximum word length")

  val minFrequency = new DoubleParam(this, "minFrequency", "Minimum frequency")

  val minEntropy = new DoubleParam(this, "minEntropy", "Minimum entropy")

  val minAggregation = new DoubleParam(this, "minEntropy", "Minimum aggregation")

  val wordSegmentMethod = new Param[String](this, "wordSegmentMethod", "How to treat a combination of shorter words: LONG, SHORT, ALL")

  val knowledgeBase = new ExternalResourceParam(this, "knowledgeBase", "Text fragment that will be used as knowledge base to segment a sentence with the words generated from it")

  def setWordSegmentMethod(method: String): this.type = {
    method.toUpperCase() match {
      case "LONG" => set(wordSegmentMethod, "LONG")
      case "SHORT" => set(wordSegmentMethod, "SHORT")
      case "ALL" => set(wordSegmentMethod, "ALL")
      case _ => throw new MatchError(s"Invalid WordSegmentMethod parameter. Must be either ${method.mkString("|")}")
    }
  }

  def setKnowledgeBase(path: String, readAs: ReadAs.Format = ReadAs.TEXT,
                       options: Map[String, String] = Map("format" -> "text")): this.type =
    set(knowledgeBase, ExternalResource(path, readAs, options))

  setDefault(maxWordLength -> 2)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): ChineseTokenizerModel = {
    //TODO: If knowledgeBase is null the text from each document row will be taken as knowledge base
    val externalKnowledgeBase = if (get(knowledgeBase).isDefined) {
      ResourceHelper.parseLines($(knowledgeBase))
    }.mkString(", ")
    val wordCandidates = generateCandidateWords(externalKnowledgeBase.toString)
    val tokenizer = new ChineseTokenizerModel()
    tokenizer
  }

  def generateCandidateWords(knowledgeBase: String) = {
    val pattern = "[\\\\s\\\\d,.<>/?:;\\'\\\"[\\\\]{}()\\\\|~!@#$%^&*\\\\-_=+a-zA-Z，。《》、？：；“”‘’｛｝【】（）…￥！—┄－]+"
    val cleanKnowledgeBase = knowledgeBase.replaceAll(pattern, " ")
    val suffixWithIndexes = indexOfSortedSuffix(cleanKnowledgeBase)
    val wordCandidates = suffixWithIndexes
      .groupBy(_._1).mapValues(_.map(_._2))
      .map(suffixInfo => {
        val leftNeighbors = suffixInfo._2.map(indexRange => cleanKnowledgeBase.slice(indexRange._1 - 1, indexRange._1))
        val rightNeighbors = suffixInfo._2.map(indexRange => cleanKnowledgeBase.slice(indexRange._2, indexRange._2 + 1))
        WordInfo(suffixInfo._1, suffixInfo._2.size, leftNeighbors.mkString(""), rightNeighbors.mkString(""), 0)
      })
    wordCandidates
  }

  case class WordInfo(text: String, frequency: Int, leftNeighbors: String, rightNeighbors: String, aggregation: Int)

  private def indexOfSortedSuffix(cleanKnowledgeBase: String) = {
    val indexes = 0 until cleanKnowledgeBase.length

    val suffixInfo = indexes.flatMap{index =>
      val suffixIndexes = getSuffixIndexes(index, cleanKnowledgeBase)
      val characters = suffixIndexes.map(suffixIndex => cleanKnowledgeBase.slice(suffixIndex._1, suffixIndex._2))
      characters zip suffixIndexes
    }
    suffixInfo.sortBy(_._1)
  }

  def getSuffixIndexes(index: Int, sentence: String): List[(Int, Int)] = {
    val begin = index + 1
    val end = (begin + $(maxWordLength)).min(sentence.length + 1)
    val secondElement = (begin until end).toList
    val firstElement = List.fill(secondElement.length)(index)
    firstElement zip secondElement
  }

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)
  override val outputAnnotatorType: AnnotatorType = TOKEN
}
